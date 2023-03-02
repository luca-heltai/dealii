// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2022 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <deal.II/base/parsed_function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/cgal/point_conversion.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <CGAL/Plane_3.h>
#include <CGAL/squared_distance_2.h>
#include <deal.II/cgal/utilities.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/base/exceptions.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polyhedral_mesh_domain_3.h>
#include <iostream>


using namespace dealii;
namespace bgi = boost::geometry::index;

namespace NonMatchingUtilities
{
  using CGALKernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
  using CGALPlane3    = CGALKernel::Plane_3;
  using CGALTriangle2 = CGALKernel::Triangle_2;
  using CGALTriangle3 = CGALKernel::Triangle_3;
  using CGALPoint2    = CGALKernel::Point_2;
  using CGALPoint3    = CGALKernel::Point_3;
  using CGALMesh      = CGAL::Surface_mesh<CGALPoint3>;
  using CGALSegment2  = CGALKernel::Segment_2;
  using CGALSegment3  = CGALKernel::Segment_3;
  using CGALTetra     = CGALKernel::Tetrahedron_3;

  struct FaceInfo2
  {
    FaceInfo2()
    {}
    int  nesting_level;
    bool in_domain()
    {
      return nesting_level % 2 == 1;
    }
  };


  typedef CGAL::Triangulation_vertex_base_2<CGALKernel>                    Vb;
  typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, CGALKernel> Fbb;
  typedef CGAL::Constrained_triangulation_face_base_2<CGALKernel, Fbb>     Fb;
  typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                     TDS;
  typedef CGAL::Exact_predicates_tag                                       Itag;
  typedef CGAL::Constrained_Delaunay_triangulation_2<CGALKernel, TDS, Itag> CDT;
  typedef CDT::Point                  Point2;
  typedef CGAL::Polygon_2<CGALKernel> Polygon_2;
  typedef CDT::Face_handle            Face_handle;



  void mark_domains(CDT &                 ct,
                    Face_handle           start,
                    int                   index,
                    std::list<CDT::Edge> &border)
  {
    if (start->info().nesting_level != -1)
      {
        return;
      }
    std::list<Face_handle> queue;
    queue.push_back(start);
    while (!queue.empty())
      {
        Face_handle fh = queue.front();
        queue.pop_front();
        if (fh->info().nesting_level == -1)
          {
            fh->info().nesting_level = index;
            for (int i = 0; i < 3; i++)
              {
                CDT::Edge   e(fh, i);
                Face_handle n = fh->neighbor(i);
                if (n->info().nesting_level == -1)
                  {
                    if (ct.is_constrained(e))
                      border.push_back(e);
                    else
                      queue.push_back(n);
                  }
              }
          }
      }
  }
  // explore set of facets connected with non constrained edges,
  // and attribute to each such set a nesting level.
  // We start from facets incident to the infinite vertex, with a nesting
  // level of 0. Then we recursively consider the non-explored facets incident
  // to constrained edges bounding the former set and increase the nesting level
  // by 1. Facets in the domain are those with an odd nesting level.
  void mark_domains(CDT &cdt)
  {
    for (CDT::Face_handle f : cdt.all_face_handles())
      {
        f->info().nesting_level = -1;
      }
    std::list<CDT::Edge> border;
    mark_domains(cdt, cdt.infinite_face(), 0, border);
    while (!border.empty())
      {
        CDT::Edge e = border.front();
        border.pop_front();
        Face_handle n = e.first->neighbor(e.second);
        if (n->info().nesting_level == -1)
          {
            mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
          }
      }
  }



  namespace LM
  {
    template <int dim, int spacedim, typename VectorType>
    double compute_H_12_norm(const GridTools::Cache<dim, spacedim> &cache,
                             const DoFHandler<dim, spacedim> &      dh,
                             const FiniteElement<dim, spacedim> &   fe,
                             const Function<spacedim> &             solution,
                             const VectorType &                     u,
                             const Quadrature<dim> &                quad)
    {
      Assert(dh.n_dofs() > 0, ExcMessage("DoFhandler is empty."));
      Assert(order > 0,
             ExcMessage("Order of quadrature rule must be larger than 0."));
      Assert(u.size() > 0, ExcMessage("Solution vector is not valid."));

      if (fe.degree == 0 && dynamic_cast<const QGaussLobatto<dim> *>(&quad))
        Assert(
          false,
          ExcMessage(
            "Gauss Lobatto quadrature should not be used with a DG_Q(0) space"));

      const auto &            mapping = cache.get_mapping();
      FEValues<dim, spacedim> fe_values(mapping,
                                        fe,
                                        quad,
                                        update_values |
                                          update_quadrature_points |
                                          update_JxW_values);

      double h = GridTools::minimal_cell_diameter(cache.get_triangulation(),
                                                  cache.get_mapping());

      double                                  local_error = 0.;
      Vector<typename VectorType::value_type> errors(
        cache.get_triangulation().n_active_cells());
      std::vector<typename VectorType::value_type> local_values(quad.size());
      const auto &                                 qpoints = quad.get_points();
      std::vector<Point<spacedim>>                 real_qpoints(qpoints.size());
      unsigned int                                 i = 0;
      for (const auto &cell : dh.active_cell_iterators())
        {
          local_error = 0.;
          h           = cell->diameter();

          fe_values.reinit(cell);
          fe_values.get_function_values(u, local_values);

          i = 0;
          for (const auto &p : quad.get_points())
            {
              real_qpoints[i] = mapping.transform_unit_to_real_cell(cell, p);
              ++i;
            }

          for (const auto q : fe_values.quadrature_point_indices())
            {
              const double diff =
                local_values[q] - solution.value(real_qpoints[q]);
              local_error += (diff * diff * fe_values.JxW(q)) * h;
            }
          errors[cell->active_cell_index()] = std::sqrt(local_error);
        }

      const double local = errors.l2_norm();
      return local;
    }
  } // namespace LM



  template <int dim, typename RTree, typename Tester = nullptr_t>
  class DiscreteLevelSet : public Function<dim>
  {
  public:
    DiscreteLevelSet() = default;

    DiscreteLevelSet(RTree *tree, CDT &tria)
      : rtree(tree)
      , tr(tria)
    {
      Assert(dim == 2, ExcMessage("This constructor works in 2D only."));
    }

    DiscreteLevelSet(RTree *tree, const Tester &tester)
      : rtree(tree)
      , is_in_tester(tester)
    {
      Assert(dim == 3, ExcMessage("This constructor works in 3D only."));
      is_in_tester(CGALPoint3(0.5, 0.5, 0.5));
    }

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

  private:
    RTree *rtree;
    CDT    tr;
    Tester is_in_tester;
  };


  template <int dim, typename RTree, typename Tester>
  double DiscreteLevelSet<dim, RTree, Tester>::value(
    const Point<dim> & p,
    const unsigned int component) const
  {
    (void)component;

    if constexpr (dim == 2)
      {
        // First, locate the point w.r.t the triangulation
        typename CDT::Face_handle fh =
          tr.locate(CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(p));

        // Find the closest BBox(es) to this point, then compute the distance
        const auto &cgal_point =
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(p);

        std::vector<double> distances;
        for (const auto &[box, cell] :
             *rtree | bgi::adaptors::queried(bgi::nearest(p, 3)))
          {
            CGALSegment2 cgal_segm(
              CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, dim>(
                cell->vertex(0)),
              CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, dim>(
                cell->vertex(1)));

            // Compute the distance from each one of the closest elements of the
            // mesh
            distances.push_back(std::sqrt(
              CGAL::to_double(CGAL::squared_distance(cgal_point, cgal_segm))));
          }

        const auto &min_dist =
          *std::min_element(distances.begin(), distances.end());
        return fh->info().in_domain() ? -min_dist : +min_dist;
      }
    else if constexpr (dim == 3)
      {
        const auto &cgal_point =
          CGALWrappers::dealii_point_to_cgal_point<CGALPoint3, 3>(p);

        std::vector<double> distances;
        for (const auto &[box, cell] :
             *rtree | bgi::adaptors::queried(bgi::nearest(p, 3)))
          {
            // Create the plane passing through 3 pts
            CGALPlane3 cgal_plane(
              CGALWrappers::dealii_point_to_cgal_point<CGALPoint3, 3>(
                cell->vertex(0)),
              CGALWrappers::dealii_point_to_cgal_point<CGALPoint3, 3>(
                cell->vertex(1)),
              CGALWrappers::dealii_point_to_cgal_point<CGALPoint3, 3>(
                cell->vertex(3)));

            // Compute the distance from each one of the closest elements of the
            // mesh
            distances.push_back(std::sqrt(
              CGAL::to_double(CGAL::squared_distance(cgal_point, cgal_plane))));

            const auto &min_dist =
              *std::min_element(distances.begin(), distances.end());

            return *is_in_tester(
                     CGALWrappers::dealii_point_to_cgal_point<CGALPoint3, 3>(
                       p)) ?
                     -min_dist :
                     +min_dist;
          }
      }
    else
      {
        Assert(false, ExcNotImplemented());
      }
    return -1.;
  }

} // namespace NonMatchingUtilities
