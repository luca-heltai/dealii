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
#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_q.h>
#include <CGAL/squared_distance_2.h>
#include <deal.II/cgal/utilities.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/base/exceptions.h>

// using K           =
// CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt;


using namespace dealii;
namespace NonMatchingUtilities
{
  namespace LM
  {
    using CGALKernel    = CGAL::Exact_predicates_inexact_constructions_kernel;
    using CGALTriangle2 = CGALKernel::Triangle_2;
    using CGALTriangle3 = CGALKernel::Triangle_3;
    using CGALPoint2    = CGALKernel::Point_2;
    using CGALPoint3    = CGALKernel::Point_3;
    using CGALSegment2  = CGALKernel::Segment_2;
    using CGALSegment3  = CGALKernel::Segment_3;
    using CGALTetra     = CGALKernel::Tetrahedron_3;

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
} // namespace NonMatchingUtilities



template <int dim, typename RTree>
class DiscreteLevelSet
{
public:
  DiscreteLevelSet() = default;

  DiscreteLevelSet(const RTree &tree)
    : rtree(std::move(tree))
  {
    Assert(dim == 2, ExcMessage("Tested so far only in 2D."));
  }

  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override
  {
    // Find the closest BBox(es) to this point, then compute the distance
    const auto &cgal_point =
      CGALWrappers::dealii_point_to_cgal_point<CGALPoint2, 2>(p);
    boost::geometry::index::query(rtree,
                                  boost::geometry::index::nearest(p),
                                  std::back_inserter(closest_elements));
    // Compute the distance each element

    for (unsigned int i = 0; i < closest_elements.size(); ++i)
      {
        distances.push_back(CGAL::to_double(
          CGAL::squared_distance(cgal_point,
                                 CGALSegment2{closest_elements[i].vertex(0),
                                              closest_elements[i].vertex(1)})));
      }

    return *std::min_element(distances.begin(), distances.end());
  }

private:
  RTree                   rtree;
  std::vector<Point<dim>> closest_elements;
  std::vector<double>     distances;
};
