// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by the deal.II authors
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

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/non_matching/quadrature_overlapped.h>

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  template <int dim0, int dim1, int spacedim>
  Quadrature<spacedim>
  compute_quadrature_on_intersection(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                           degree,
    const Mapping<dim0, spacedim> &                              mapping0,
    const Mapping<dim1, spacedim> &                              mapping1)
  {
    Assert((dim0 <= dim1) && (dim1 <= spacedim),
           ExcMessage("Invalid combination of dimension templates."));
    (void)cell0;
    (void)cell1;
    (void)degree;
    (void)mapping0;
    (void)mapping1;
    return Quadrature<spacedim>{};
  }



  template <>
  Quadrature<1>
  compute_quadrature_on_intersection<1, 1, 1>(
    const typename Triangulation<1, 1>::cell_iterator &cell0,
    const typename Triangulation<1, 1>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<1, 1> &                              mapping0,
    const Mapping<1, 1> &                              mapping1)
  {
    Assert(false, ExcNotImplemented());
    (void)cell0;
    (void)cell1;
    (void)degree;
    (void)mapping0;
    (void)mapping1;
    return Quadrature<1>{};
  }



  template <>
  Quadrature<2>
  compute_quadrature_on_intersection<2, 1, 2>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<1, 2>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<1, 2> &                              mapping1)
  {
    const std::vector<std::array<Point<2>, 2>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);


    return QGaussSimplex<1>(degree).mapped_quadrature(vec_of_simplices);
  }



  template <>
  Quadrature<2>
  compute_quadrature_on_intersection<2, 2, 2>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<2, 2>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<2, 2> &                              mapping1)
  {
    const std::vector<std::array<Point<2>, 3>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<2>(degree).mapped_quadrature(vec_of_simplices);
  }



#  if defined(CGAL_GEQ_515)
  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 1, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<1, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<1, 3> &                              mapping1)
  {
    const std::vector<std::array<Point<3>, 2>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<1>(degree).mapped_quadrature(vec_of_simplices);
  }



  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 2, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<2, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<2, 3> &                              mapping1)
  {
    const std::vector<std::array<Point<3>, 4>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<2>(degree).mapped_quadrature(vec_of_simplices);
  }
#  else
  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 1, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<1, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<1, 3> &                              mapping1)
  {
    (void)cell0;
    (void)cell1;
    (void)degree;
    (void)mapping0;
    (void)mapping1;
    Assert(
      false,
      ExcMessage(
        "This function requires a version of CGAL greater of equal than 5.1.5."));
    return Quadrature<3>();
  }



  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 2, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<2, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<2, 3> &                              mapping1)
  {
    (void)cell0;
    (void)cell1;
    (void)degree;
    (void)mapping0;
    (void)mapping1;
    Assert(
      false,
      ExcMessage(
        "This function requires a version of CGAL greater of equal than 5.1.5."));
    return Quadrature<3>();
  }
#  endif


  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 3, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<3, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<3, 3> &                              mapping1)
  {
    return CGALWrappers::compute_quadrature_on_intersection(
      cell0, cell1, degree, mapping0, mapping1);
    // using K         = CGAL::Exact_predicates_inexact_constructions_kernel;
    // using CGALPoint = CGAL::Point_3<K>;
    // using CGALTriangulation = CGAL::Triangulation_3<K>;

    // CGAL::Surface_mesh<CGALPoint> surface_1, surface_2, out_surface;
    // dealii_cell_to_cgal_surface_mesh(cell0, mapping0, surface_1);
    // dealii_cell_to_cgal_surface_mesh(cell1, mapping1, surface_2);
    // // They have to be triangle meshes
    // CGAL::Polygon_mesh_processing::triangulate_faces(surface_1);
    // CGAL::Polygon_mesh_processing::triangulate_faces(surface_2);

    // compute_boolean_operation(surface_1,
    //                           surface_2,
    //                           BooleanOperation::compute_intersection,
    //                           out_surface);
    // CGAL::Surface_mesh<CGALPoint> dummy;
    // CGALTriangulation             tr;
    // CGAL::convex_hull_3(out_surface.points().begin(),
    //                     out_surface.points().end(),
    //                     dummy);
    // tr.insert(dummy.points().begin(), dummy.points().end());
    // std::vector<std::array<Point<3>, 4>> vec_of_simplices;

    // for (const auto &cell : tria.finite_cell_handles())
    //   {
    //     const auto &tet = tria.tetrahedron(cell);
    //     vec_of_simplices.push_back(
    //       {{cgal_point_to_dealii_point<3>(tet.vertex(0)),
    //         cgal_point_to_dealii_point<3>(tet.vertex(1)),
    //         cgal_point_to_dealii_point<3>(tet.vertex(2)),
    //         cgal_point_to_dealii_point<3>(tet.vertex(3))}});
    //   }

    // return QGaussSimplex<3>(degree).mapped_quadrature<3,
    // 3>(vec_of_simplices);
  }

} // namespace NonMatching
#  include "quadrature_overlapped.inst"

DEAL_II_NAMESPACE_CLOSE
#endif
