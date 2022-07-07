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

#include <deal.II/grid/grid_tools_cache.h>


#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/non_matching/quadrature_overlapped_grids.h>

DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
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
    return Quadrature<1>();
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



  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 1, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<1, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<1, 3> &                              mapping1)
  {
#  if defined(CGAL_GEQ_515)
    const std::vector<std::array<Point<3>, 2>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<1>(degree).mapped_quadrature(vec_of_simplices);
#  else
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
#  endif
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
#  if defined(CGAL_GEQ_515)
    const std::vector<std::array<Point<3>, 4>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<2>(degree).mapped_quadrature(vec_of_simplices);
#  else
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
#  endif
  }



  template <>
  Quadrature<3>
  compute_quadrature_on_intersection<3, 3, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<3, 3>::cell_iterator &cell1,
    const unsigned int                                 degree,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<3, 3> &                              mapping1)
  {
    // return CGALWrappers::compute_quadrature_on_intersection(
    //   cell0, cell1, degree, mapping0, mapping1);

    const std::vector<std::array<Point<3>, 4>> &vec_of_simplices =
      CGALWrappers::compute_intersection_of_cells(cell0,
                                                  cell1,
                                                  mapping0,
                                                  mapping1);
    return QGaussSimplex<3>(degree).mapped_quadrature(vec_of_simplices);
  }



  template <int dim0, int dim1, int spacedim>
  std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                         typename Triangulation<dim1, spacedim>::cell_iterator,
                         Quadrature<spacedim>>>
  collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const unsigned int                      degree,
    const double                            tol)
  {
    AssertThrow(
      dim1 <= dim0,
      ExcMessage(
        "Intrinsic dimension of the immersed object must be smaller than dim0."));
    AssertThrow(degree > 0, ExcMessage("Invalid quadrature degree."));
    Assert((dim1 <= dim0) && (dim0 <= spacedim),
           ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
    std::vector<
      std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                 typename Triangulation<dim1, spacedim>::cell_iterator,
                 Quadrature<spacedim>>>
      cells_with_quadratures;

    const auto &space_tree =
      space_cache.get_locally_owned_cell_bounding_boxes_rtree();

    // The immersed tree *must* contain all cells, also the non-locally owned
    // ones.
    const auto &immersed_tree = immersed_cache.get_cell_bounding_boxes_rtree();

    // references to triangulations' info (cp cstrs marked as delete)
    const auto &mapping0 = space_cache.get_mapping();
    const auto &mapping1 = immersed_cache.get_mapping();
    namespace bgi        = boost::geometry::index;
    // Whenever the BB space_cell intersects the BB of an embedded cell,
    // store the space_cell in the set of intersected_cells
    for (const auto &[immersed_box, immersed_cell] : immersed_tree)
      {
        for (const auto &[space_box, space_cell] :
             space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box)))
          {
            const auto test_intersection = compute_quadrature_on_intersection(
              space_cell, immersed_cell, degree, mapping0, mapping1);

            const auto & weights = test_intersection.get_weights();
            const double area =
              std::accumulate(weights.begin(), weights.end(), 0.0);
            if (area > tol) // non-trivial intersection
              {
                cells_with_quadratures.push_back(std::make_tuple(
                  space_cell, immersed_cell, test_intersection));
              }
          }
      }
    return cells_with_quadratures;
  }

  //#  include "quadrature_overlapped_grids.inst"
  template std::vector<std::tuple<typename Triangulation<2, 2>::cell_iterator,
                                  typename Triangulation<1, 2>::cell_iterator,
                                  Quadrature<2>>>
  collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<2, 2> &space_cache,
    const GridTools::Cache<1, 2> &immersed_cache,
    const unsigned int            degree,
    const double                  tol);

  template std::vector<std::tuple<typename Triangulation<2, 2>::cell_iterator,
                                  typename Triangulation<2, 2>::cell_iterator,
                                  Quadrature<2>>>
  collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<2, 2> &space_cache,
    const GridTools::Cache<2, 2> &immersed_cache,
    const unsigned int            degree,
    const double                  tol);

  template std::vector<std::tuple<typename Triangulation<3, 3>::cell_iterator,
                                  typename Triangulation<3, 3>::cell_iterator,
                                  Quadrature<3>>>
  collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<3, 3> &space_cache,
    const GridTools::Cache<3, 3> &immersed_cache,
    const unsigned int            degree,
    const double                  tol);
} // namespace NonMatching

DEAL_II_NAMESPACE_CLOSE
#endif
