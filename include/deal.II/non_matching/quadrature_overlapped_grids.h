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

#ifndef dealii_non_matching_quadrature_overlapped_grids_h
#define dealii_non_matching_quadrature_overlapped_grids_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/cgal/intersections.h>
#  include <deal.II/cgal/triangulation.h>

#  include <type_traits>


DEAL_II_NAMESPACE_OPEN

namespace NonMatching
{
  /**
   *
   * Given two cell_iterator objects to two intersecting cells, computes a
   * Quadrature rule of degree `degree` formula by splitting the intersection
   * into simplices.
   *
   * @param [in] cell0 A cell_iterator to the first deal.II cell.
   * @param [in] cell1 A cell_iterator to the second deal.II cell.
   * @param [in] degree The degree of accuracy you wish to get for the global quadrature formula.
   * @param [in] mapping0 Mapping object for the first cell.
   * @param [in] mapping1 Mapping object for the first cell.
   * @return [out] Quadrature<spacedim> The global quadrature rule on the polygon/polyhedron.
   */
  template <int dim0, int dim1, int spacedim>
  Quadrature<spacedim>
  compute_quadrature_on_intersection(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                           degree,
    const Mapping<dim0, spacedim> &                              mapping0,
    const Mapping<dim1, spacedim> &                              mapping1);



  /**
   *
   * Given two cached, arbitrarily overlapped grids, the following function
   * computes Quadrature rules on the intersection of the embedding grid with
   * the embedded one, of degree `degree`. The return type is a vector of tuples
   * `v`, where `v[i][0]` is an iterator to a cell of the embedding grid,
   * `v[i][1]` is an iterator to a cell of the embedded grid,
   * `v[i][2]` is a Quadrature formula to integrate over the intersection of the
   * two.
   *
   * The last parameter `tol` defaults to 1e-6, and can be used to discard small
   * intersections.
   *
   * @note This function calls compute_quadrature_on_intersection().
   *
   * @param [in] space_cache First cached triangulation.
   * @param [in] immersed_cache Second cached triangulation.
   * @param [in] degree The degree of accuracy of each quadrature formula.
   * @param [in] tol Tolerance used to discard small intersections.
   * @return std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
   * typename Triangulation<dim1, spacedim>::cell_iterator,
   * Quadrature<spacedim>>>.
   */
  template <int dim0, int dim1, int spacedim>
  std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                         typename Triangulation<dim1, spacedim>::cell_iterator,
                         Quadrature<spacedim>>>
  collect_quadratures_on_overlapped_grids(
    const GridTools::Cache<dim0, spacedim> &space_cache,
    const GridTools::Cache<dim1, spacedim> &immersed_cache,
    const unsigned int                      degree,
    const double                            tol = 1e-12);


} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif
#endif
