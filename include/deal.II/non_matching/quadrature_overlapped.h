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

#ifndef dealii_non_matching_quadrature_overlapped_h
#define dealii_non_matching_quadrature_overlapped_h

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
   * A specialization of the function above when the BooleanOperation is an
   * intersection. The rationale behind this specialization is that deal.II
   * affine cells are convex sets, and as the intersection of convex sets is
   * itself convex, this function internally exploits this to use a cheaper way
   * to mesh the inside.
   *
   *
   * @param [in] cell0 A cell_iterator to the first deal.II cell.
   * @param [in] cell1 A cell_iterator to the second deal.II cell.
   * @param [in] mapping0 Mapping object for the first cell.
   * @param [in] mapping1 Mapping object for the first cell.
   * @param [in] degree The degree of accuracy you wish to get for the global quadrature formula.
   * @return [out] Quadrature<spacedim> The global quadrature rule on the polygon/polyhedron.
   */
  template <int dim0, int dim1, int spacedim>
  dealii::Quadrature<spacedim>
  compute_quadrature_on_intersection(
    const typename dealii::Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename dealii::Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                                   degree,
    const Mapping<dim0, spacedim> &mapping0,
    const Mapping<dim1, spacedim> &mapping1);


} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif
#endif
