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
    return CGALWrappers::compute_quadrature_on_intersection(
      cell0, cell1, degree, mapping0, mapping1);
  }

} // namespace NonMatching
//#  include "quadrature_overlapped_grids.inst"

DEAL_II_NAMESPACE_CLOSE
#endif
