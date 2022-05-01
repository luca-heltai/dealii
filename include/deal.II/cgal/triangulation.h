// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

#ifndef dealii_cgal_triangulation_h
#define dealii_cgal_triangulation_h

#include <deal.II/base/config.h>

#include <deal.II/grid/tria.h>

#include <deal.II/cgal/utilities.h>

#ifdef DEAL_II_WITH_CGAL
#  include <CGAL/Surface_mesh.h>
#  include <CGAL/Triangulation_3.h>

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  /**
   * Build any CGAL triangulation from a list of deal.II points.
   *
   * The CGAL library offers several types of triangulations, including surface
   * triangulations, cell complexes, Delaunay triangulations in two and three
   * dimensions, and many others. For simple cases, all of these triangulations
   * allow you to incrementally build it from a list of points. This function
   * provides a convenient way to add a vector of deal.II points to any CGAL
   * triangulation that admits the insertion of new points via iterators.
   *
   * More information on the available CGAL triangulation classes is available
   * at https://doc.cgal.org/latest/Triangulation_2/index.html for two
   * dimensional triangulations, and at
   * https://doc.cgal.org/latest/Triangulation_3/index.html for three
   * dimensional triangulations.
   *
   * Notice that CGAL distinguishes between a triangulation and a polygonal or
   * polyhedral mesh. Generally speaking, a triangulation is made of simplices,
   * whereas a polygonal or polyhedral mesh is made of general polygons or
   * general polyhedrons. While CGAL implements the two concepts in a similar
   * fashion using half-edge data structures, some optimizations are performed
   * for triangulations, where only triangles or tetrahedra are used.
   *
   * @param[in] points The input points to build the triangulation from.
   * @param[out] triangulation The output triangulation.
   */
  template <int spacedim, typename CGALTriangulation>
  void
  add_points_to_cgal_triangulation(const std::vector<Point<spacedim>> &points,
                                   CGALTriangulation &triangulation);

#  ifndef DOXYGEN
  // Template implementation

  template <int spacedim, typename CGALTriangulation>
  void
  add_points_to_cgal_triangulation(const std::vector<Point<spacedim>> &points,
                                   CGALTriangulation &triangulation)
  {
    Assert(triangulation.is_valid(),
           ExcMessage(
             "The triangulation you pass to this function should be a valid "
             "CGAL triangulation."));
    using CGALPoint = typename CGALTriangulation::Point;
    std::vector<CGALPoint> cgal_points(points.size());
    std::transform(points.begin(),
                   points.end(),
                   cgal_points.begin(),
                   [](const auto &p) {
                     return CGALWrappers::to_cgal<CGALPoint>(p);
                   });

    triangulation.insert(cgal_points.begin(), cgal_points.end());
    Assert(triangulation.is_valid(),
           ExcMessage(
             "The Triangulation is no longer valid after inserting the points. "
             "Bailing out."));
  }
#  endif
} // namespace CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
