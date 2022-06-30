// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

#ifndef dealii_cgal_intersections_h
#define dealii_cgal_intersections_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/cgal/point_conversion.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/grid/tria.h>

#  include <CGAL/Cartesian.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#  include <CGAL/Kernel_traits.h>
#  include <CGAL/Segment_3.h>
#  include <CGAL/Simple_cartesian.h>
#  include <CGAL/Tetrahedron_3.h>
#  include <CGAL/Triangle_2.h>
#  include <CGAL/Triangle_3.h>
#  include <CGAL/Triangulation_3.h>

//#  include <CGAL/tetrahedral_remeshing.h> REQUIRES CGAL_VERSION>=5.1.5

#  include <fstream>
#  include <type_traits>



DEAL_II_NAMESPACE_OPEN

using K             = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGALTriangle2 = K::Triangle_2;
using CGALTriangle3 = K::Triangle_3;
using CGALPoint2    = K::Point_2;
using CGALPoint3    = K::Point_3;
using CGALSegment2  = K::Segment_2;
using CGALSegment3  = K::Segment_3;
using CGALTetra     = K::Tetrahedron_3;
namespace CGALWrappers
{
  namespace internal
  {
    // Collection of utilities that compute intersection between simplices
    // identified by array of points. The return type is the one of
    // CGAL::intersection(), i.e. a boost::optional<boost::variant<>>.
    // Intersection between 2D and 3D objects and 1D/3D objects are available
    // only with versions greater or equal than 5.1.5, hence the corresponding
    // functions are guarded by #ifdef directives.

    // triangle,triangle
    boost::optional<boost::variant<CGALPoint2,
                                   CGALSegment2,
                                   CGALTriangle2,
                                   std::vector<CGALPoint2>>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex);

    // line,triangle
    boost::optional<boost::variant<CGALPoint2, CGALSegment2>>
    compute_intersection(const std::array<Point<2>, 2> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex);

#  if defined(CGAL_GEQ_515)
    // line, tetra
    boost::optional<boost::variant<CGALPoint3, CGALSegment3>>
    compute_intersection(const std::array<Point<3>, 2> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex);

    // triangle, tetra
    boost::optional<boost::variant<CGALPoint3,
                                   CGALSegment3,
                                   CGALTriangle3,
                                   std::vector<CGALPoint3>>>
    compute_intersection(const std::array<Point<3>, 3> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex);
#  endif

  } // namespace internal

} // namespace CGALWrappers


DEAL_II_NAMESPACE_CLOSE



#endif
#endif
