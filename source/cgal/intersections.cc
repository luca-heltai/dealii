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

#include <deal.II/base/config.h>

#include <deal.II/cgal/intersections.h>

#ifdef DEAL_II_WITH_CGAL

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  namespace internal
  {
    boost::optional<boost::variant<CGALPoint2,
                                   CGALSegment2,
                                   CGALTriangle2,
                                   std::vector<CGALPoint2>>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex)
    {
      CGALPoint2 p1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[0]);
      CGALPoint2 q1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[1]);
      CGALPoint2 r1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[2]);
      CGALTriangle2 triangle1{p1, q1, r1};

      CGALPoint2 p2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[0]);
      CGALPoint2 q2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[1]);
      CGALPoint2 r2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[2]);
      CGALTriangle2 triangle2{p2, q2, r2};

      return CGAL::intersection(triangle1, triangle2);
    }



    boost::optional<boost::variant<CGALPoint2, CGALSegment2>>
    compute_intersection(const std::array<Point<2>, 2> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex)
    {
      CGALPoint2 p1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[0]);
      CGALPoint2 q1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[1]);
      CGALSegment2 segm{p1, q1};

      CGALPoint2 p2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[0]);
      CGALPoint2 q2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[1]);
      CGALPoint2 r2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[2]);
      CGALTriangle2 triangle{p2, q2, r2};

      return CGAL::intersection(segm, triangle);
    }

#  if defined(CGAL_GEQ_515)


    // line, tetra
    boost::optional<boost::variant<CGALPoint3, CGALSegment3>>
    compute_intersection(const std::array<Point<3>, 2> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex)
    {
      CGALPoint3 p1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[0]);
      CGALPoint3 q1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[1]);
      CGALSegment3 segm{p1, q1};

      CGALPoint3 p2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[0]);
      CGALPoint3 q2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[1]);
      CGALPoint3 r2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[2]);
      CGALPoint3 s2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[3]);
      CGALTetra  tetra{p2, q2, r2, s2};

      return CGAL::intersection(segm, tetra);
    }



    // triangle, tetra
    boost::optional<boost::variant<CGALPoint3,
                                   CGALSegment3,
                                   CGALTriangle3,
                                   std::vector<CGALPoint3>>>
    compute_intersection(const std::array<Point<3>, 3> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex)
    {
      CGALPoint3 p1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[0]);
      CGALPoint3 q1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[1]);
      CGALPoint3 r1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[2]);
      CGALTriangle3 triangle{p1, q1, r1};

      CGALPoint3 p2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[0]);
      CGALPoint3 q2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[1]);
      CGALPoint3 r2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[2]);
      CGALPoint3 s2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[3]);
      CGALTetra  tetra{p2, q2, r2, s2};

      return CGAL::intersection(triangle, tetra);
    }
#  endif


  } // namespace internal
} // namespace CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
