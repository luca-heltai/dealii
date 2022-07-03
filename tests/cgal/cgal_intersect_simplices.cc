// ---------------------------------------------------------------------

// Copyright (C) 2022 by the deal.II authors

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

// Compute intersection of simplices in 2D, and check they are correct. In 3D,
// this requires an older version of CGAL

#include <deal.II/base/quadrature_lib.h>

#include <CGAL/IO/io.h>
#include <CGAL/Simple_cartesian.h>
#include <deal.II/cgal/intersections.h>
#include <deal.II/cgal/utilities.h>

#include "../tests.h"

using namespace CGALWrappers;

int
main()
{
  initlog();
  std::array<Point<2>, 3> t1{
    {Point<2>{-1., 0.}, Point<2>{1., 0.}, Point<2>{0., 1.}}};
  std::array<Point<2>, 3> t2{
    {Point<2>{-0.5, 1.5}, Point<2>{0.1, 0.5}, Point<2>{0.6, 1.6}}};
  std::array<Point<2>, 3> t3{
    {Point<2>{0.2, 0.2}, Point<2>{1.2, 0.5}, Point<2>{1.2, 1.2}}};
  std::array<Point<2>, 3> t4{
    {Point<2>{0.1, -0.5}, Point<2>{1.8, -0.5}, Point<2>{0.8, 1.1}}};
  std::array<Point<2>, 2> segm{{Point<2>{-.1, 0.1}, Point<2>{1.4, 1.2}}};

  deallog << "Segment-Triangle" << std::endl;
  auto test_segment = CGALWrappers::internal::compute_intersection(t1, segm);
  if (const CGALSegment2 *s = boost::get<CGALSegment2>(&*test_segment))
    deallog << *s << std::endl;

  deallog << "Triange-Triangle" << std::endl;
  const auto test_vec_of_pts =
    CGALWrappers::internal::compute_intersection(t1, t2);
  const auto test_triangle =
    CGALWrappers::internal::compute_intersection(t1, t3);

  deallog << "Case: Vector of points" << std::endl;
  if (const std::vector<CGALPoint2> *vp =
        boost::get<std::vector<CGALPoint2>>(&*test_vec_of_pts))
    {
      for (const auto &p : *vp)
        deallog << p << std::endl;
    }

  deallog << "Case: Triangle" << std::endl;
  if (const CGALTriangle2 *t = boost::get<CGALTriangle2>(&*test_triangle))
    {
      deallog << *t << std::endl;
    }
}
