// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2021 by the deal.II authors
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

#ifndef dealii_base_bounding_box_cgal_h
#define dealii_base_bounding_box_cgal_h

#include <deal.II/base/bounding_box.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/qr.h>

#include <CGAL/Aff_transformation_3.h>
#include <CGAL/Optimal_bounding_box/oriented_bounding_box.h>
#include <deal.II/cgal/point_conversion.h>

#include <algorithm>
#include <limits>

DEAL_II_NAMESPACE_OPEN



template <int spacedim, typename Number = double>
class OptimalBoundingBox : public BoundingBox<spacedim, Number>
{
public:
  OptimalBoundingBox() = default;

  OptimalBoundingBox(const std::vector<Point<spacedim, Number>> &points);

private:
  std::array<Point<spacedim, Number>, Utilities::pow(2, spacedim)> extreme_pts;
};

// extern template class OptimalBoundingBox<2>;

DEAL_II_NAMESPACE_CLOSE

#endif
