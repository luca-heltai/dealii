// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
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

#ifndef dealii_numerics_rtree_h
#define dealii_numerics_rtree_h

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/boost_adaptors/bounding_box.h>
#include <deal.II/boost_adaptors/point.h>

#include <boost/geometry.hpp>

#include <memory>


DEAL_II_NAMESPACE_OPEN



/**
 * A wrapper for the boost::geometry library, used to compute am RTree from a
 * collection of Point<dim> based spatial objects, or BoundingBox<dim> based
 * spatial objects.
 *
 * @author Luca Heltai, 2018.
 */
template <typename LeafType>
using RTree =
  boost::geometry::index::rtree<LeafType, boost::geometry::index::linear<16>>;

/**
 * Construct the correct RTree object by passing an iterator range.
 */
template <typename LeafTypeIterator>
RTree<typename LeafTypeIterator::value_type>
pack_rtree(LeafTypeIterator begin, LeafTypeIterator end);


// Inline and template functions
template <typename LeafTypeIterator>
RTree<typename LeafTypeIterator::value_type>
pack_rtree(LeafTypeIterator begin, LeafTypeIterator end)
{
  return RTree<LeafTypeIterator::value_type>(begin, end);
}

DEAL_II_NAMESPACE_CLOSE

#endif
