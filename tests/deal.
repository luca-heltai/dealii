// ---------------------------------------------------------------------
//
// Copyright (C) 2018 by the deal.II authors
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

// Extract cell bounding boxes rtree from the cache, and try to use it


#include <deal.II/base/bounding_box.h>

#include <deal.II/boost_adaptors/point.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <algorithm>

using namespace dealii;
namespace bg  = boost::geometry;
namespace bgi = boost::geometry::index;

template <int dim, int spacedim>
void
test(const unsigned int ref = 6)
{
  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_ball(tria);
  tria.refine_global(ref);
  GridTools::Cache<dim, spacedim> cache(tria);

  const auto &b_tree = cache.get_cell_bounding_boxes_rtree();

  std::cout << "Testing dim = " << dim << ", spacedim = " << spacedim
            << std::endl;

  // for (const auto &p : points)
  //   {
  //     std::vector<
  //       std::pair<BoundingBox<spacedim>,
  //                 typename Triangulation<dim,
  //                 spacedim>::active_cell_iterator>>
  //       res;
  //     b_tree.query(bgi::nearest(p, 1), std::back_inserter(res));
  //     std::cout << "Nearest cell to " << p << ":  " << res[0].second
  //               << std::endl;
  //   }
  const auto &vec_bboxes = extract_rtree_level(b_tree, 1);
  for (const auto &bbox : vec_bboxes)
    {
      std::cout << "Boundary points:" << std::endl;

      std::cout << bbox.get_boundary_points().first << "\t"
                << bbox.get_boundary_points().second << std::endl;
    }

  std::cout << "Number of bboxes: " << vec_bboxes.size() << std::endl;
}

int
main()
{
  test<2, 2>();
}
