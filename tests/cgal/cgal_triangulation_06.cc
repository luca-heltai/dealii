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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/io.h>
#include <deal.II/cgal/triangulation.h>

#include "../tests.h"

using namespace CGALWrappers;

using K                 = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGALTriangulation = CGAL::Delaunay_triangulation_3<K>;
using CGALPointType     = K::Point_3;
using CGALMesh          = CGAL::Surface_mesh<CGALPointType>;
using CGALPoly          = CGAL::Polyhedron_3<K>;

template <int dim, int spacedim>
void
test()
{
  // Build a deal.II triangulation of a non-convex domain
  Triangulation<dim, spacedim> tria;
  GridGenerator::hyper_L(tria);
  tria.refine_global(2);

  // Use its vertices to find the convex hull
  CGALMesh chull_out;
  compute_convex_hull<3, CGALMesh>(tria.get_vertices(), chull_out);
  deallog << chull_out << std::endl;

  CGALPoly chull_out_poly;
  compute_convex_hull<3, CGALPoly>(tria.get_vertices(), chull_out_poly);
  deallog << chull_out_poly << std::endl;
}

int
main()
{
  initlog();
  test<3, 3>();
}
