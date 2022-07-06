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

// Compute intersection of simplices in 2D, and return a vector of arrays where
// you can build Quadrature rules. Then check that the sum of weights give the
// correct area for each region.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <CGAL/Triangulation_3.h>
#include <deal.II/cgal/intersections.h>
#include <deal.II/cgal/triangulation.h>
using K_exact              = CGAL::Exact_predicates_exact_constructions_kernel;
using Triangulation3_exact = CGAL::Triangulation_3<K_exact>;
#include "../tests.h"


void
test_intersection_inside(Triangulation<3> &tria0, Triangulation<1, 3> &tria1)
{
  GridGenerator::hyper_cube(tria0, -1., 1.);
  GridGenerator::hyper_cube(tria1, -0.4, 0.2);
  tria1.begin_active()->vertex(0) = Point<3>(-.4, 0., 0.);
  tria1.begin_active()->vertex(1) = Point<3>(.5, 0., 0.2);
  const double expected_measure =
    (tria1.begin_active()->vertex(1) - tria1.begin_active()->vertex(0)).norm();

  GridOut       go;
  std::ofstream out_name("line_inside.vtk");
  go.write_vtk(tria1, out_name);

  // go.write_vtk(tria1, out_name);
  const auto vec_of_arrays =
    CGALWrappers::compute_intersection_of_cells<3, 1, 3>(tria0.begin_active(),
                                                         tria1.begin_active(),
                                                         MappingQ1<3>(),
                                                         MappingQ1<1, 3>());

  const auto quad = QGaussSimplex<1>(1).mapped_quadrature(vec_of_arrays);
  for (const auto &v : vec_of_arrays)
    {
      for (const auto &p : v)
        {
          deallog << p << std::endl;
        }
      deallog << std::endl;
    }
  const double sum =
    std::accumulate(quad.get_weights().begin(), quad.get_weights().end(), 0.);
  // assert(std::abs(sum - expected_measure) < 1e-15);
  deallog << "OK" << sum << "\t" << expected_measure << std::endl;
  { // Build a deal.II triangulation

    // Use its vertices to build a Delaunay_triangulation_3
    Triangulation3_exact tr;
    CGALWrappers::add_points_to_cgal_triangulation(tria0.get_vertices(), tr);

    // Transform the Dealaunay_triangulation_3 to a deal.II triangulation
    tria0.clear();
    CGALWrappers::cgal_triangulation_to_dealii_triangulation(tr, tria0);
    GridOut       go;
    std::ofstream out_name_out("cube_outside.vtk");
    go.write_vtk(tria0, out_name_out);
  }
}



void
test_intersection(Triangulation<3> &tria0, Triangulation<1, 3> &tria1)
{
  GridGenerator::hyper_cube(tria0, -1., 1.);
  GridGenerator::hyper_cube(tria1);
  auto cell1                    = tria1.begin_active();
  cell1->vertex(0)              = Point<3>();
  cell1->vertex(1)              = Point<3>(1.5, 1.5, 1.5);
  const double expected_measure = std::sqrt(3.);

  const auto vec_of_arrays =
    CGALWrappers::compute_intersection_of_cells<3, 1, 3>(tria0.begin_active(),
                                                         cell1,
                                                         MappingQ1<3>(),
                                                         MappingQ1<1, 3>());

  const auto   quad = QGaussSimplex<1>(1).mapped_quadrature(vec_of_arrays);
  const double sum =
    std::accumulate(quad.get_weights().begin(), quad.get_weights().end(), 0.);
  assert(std::abs(sum - expected_measure) < 1e-15);
  deallog << "OK" << sum << std::endl;
}


//
int
main()
{
  initlog();
  Triangulation<3>    tria0;
  Triangulation<1, 3> tria1;

  test_intersection_inside(tria0, tria1);
  tria0.clear();
  tria1.clear();
  test_intersection(tria0, tria1);
}
