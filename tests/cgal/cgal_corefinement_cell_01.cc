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

// Given two deal.II cells, compute their union and intersection and store them
// in valid CGAL::Surface_mesh The output meshes are written in the output file
// and can then be visualized in ParaView using the .off files mesh_union.stl
// and mesh_intersection.stl

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <CGAL/IO/io.h>
#include <deal.II/cgal/surface_mesh.h>
#include <deal.II/cgal/triangulation.h>

#include "../tests.h"


using namespace CGALWrappers;
using K                 = CGAL::Exact_predicates_exact_constructions_kernel;
using CGALPoint         = CGAL::Point_3<K>;
using CGALTriangulation = CGAL::Triangulation_3<K>;

void
test()
{
  Triangulation<3, 3> tria0;
  Triangulation<3, 3> tria1;
  GridGenerator::hyper_cube(tria0, 0.5, 1.5);
  GridGenerator::hyper_cube(tria1, 0., 1.);
  GridTools::rotate(numbers::PI_4, 1, tria1);
  std::array<BooleanOperation, 2> operations{BooleanOperation::union_op,
                                             BooleanOperation::intersection_op};
  const auto                      cell0 = tria0.begin_active();
  const auto                      cell1 = tria1.begin_active();
  CGAL::Surface_mesh<CGALPoint>   corefined;
  for (const auto &bool_op : operations)
    {
      corefined.clear();
      corefine_and_compute_boolean_operation_dealii_cells<CGALPoint, 3, 3>(
        cell0, cell1, corefined, bool_op);
      Assert(corefined.is_valid(),
             ExcMessage("The boolean operation you performed is not valid!"));
      deallog << corefined << std::endl;
    }
  // Fill a deal.II triangulation with tets
  CGALTriangulation   tr;
  Triangulation<3, 3> tria_test;
  tr.insert(corefined.points().begin(), corefined.points().end());
  cgal_triangulation_to_dealii_triangulation(tr, tria_test);
  deallog << "Show triangulation vertices: " << std::endl;
  for (const auto &f : tr.finite_cell_handles())
    {
      for (unsigned int i = 0; i < 4; ++i)
        {
          deallog << f->vertex(i)->point() << std::endl;
        }
      deallog << "\n" << std::endl;
    }

  GridOut go;
  go.write_vtk(tria_test, deallog.get_file_stream());
  std::ofstream grid_name("corefined_with_tets.vtk");
  go.write_vtk(tria_test, grid_name);

  tr.clear();
  const int  degree = 3;
  const auto quad_over_intersection =
    quadrature_inside_region<3>(corefined, degree, tr);
  deallog << "Area of the surface:"
          << std::accumulate(quad_over_intersection.get_weights().begin(),
                             quad_over_intersection.get_weights().end(),
                             0.0)
          << std::endl;
}

int
main()
{
  initlog();
  test();
}
