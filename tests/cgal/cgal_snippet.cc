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

// Compute intersection of a *segment* embedded in 2D and a deal.II cell in 2D,
// and return a vector of arrays where you can build Quadrature rules. Then
// check that the sum of weights give the correct area for each region.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/cgal/intersections.h>
#include <deal.II/cgal/triangulation.h>

#include "../tests.h"

using namespace CGALWrappers;



int
main()
{
  initlog();
  Triangulation<3> tria0; // ambient space
  GridGenerator::hyper_cube(tria0, -1., 1.);
  const auto & cell = tria0.begin_active();
  MappingQ1<3> mapping;
  const auto & vertices = mapping.get_vertices(cell);

  std::array<CGALPoint3, 8> pts;
  std::transform(vertices.begin(),
                 vertices.end(),
                 pts.begin(),
                 [&](const Point<3> &p) {
                   return dealii_point_to_cgal_point<CGALPoint3>(p);
                 });
  Triangulation3 cgal_tria;
  cgal_tria.insert(pts.begin(), pts.end());
  deallog << std::boolalpha << "Is valid tria?: " << cgal_tria.is_valid()
          << std::endl;

  deallog << "Number of finite cells: " << cgal_tria.number_of_finite_cells()
          << std::endl;

  for (const auto &c : cgal_tria.finite_cell_handles())
    {
      const auto &tet = cgal_tria.tetrahedron(c);
      for (unsigned int i = 0; i < 4; ++i)
        deallog << "Vertex: " << tet.vertex(i) << std::endl;
      deallog << std::endl;
    }
  Triangulation<3> deal_tria;
  CGALWrappers::cgal_triangulation_to_dealii_triangulation(cgal_tria,
                                                           deal_tria);
  GridOut       go;
  std::ofstream output_name_tria("test_tetra.vtk"); // just to test
  go.write_vtk(deal_tria, output_name_tria);
  deallog << "OK" << std::endl;
}
