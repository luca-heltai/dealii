// ---------------------------------------------------------------------

// Copyright (C) 2022 by the deal.II authors

// This file is part of the deal.II library.

// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.

// ---------------------------------------------------------------------

// construct a simplex mesh out of a quad mesh, and compute the measure of it
// by using quadratures

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"

// #include "simplex.h"

template <int dim, int spacedim>
void
test(const int degree)
{
  QGaussSimplex<dim>           quad(degree);
  Triangulation<dim, spacedim> tria, out_tria;
  GridGenerator::hyper_cube(tria, -1., 1.);
  GridGenerator::convert_hypercube_to_simplex_mesh(tria, out_tria);

  std::vector<std::array<Point<spacedim>, dim + 1>> simplices;
  for (const auto &cell : out_tria.active_cell_iterators())
    {
      std::array<Point<spacedim>, dim + 1> vertices;
      for (unsigned int i = 0; i < (dim + 1); ++i)
        {
          vertices[i] = cell->vertex(i);
        }
      simplices.push_back(vertices);
    }

  deallog << "# dim = " << dim << std::endl;
  deallog << "# spacedim = " << spacedim << std::endl;

  auto quad2 = quad.mapped_quadrature(simplices);

  for (auto p : quad2.get_points())
    deallog << p << std::endl;

  const double area = std::accumulate(quad2.get_weights().begin(),
                                      quad2.get_weights().end(),
                                      0.0);
  deallog << std::endl
          << "# Area: " << std::scientific << std::setprecision(16) << area
          << std::endl
          << std::endl;
}



int
main()
{
  initlog();

  test<1, 2>(1);
  test<2, 2>(1);
  test<2, 3>(1);
  test<3, 3>(1);
}
