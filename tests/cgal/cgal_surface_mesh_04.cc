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

// Convert a deal.II triangulation to a CGAL surface mesh. In the 2D case,
// thw whole triangulation is a 2D surface mesh. In 3D, the surface mesh
// describes the boundary of the deal.II Triangulation.

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <deal.II/cgal/surface_mesh.h>
#include <deal.II/cgal/triangulation.h>
#include <string.h>

#include "../tests.h"

using namespace CGALWrappers;
using K         = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGALPoint = CGAL::Point_3<K>;
using CGALMesh  = CGAL::Surface_mesh<CGALPoint>;

template <int dim, int spacedim>
void
test()
{
  deallog << "dim= " << dim << ",\t spacedim= " << spacedim << std::endl;

  Triangulation<spacedim> tria_in;
  Triangulation<3, 3>     tria_out;
  GridOut                 go;
  CGALMesh                surface_mesh;

  // GridGenerator::hyper_ball(tria_out, {.3, .2, .4}, 0.2, false);
  GridGenerator::hyper_cube(tria_out, 0., 1.);

  tria_out.refine_global(1);
  CGALMesh surf;
  int      i = 0;
  for (const auto &cell : tria_out.active_cell_iterators())
    {
      // deallog << "iterata: " << i++ << std::endl;
      // deallog << "Cella: " << cell->index() << std::endl;
      for (const auto &f : cell->face_indices())
        {
          // deallog << "Cella face orientation: " << cell->face_orientation(f)
          //         << std::endl;
          // if (cell->face_orientation(f) == 0)
          //   {
          //     // [TODO:] quando face orientation == 0 fai in modo che si
          //     possa
          //     // fare add_face di CGAL
          //     deallog << "Entrato" << std::endl;
          //     // std::swap(cell->face(f)->vertex(1),
          //     cell->face(f)->vertex(2)); deallog << "Prova: " <<
          //     cell->face_orientation(f) << std::endl;
          //   }
          // dealii_cell_to_cgal_surface_mesh(cell, MappingQ1<3>(), surf);
          deallog << cell->face(f)->vertex_index(0) << std::endl;
          deallog << cell->face(f)->vertex_index(1) << std::endl;
          deallog << cell->face(f)->vertex_index(3) << std::endl;
          deallog << cell->face(f)->vertex_index(2) << std::endl;
        }
      // for (const auto &i : cell->vertex_indices())
      //   {
      //     deallog << i << std::endl;
      //   }

      surf.clear();
    }


  // std::ofstream output_test_embedded("test_ball.vtk");
  // GridOut().write_vtk(tria_out, output_test_embedded);

  // Triangulation<2, 3> provazza;
  // cgal_surface_mesh_to_dealii_triangulation(surf, provazza);
  std::ofstream test_grid_name("test_grid_square.vtk");
  GridOut().write_vtk(tria_out, test_grid_name);
}

int
main()
{
  initlog();
  // test<2, 2>();
  test<3, 3>();
}
