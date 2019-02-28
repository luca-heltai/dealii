// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2017 by the deal.II authors
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


// try constructing FE_Q1_NonLocal

#include <deal.II/grid/grid_generator.h>

#include <iostream>

#include "../tests.h"
#include "fe_q1_non_local.h"

int
main()
{
  initlog();
  Triangulation<2> tria;
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);

  FE_Q1_NonLocal<2> fe(tria);
  deallog << fe.get_name() << std::endl;

  deallog << "n_non_local_dofs: " << fe.n_non_local_dofs() << std::endl;
  for (auto cell = tria.begin_active(); cell != tria.end(); ++cell)
    {
      deallog << "cell " << cell->index() << ": ";
      fe.get_non_local_dofs_on_cell(cell->index())
        .print(deallog.get_file_stream());
    }

  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  // AssertDimension (dof_handler.n_dofs(), tria.n_vertices());

  std::vector<types::global_dof_index> local_dof_indices(
    GeometryInfo<2>::vertices_per_cell);
  std::vector<types::global_dof_index> vertex_indices(
    GeometryInfo<2>::vertices_per_cell);
  for (auto cell = dof_handler.begin_active(); cell != dof_handler.end();
       ++cell)
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
          vertex_indices[v] = cell->vertex_index(v);
          // AssertDimension(local_dof_indices[v], vertex_indices[v]);
        }
    }
}
