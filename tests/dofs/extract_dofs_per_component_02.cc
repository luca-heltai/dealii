// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2018 by the deal.II authors
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


// Test extract_dofs_per_components on an hyper cube
// with three components. The mask of component 1 is set to false
// and only component 0 and component 1 should be extracted


#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include "../tests.h"



template <int dim>
void
test()
{
  parallel::distributed::Triangulation<dim> tr(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tr);
  tr.refine_global(2);

  FESystem<dim>   fe(FE_Q<dim>(1), 3);
  DoFHandler<dim> dof(tr);
  dof.distribute_dofs(fe);

  ComponentMask mask(fe.n_components(), true);
  mask.set(1, false);

  std::vector<IndexSet> dofs_per_components =
    DoFTools::extract_dofs_per_component(dof,
                                         DoFTools::OwnershipType::owned,
                                         mask);

  MappingQ<dim>                                 mapping(1);
  std::map<types::global_dof_index, Point<dim>> support_points;

  DoFTools::map_dofs_to_support_points(mapping, dof, support_points);

  for (unsigned int c = 0; c < dofs_per_components.size(); ++c)
    {
      deallog << "Component : " << c << std::endl;
      auto index = dofs_per_components[c].begin();
      for (; index != dofs_per_components[c].end(); ++index)
        {
          deallog << support_points[*index] << std::endl;
        }
    }
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPILogInitAll init;

  deallog.push("2d");
  test<2>();
  deallog.pop();
  deallog.push("3d");
  test<3>();
  deallog.pop();
}
