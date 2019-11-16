// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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


// check the creation and destruction of particle within the particle handler
// class using a particle generator based on a DoFHandler and a quadrature

#include <deal.II/base/function_lib.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/particles/generators.h>
#include <deal.II/particles/particle_handler.h>

#include "../tests.h"

template <int dim, int spacedim>
void
test()
{
  parallel::distributed::Triangulation<dim, spacedim> tr(MPI_COMM_WORLD);

  GridGenerator::hyper_cube(tr);
  tr.refine_global(2);

  DoFHandler<dim, spacedim> dof_handler(tr);
  FE_Nothing<dim, spacedim> fe_nothing;
  dof_handler.distribute_dofs(fe_nothing);

  MappingQ<dim, spacedim> mapping(1);

  Particles::ParticleHandler<dim, spacedim> particle_handler(tr, mapping);

  parallel::distributed::Triangulation<dim, spacedim> particles_tr(
    MPI_COMM_WORLD);
  GridGenerator::hyper_cube(particles_tr, 0.1, 0.9);

  QGauss<dim> quadrature(2);


  DoFHandler<dim, spacedim> particles_dof_handler(particles_tr);
  FE_Q<dim, spacedim>       particles_fe(1);
  particles_dof_handler.distribute_dofs(particles_fe);
  Particles::Generators::non_matching_quadrature_points(tr,
                                                        particles_dof_handler,
                                                        quadrature.get_points(),
                                                        particle_handler);


  {
    deallog << "Locally owned active cells: "
            << tr.n_locally_owned_active_cells() << std::endl;

    deallog << "Global particles: " << particle_handler.n_global_particles()
            << std::endl;

    for (const auto &cell : tr.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            deallog << "Cell " << cell << " has "
                    << particle_handler.n_particles_in_cell(cell)
                    << " particles." << std::endl;
          }
      }

    for (const auto &particle : particle_handler)
      {
        deallog << "Particle index " << particle.get_id() << " is in cell "
                << particle.get_surrounding_cell(tr) << std::endl;
        deallog << "Particle location: " << particle.get_location()
                << std::endl;
      }
  }
  deallog << "OK" << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  MPILogInitAll init;

  deallog.push("2d/2d");
  test<2, 2>();
  deallog.pop();
  deallog.push("2d/3d");
  test<2, 3>();
  deallog.pop();
  deallog.push("3d/3d");
  test<3, 3>();
  deallog.pop();
}
