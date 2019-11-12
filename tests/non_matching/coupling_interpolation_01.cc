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

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/particles/particle_handler.h>

#include "../tests.h"

namespace LA {
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) &&     \
    !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
using namespace ::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
using namespace ::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

using namespace dealii;

// Test that an interpolation matrix can be constructed for a single
// particle for every valid dimension pair that exist

template <int dim, int spacedim> void test() {
  deallog << "dim: " << dim << ", spacedim: " << spacedim << std::endl;
  parallel::distributed::Triangulation<dim, spacedim> space_tria(
      MPI_COMM_WORLD);
  MappingQ1<dim, spacedim> mapping;
  GridGenerator::hyper_cube(space_tria, -1, 1);
  space_tria.refine_global(2);

  Particles::ParticleHandler<dim, spacedim> particle_handler(space_tria,
                                                             mapping);

  // Create a single particle at an arbitrary point of the triangulation
  // Insert one particle per processor
  std::vector<Point<spacedim>> particles_positions;
  const unsigned int n_particles = 1;
  for (unsigned int i = 0; i < n_particles; ++i)
    particles_positions.emplace_back(random_point<spacedim>());

  particle_handler.insert_particles(particles_positions);

  deallog << "Number of particles: " << particle_handler.n_global_particles()
          << std::endl;

  FE_Q<dim, spacedim> space_fe(1);

  deallog << "Space FE: " << space_fe.get_name() << std::endl;

  DoFHandler<dim, spacedim> space_dh(space_tria);
  space_dh.distribute_dofs(space_fe);

  IndexSet locally_owned_dofs = space_dh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(space_dh, locally_relevant_dofs);

  deallog << "Space dofs: " << space_dh.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(particle_handler.n_global_particles(),
                             space_dh.n_dofs());

  NonMatching::create_interpolation_sparsity_pattern(space_dh, particle_handler,
                                                     dsp);

  // Temporary - Display the sparsity pattern
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);
  std::string fname("sparsity_pattern_" + Utilities::int_to_string(dim) + "_" +
                    Utilities::int_to_string(spacedim) + ".svg");
  std::ofstream out(fname.c_str());
  sparsity_pattern.print_svg(out);
  // End temporary

  IndexSet local_particle_index_set(n_particles);
  const unsigned int my_mpi_id =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  local_particle_index_set.add_range(my_mpi_id * n_particles,
                                     (my_mpi_id + 1) * n_particles);
  auto global_particles_index_set =
      Utilities::MPI::all_gather(MPI_COMM_WORLD, n_particles);

  SparsityTools::distribute_sparsity_pattern(dsp, global_particles_index_set,
                                             MPI_COMM_WORLD,
                                             local_particle_index_set);

  LA::MPI::SparseMatrix system_matrix;
  system_matrix.reinit(local_particle_index_set, locally_owned_dofs, dsp,
                       MPI_COMM_WORLD);

  // NonMatching::create_interpolation_matrix(space_dh, dh, quad, coupling);

  // now take ones in space and interpolate it to the points
  // Locally locally_relevant_dofs are used to allow for modifications
  // of the vectors
  // LA::MPI::Vector<double> space_ones(locally_relevant_dofs);
  // LA::MPI::Vector<double> ones(locally_relevant_dofs);

  // space_ones = 1.0;
  // coupling.Tvmult(ones, space_ones);
  // mass_matrix_inv.solve(ones);

  // Vector<double> real_ones(dh.n_dofs());
  // real_ones = 1.0;
  // ones -= real_ones;

  // deallog << "Error on constants: " << ones.l2_norm() << std::endl;
}

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  initlog();
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
