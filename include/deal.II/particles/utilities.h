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

#ifndef dealii_particles_utilities
#define dealii_particles_utilities

#include <deal.II/base/config.h>

#include <deal.II/base/quadrature.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/component_mask.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/particles/particle_handler.h>


DEAL_II_NAMESPACE_OPEN


namespace Particles
{
  /**
   * A namespace for functions offering tools to handle ParticleHandlers and
   * their coupling with DoFHandlers
   */
  namespace Utilities
  {
    /**
     * Create a interpolation sparsity pattern for particles.
     *
     * Given a triangulation representing the domains $\Omega$
     * and a particle handler of particles in $\Omega$,
     * and a finite element space
     * $V(\Omega) = \text{span}\{v_i\}_{i=0}^n$,
     * compute the sparsity pattern that would be
     * necessary to assemble the matrix
     * \f[
     * M_{ij} \dealcoloneq v_j(x_i) ,
     * \f]
     * where $V(\Omega)$ is the finite element space associated with the
     * `space_dh`
     *
     * The `sparsity` is filled by locating the position of the points
     * within the particle handler with respect to the embedding triangulation
     * $\Omega$.
     *
     * The `space_comps` masks will assume which components are coupled
     *
     * If a particle does not fall within $\Omega$, it is ignored, and its
     * interpolated field will be zero.
     *
     * See the tutorial program step-**** for an example on how to use this
     * function.
     *
     * @author Bruno Blais, Luca Heltai, 2019
     */
    template <int dim,
              int spacedim,
              typename Sparsity,
              typename number = double>
    void
    create_interpolation_sparsity_pattern(
      const DoFHandler<dim, spacedim> &                space_dh,
      const Particles::ParticleHandler<dim, spacedim> &particle_handler,
      Sparsity &                                       sparsity,
      const AffineConstraints<number> &                constraints =
        AffineConstraints<number>(),
      const ComponentMask &space_comps = ComponentMask())
    {
      if (particle_handler.n_locally_owned_particles() == 0)
        return; // nothing to do here

      const auto &tria     = space_dh.get_triangulation();
      const auto &fe       = space_dh.get_fe();
      auto        particle = particle_handler.begin();
      const auto  max_particles_per_cell =
        particle_handler.n_global_max_particles_per_cell();


      // Take care of components
      const ComponentMask comps =
        (space_comps.size() == 0 ? ComponentMask(fe.n_components(), true) :
                                   space_comps);
      AssertDimension(comps.size(), fe.n_components());

      const auto n_comps = comps.n_selected_components();

      // Global to local indices
      std::vector<unsigned int> space_gtl(fe.n_components(),
                                          numbers::invalid_unsigned_int);
      for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
        if (comps[i])
          space_gtl[i] = j++;

      // [TODO]: when the add_entries_local_to_global below will implement
      // the version with the dof_mask, this should be uncommented.
      // // Construct a dof_mask, used to distribute entries to the sparsity
      // Table<2, bool> dof_mask(max_particles_per_cell * n_comps,
      //                         fe.dofs_per_cell);
      // dof_mask.fill(false);
      // for (unsigned int i = 0; i < space_fe.dofs_per_cell; ++i)
      //   {
      //     const auto comp_i = space_fe.system_to_component_index(i).first;
      //     if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
      //       for (unsigned int j = 0; j < max_particles_per_cell; ++j)
      //         dof_mask(i, j * n_comps + space_gtl[comp_i]) = true;
      //   }

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      std::vector<types::particle_index>   particle_indices(
        max_particles_per_cell * n_comps);

      // FullMatrix<double> local_matrix(max_particles_per_cell,
      // fe.dofs_per_cell);

      while (particle != particle_handler.end())
        {
          const auto &cell = particle->get_surrounding_cell(tria);
          const auto &dh_cell =
            typename DoFHandler<dim, spacedim>::cell_iterator(*cell, &space_dh);
          dh_cell->get_dof_indices(dof_indices);
          const auto pic         = particle_handler.particles_in_cell(cell);
          const auto n_particles = particle_handler.n_particles_in_cell(cell);
          particle_indices.resize(n_particles * n_comps);
          // local_matrix.reinit({n_particles, fe.dofs_per_cell});
          Assert(pic.begin() == particle, ExcInternalError());
          for (unsigned int i = 0; particle != pic.end(); ++particle, ++i)
            {
              const auto p_id = particle->get_id();
              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                {
                  const auto comp_j =
                    space_gtl[fe.system_to_component_index(j).first];
                  if (comp_j != numbers::invalid_unsigned_int)
                    constraints.add_entries_local_to_global(
                      {p_id * n_comps + comp_j}, {dof_indices[j]}, sparsity);
                }
            }
          // [TODO]: when this works, use this:
          // constraints.add_entries_local_to_global(particle_indices,
          //                                         dof_indices,
          //                                         sparsity,
          //                                         dof_mask);
        }
    }


    /**
     * Create a interpolation matrix for particles.
     *
     * Given a triangulation representing the domains $\Omega$
     * and a particle handler of particles in $\Omega$,
     * and a finite element space
     * $V(\Omega) = \text{span}\{v_i\}_{i=0}^n$,
     * compute the matrix that would be
     * necessary to assemble the matrix
     * \f[
     * M_{ij} \dealcoloneq v_j(x_i) ,
     * \f]
     * where $V(\Omega)$ is the finite element space associated with the
     * `space_dh`
     *
     * The `sparsity` is filled by locating the position of the points
     * within the particle handler with respect to the embedding triangulation
     * $\Omega$.
     *
     * The `space_comps` masks will assume which components are coupled
     *
     * If a particle does not fall within $\Omega$, it is ignored, and its
     * interpolated field will be zero.
     *
     * See the tutorial program step-**** for an example on how to use this
     * function.
     *
     * @author Bruno Blais, Luca Heltai, 2019
     */
    template <int dim, int spacedim, typename Matrix, typename number = double>
    void
    create_interpolation_matrix(
      const DoFHandler<dim, spacedim> &                space_dh,
      const Particles::ParticleHandler<dim, spacedim> &particle_handler,
      Matrix &                                         matrix,
      const AffineConstraints<number> &                constraints =
        AffineConstraints<number>(),
      const ComponentMask &space_comps = ComponentMask())
    {
      AssertDimension(matrix.n(), space_dh.n_dofs());

      if (particle_handler.n_locally_owned_particles() == 0)
        return; // nothing to do here

      const auto &tria     = space_dh.get_triangulation();
      const auto &fe       = space_dh.get_fe();
      auto        particle = particle_handler.begin();
      const auto  max_particles_per_cell =
        particle_handler.n_global_max_particles_per_cell();

      // Take care of components
      const ComponentMask comps =
        (space_comps.size() == 0 ? ComponentMask(fe.n_components(), true) :
                                   space_comps);
      AssertDimension(comps.size(), fe.n_components());
      const auto n_comps = comps.n_selected_components();

      AssertDimension(matrix.m(),
                      particle_handler.n_global_particles() * n_comps);


      // Global to local indices
      std::vector<unsigned int> space_gtl(fe.n_components(),
                                          numbers::invalid_unsigned_int);
      for (unsigned int i = 0, j = 0; i < space_gtl.size(); ++i)
        if (comps[i])
          space_gtl[i] = j++;

      // [TODO]: when the add_entries_local_to_global below will implement
      // the version with the dof_mask, this should be uncommented.
      // // Construct a dof_mask, used to distribute entries to the sparsity
      // Table<2, bool> dof_mask(max_particles_per_cell * n_comps,
      //                         fe.dofs_per_cell);
      // dof_mask.fill(false);
      // for (unsigned int i = 0; i < space_fe.dofs_per_cell; ++i)
      //   {
      //     const auto comp_i = space_fe.system_to_component_index(i).first;
      //     if (space_gtl[comp_i] != numbers::invalid_unsigned_int)
      //       for (unsigned int j = 0; j < max_particles_per_cell; ++j)
      //         dof_mask(i, j * n_comps + space_gtl[comp_i]) = true;
      //   }

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      std::vector<types::particle_index>   particle_indices(
        max_particles_per_cell * n_comps);

      FullMatrix<double> local_matrix(max_particles_per_cell * n_comps,
                                      fe.dofs_per_cell);

      while (particle != particle_handler.end())
        {
          const auto &cell = particle->get_surrounding_cell(tria);
          const auto &dh_cell =
            typename DoFHandler<dim, spacedim>::cell_iterator(*cell, &space_dh);
          dh_cell->get_dof_indices(dof_indices);
          const auto pic         = particle_handler.particles_in_cell(cell);
          const auto n_particles = particle_handler.n_particles_in_cell(cell);
          particle_indices.resize(n_particles * n_comps);
          local_matrix.reinit({n_particles * n_comps, fe.dofs_per_cell});
          Assert(pic.begin() == particle, ExcInternalError());
          for (unsigned int i = 0; particle != pic.end(); ++particle, ++i)
            {
              const auto &reference_location =
                particle->get_reference_location();

              for (unsigned int d = 0; d < n_comps; ++d)
                particle_indices[i * n_comps + d] =
                  particle->get_id() * n_comps + d;

              for (unsigned int j = 0; j < fe.dofs_per_cell; ++j)
                {
                  const auto comp_j =
                    space_gtl[fe.system_to_component_index(j).first];
                  if (comp_j != numbers::invalid_unsigned_int)
                    local_matrix(i * n_comps + comp_j, j) =
                      fe.shape_value(j, reference_location);
                }
            }
          constraints.distribute_local_to_global(local_matrix,
                                                 particle_indices,
                                                 dof_indices,
                                                 matrix);
        }
      matrix.compress(VectorOperation::add);
    }
  } // namespace Utilities
} // namespace Particles
DEAL_II_NAMESPACE_CLOSE

#endif
