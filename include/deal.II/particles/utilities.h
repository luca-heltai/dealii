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

      std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
      std::vector<types::particle_index>   particle_indices(
        max_particles_per_cell);

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
          particle_indices.resize(n_particles);
          // local_matrix.reinit({n_particles, fe.dofs_per_cell});
          Assert(pic.begin() == particle, ExcInternalError());
          for (unsigned int i = 0; particle != pic.end(); ++particle, ++i)
            {
              // const auto &reference_location =
              // particle->get_reference_location();

              // To Discuss -
              // Particles ids are numbered from [1,n_p] instead of [0,n_p[
              // when they are created from add_particles
              // However the particles generator number them [0,n_p[
              // This appears to be a bug?
              particle_indices[i] = particle->get_id();
            }
          constraints.add_entries_local_to_global(particle_indices,
                                                  dof_indices,
                                                  sparsity);
        }
    }
  } // namespace Utilities
} // namespace Particles
DEAL_II_NAMESPACE_CLOSE

#endif
