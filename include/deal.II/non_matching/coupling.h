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

#ifndef dealii_non_matching_coupling
#define dealii_non_matching_coupling

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

/**
 * A namespace for functions offering tools to handle two meshes with no
 * alignment requirements, but where one of the meshes is embedded
 * inside the other in the real-space.
 *
 * Typically these functions allow for computations on the real-space
 * intersection between the two meshes e.g. surface integrals and
 * construction of mass matrices.
 */
namespace NonMatching
{
  /**
   * Create a coupling sparsity pattern for non-matching, overlapping grids.
   *
   * Given two non-matching triangulations, representing the domains $\Omega$
   * and $B$, with $B \subseteq \Omega$, and two finite element spaces
   * $V(\Omega) = \text{span}\{v_i\}_{i=0}^n$ and $Q(B) =
   * \text{span}\{w_j\}_{j=0}^m$, compute the sparsity pattern that would be
   * necessary to assemble the matrix
   * \f[
   * M_{ij} \dealcoloneq \int_{B} v_i(x) w_j(x) dx,
   *                     \quad i \in [0,n), j \in [0,m),
   * \f]
   * where $V(\Omega)$ is the finite element space associated with the
   * `space_dh` passed to this function (or part of it, if specified in
   * `space_comps`), while $Q(B)$ is the finite element space associated with
   * the `immersed_dh` passed to this function (or part of it, if specified in
   * `immersed_comps`).
   *
   * The `sparsity` is filled by locating the position of quadrature points
   * (obtained by the reference quadrature `quad`) defined on elements of $B$
   * with respect to the embedding triangulation $\Omega$. For each overlapping
   * cell, the entries corresponding to `space_comps` in `space_dh` and
   * `immersed_comps` in `immersed_dh` are added to the sparsity pattern.
   *
   * The `space_comps` and `immersed_comps` masks are assumed to be ordered in
   * the same way: the first component of `space_comps` will couple with the
   * first component of `immersed_comps`, the second with the second, and so
   * on. If one of the two masks has more non-zero than the other, then the
   * excess components will be ignored.
   *
   * If the domain $B$ does not fall within $\Omega$, an exception will be
   * thrown by the algorithm that computes the quadrature point locations. In
   * particular, notice that this function only makes sens for `dim1` lower or
   * equal than `dim0`. A static assert guards that this is actually the case.
   *
   * For both spaces, it is possible to specify a custom Mapping, which
   * defaults to StaticMappingQ1 for both.
   *
   * This function will also work in parallel, provided that the immersed
   * triangulation is of type parallel::shared::Triangulation<dim1,spacedim>.
   * An exception is thrown if you use an immersed
   * parallel::distributed::Triangulation<dim1,spacedim>.
   *
   * See the tutorial program step-60 for an example on how to use this
   * function.
   *
   * @author Luca Heltai, 2018
   */
  template <int dim0,
            int dim1,
            int spacedim,
            typename Sparsity,
            typename number = double>
  void
  create_coupling_sparsity_pattern(
    const DoFHandler<dim0, spacedim> &space_dh,
    const DoFHandler<dim1, spacedim> &immersed_dh,
    const Quadrature<dim1> &          quad,
    Sparsity &                        sparsity,
    const AffineConstraints<number> & constraints = AffineConstraints<number>(),
    const ComponentMask &             space_comps = ComponentMask(),
    const ComponentMask &             immersed_comps = ComponentMask(),
    const Mapping<dim0, spacedim> &   space_mapping =
      StaticMappingQ1<dim0, spacedim>::mapping,
    const Mapping<dim1, spacedim> &immersed_mapping =
      StaticMappingQ1<dim1, spacedim>::mapping);

  /**
   * Same as above, but takes an additional GridTools::Cache object, instead of
   * creating one internally. In this version of the function, the parameter @p
   * space_mapping cannot be specified, since it is taken from the @p cache
   * parameter.
   *
   * @author Luca Heltai, 2018
   */
  template <int dim0,
            int dim1,
            int spacedim,
            typename Sparsity,
            typename number = double>
  void
  create_coupling_sparsity_pattern(
    const GridTools::Cache<dim0, spacedim> &cache,
    const DoFHandler<dim0, spacedim> &      space_dh,
    const DoFHandler<dim1, spacedim> &      immersed_dh,
    const Quadrature<dim1> &                quad,
    Sparsity &                              sparsity,
    const AffineConstraints<number> &constraints = AffineConstraints<number>(),
    const ComponentMask &            space_comps = ComponentMask(),
    const ComponentMask &            immersed_comps = ComponentMask(),
    const Mapping<dim1, spacedim> &  immersed_mapping =
      StaticMappingQ1<dim1, spacedim>::mapping);


  /**
   * Create a coupling mass matrix for non-matching, overlapping grids.
   *
   * Given two non-matching triangulations, representing the domains $\Omega$
   * and $B$, with $B \subseteq \Omega$, and two finite element spaces
   * $V(\Omega) = \text{span}\{v_i\}_{i=0}^n$ and $Q(B) =
   * \text{span}\{w_j\}_{j=0}^m$, compute the coupling matrix
   * \f[
   * M_{ij} \dealcoloneq \int_{B} v_i(x) w_j(x) dx,
   *                     \quad i \in [0,n), j \in [0,m),
   * \f]
   * where $V(\Omega)$ is the finite element space associated with the
   * `space_dh` passed to this function (or part of it, if specified in
   * `space_comps`), while $Q(B)$ is the finite element space associated with
   * the `immersed_dh` passed to this function (or part of it, if specified in
   * `immersed_comps`).
   *
   * The corresponding sparsity patterns can be computed by calling the
   * make_coupling_sparsity_pattern function. The elements of the matrix are
   * computed by locating the position of quadrature points defined on elements
   * of $B$ with respect to the embedding triangulation $\Omega$.
   *
   * The `space_comps` and `immersed_comps` masks are assumed to be ordered in
   * the same way: the first component of `space_comps` will couple with the
   * first component of `immersed_comps`, the second with the second, and so
   * on. If one of the two masks has more non-zero entries non-zero than the
   * other, then the excess components will be ignored.
   *
   * If the domain $B$ does not fall within $\Omega$, an exception will be
   * thrown by the algorithm that computes the quadrature point locations. In
   * particular, notice that this function only makes sense for `dim1` lower or
   * equal than `dim0`. A static assert guards that this is actually the case.
   *
   * For both spaces, it is possible to specify a custom Mapping, which
   * defaults to StaticMappingQ1 for both.
   *
   * This function will also work in parallel, provided that the immersed
   * triangulation is of type parallel::shared::Triangulation<dim1,spacedim>.
   * An exception is thrown if you use an immersed
   * parallel::distributed::Triangulation<dim1,spacedim>.
   *
   * See the tutorial program step-60 for an example on how to use this
   * function.
   *
   * @author Luca Heltai, 2018
   */
  template <int dim0, int dim1, int spacedim, typename Matrix>
  void
  create_coupling_mass_matrix(
    const DoFHandler<dim0, spacedim> &                    space_dh,
    const DoFHandler<dim1, spacedim> &                    immersed_dh,
    const Quadrature<dim1> &                              quad,
    Matrix &                                              matrix,
    const AffineConstraints<typename Matrix::value_type> &constraints =
      AffineConstraints<typename Matrix::value_type>(),
    const ComponentMask &          space_comps    = ComponentMask(),
    const ComponentMask &          immersed_comps = ComponentMask(),
    const Mapping<dim0, spacedim> &space_mapping =
      StaticMappingQ1<dim0, spacedim>::mapping,
    const Mapping<dim1, spacedim> &immersed_mapping =
      StaticMappingQ1<dim1, spacedim>::mapping);

  /**
   * Same as above, but takes an additional GridTools::Cache object, instead of
   * creating one internally. In this version of the function, the parameter @p
   * space_mapping cannot specified, since it is taken from the @p cache
   * parameter.
   *
   * @author Luca Heltai, 2018
   */
  template <int dim0, int dim1, int spacedim, typename Matrix>
  void
  create_coupling_mass_matrix(
    const GridTools::Cache<dim0, spacedim> &              cache,
    const DoFHandler<dim0, spacedim> &                    space_dh,
    const DoFHandler<dim1, spacedim> &                    immersed_dh,
    const Quadrature<dim1> &                              quad,
    Matrix &                                              matrix,
    const AffineConstraints<typename Matrix::value_type> &constraints =
      AffineConstraints<typename Matrix::value_type>(),
    const ComponentMask &          space_comps    = ComponentMask(),
    const ComponentMask &          immersed_comps = ComponentMask(),
    const Mapping<dim1, spacedim> &immersed_mapping =
      StaticMappingQ1<dim1, spacedim>::mapping);

  /**
   * Create a interpolation sparsity pattern for non-matching,
   * overlapping grids.
   *
   * Given a triangulation representing the domains $\Omega$
   * and a particle handler of particles in $\Omega$,
   * and a finite element space
   * $V(\Omega) = \text{span}\{v_i\}_{i=0}^n$,
   * compute the sparsity pattern that would be
   * necessary to assemble the matrix
   * \f[
   * M_{ij} \dealcoloneq v_i(x_j) ,
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
   * If a particle does not fall within $\Omega$, an exception will be
   * thrown by the algorithm that computes the point location. In
   * particular, notice that this function only makes sens for `dim1` lower or
   * equal than `dim0`. A static assert guards that this is actually the case.
   *
   * For both spaces, it is possible to specify a custom Mapping, which
   * defaults to StaticMappingQ1 for both.
   *
   * This function will also work in parallel, provided that the immersed
   * triangulation is of type
   * parallel::distributed::Triangulation<dim1,spacedim> or
   * parallel::shared::Triangulation<dim1,spacedim>.
   *
   * See the tutorial program step-**** for an example on how to use this
   * function.
   *
   * @author Bruno Blais, Luca Heltai, 2019
   */
  template <int dim, int spacedim, typename Sparsity, typename number = double>
  void
  create_interpolation_sparsity_pattern(
    const DoFHandler<dim, spacedim> &                space_dh,
    const Particles::ParticleHandler<dim, spacedim> &particle_handler,
    Sparsity &                                       sparsity,
    const AffineConstraints<number> &constraints = AffineConstraints<number>(),
    const ComponentMask &            space_comps = ComponentMask())
  {
    if (particle_handler.n_locally_owned_particles() == 0)
      return; // nothing to do here

    const auto &tria     = space_dh.get_triangulation();
    const auto &fe       = space_dh.get_fe();
    auto        particle = particle_handler.begin();
    const auto  max_particles_per_cell =
      particle_handler.n_global_max_particles_per_cell();

    std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);
    std::vector<types::particle_index> particle_indices(max_particles_per_cell);

    // FullMatrix<double> local_matrix(max_particles_per_cell,
    // fe.dofs_per_cell);

    while (particle != particle_handler.end())
      {
        const auto &cell = particle->get_surrounding_cell(tria);
        const auto &dh_cell =
          typename DoFHandler<dim, spacedim>::cell_iterator(*cell, &space_dh);
        dh_cell->get_dof_indices(dof_indices);
        const auto pic         = particle_handler.particles_in_cells(cell);
        const auto n_particles = pic.end() - pic.begin();
        particle_indices.resize(n_particles);
        // local_matrix.reinit({n_particles, fe.dofs_per_cell});
        Assert(pic.begin() == particle, ExcInternalError());
        for (unsigned int i = 0; particle != pic.end(); ++particle, ++i)
          {
            // const auto &reference_location =
            // particle->get_reference_location();
            particle_indices[i] = particle->get_id();
          }
        constraints.add_entries_local_to_global(particle_indices,
                                                dof_indices,
                                                sparsity);
      }
  }
} // namespace NonMatching
DEAL_II_NAMESPACE_CLOSE

#endif
