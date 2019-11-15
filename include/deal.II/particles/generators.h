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

#ifndef dealii_particles_particle_generator_h
#define dealii_particles_particle_generator_h

#include <deal.II/base/function.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/particles/particle.h>
#include <deal.II/particles/particle_handler.h>

#include <random>

DEAL_II_NAMESPACE_OPEN

#ifdef DEAL_II_WITH_P4EST

namespace Particles
{
  /**
   * A namespace that contains all classes that are related to the particle
   * generation.
   */
  namespace Generators
  {
    /**
     * A function that generates particles in every cell at specified @p particle_reference_locations.
     * The total number of particles that is added to the @p particle_handler object is
     * the number of locally owned cells of the @p triangulation times the number of
     * locations in @p particle_reference_locations. An optional @p mapping argument
     * can be used to map from @p particle_reference_locations to the real particle locations.
     *
     * @param triangulation The triangulation associated with the @p particle_handler.
     *
     * @param particle_reference_locations A vector of positions in the unit cell.
     * Particles will be generated in every cell at these locations.
     *
     * @param particle_handler The particle handler that will take ownership
     * of the generated particles.
     *
     * @param mapping An optional mapping object that is used to map reference
     * location in the unit cell to the real cells of the triangulation. If no
     * mapping is provided a MappingQ1 is assumed.
     */
    template <int dim, int spacedim = dim>
    void
    regular_reference_locations(
      const Triangulation<dim, spacedim> &triangulation,
      const std::vector<Point<dim>> &     particle_reference_locations,
      ParticleHandler<dim, spacedim> &    particle_handler,
      const Mapping<dim, spacedim> &      mapping =
        StaticMappingQ1<dim, spacedim>::mapping);

    /**
     * A function that generates one particle at a random location in cell @p cell and with
     * index @p id. The function expects a random number generator to avoid the expensive generation
     * and destruction of a generator for every particle and optionally takes
     * into account a mapping for the cell. The algorithm implemented in the
     * function is described in @cite GLHPW2018. In short, the algorithm
     * generates
     * random locations within the bounding box of the @p cell. It then inverts the mapping
     * to check if the generated particle is within the cell itself. This makes
     * sure the algorithm produces statistically random locations even for
     * nonlinear mappings and distorted cells. However, if the ratio between
     * bounding box and cell volume becomes very large -- i.e. the cells become
     * strongly deformed, for example a pencil shaped cell that lies diagonally
     * in the domain -- then the algorithm can become very inefficient.
     * Therefore, it only tries to find a location ni the cell a fixed number of
     * times before throwing an error message.
     *
     * @param[in] cell The cell in which a particle is generated.
     *
     * @param[in] id The particle index that will be assigned to the new
     * particle.
     *
     * @param[in,out] random_number_generator A random number generator that
     * will be used for the creation of th particle.
     *
     * @param[in] mapping An optional mapping object that is used to map
     * reference location in the unit cell to the real cell. If no mapping is
     * provided a MappingQ1 is assumed.
     */
    template <int dim, int spacedim = dim>
    Particle<dim, spacedim>
    random_particle_in_cell(
      const typename Triangulation<dim, spacedim>::active_cell_iterator &cell,
      const types::particle_index                                        id,
      std::mt19937 &                random_number_generator,
      const Mapping<dim, spacedim> &mapping =
        StaticMappingQ1<dim, spacedim>::mapping);

    /**
     * A function that generates particles randomly in the domain with a
     * particle density
     * according to a provided probability density function @p probability_density_function.
     * The total number of particles that is added to the @p particle_handler object is
     * @p n_particles_to_create. An optional @p mapping argument
     * can be used to map from @p particle_reference_locations to the real particle locations.
     * The function can compute the number of particles per cell either
     * deterministically by computing the integral of the probability density
     * function for each cell and creating
     * particles accordingly (if option @p random_cell_selection set to false), or it can
     * select cells randomly based on the probability density function and the
     * cell size
     * (if option @p random_cell_selection set to true). In either case the position of
     * individual particles inside the cell is computed randomly.
     *
     * The algorithm implemented in the function is described in @cite
     * GLHPW2018.
     *
     * @param[in] triangulation The triangulation associated with the @p particle_handler.
     *
     * @param[in] probability_density_function A function with non-negative
     * values that determines the probability density of a particle to be
     * generated in this location. The function does not need to be normalized.
     *
     * @param[in] random_cell_selection A bool that determines, how the number
     * of particles per cell is computed (see the description above).
     *
     * @param[in] n_particles_to_create The number of particles that will be
     * created by this function.
     *
     * @param[in,out] particle_handler The particle handler that will take
     * ownership of the generated particles.
     *
     * @param[in] mapping An optional mapping object that is used to map
     * reference location in the unit cell to the real cells of the
     * triangulation. If no mapping is provided a MappingQ1 is assumed.
     *
     * @param[in] random_number_seed An optional seed that determines the
     * initial state of the random number generator. Use the same number to get
     * a reproducible particle distribution, or a changing number (e.g. based on
     * system time) to generate different particle distributions for each call
     * to this function.
     */
    template <int dim, int spacedim = dim>
    void
    probabilistic_locations(
      const Triangulation<dim, spacedim> &triangulation,
      const Function<spacedim> &          probability_density_function,
      const bool                          random_cell_selection,
      const types::particle_index         n_particles_to_create,
      ParticleHandler<dim, spacedim> &    particle_handler,
      const Mapping<dim, spacedim> &      mapping =
        StaticMappingQ1<dim, spacedim>::mapping,
      const unsigned int random_number_seed = 5432);


    /**
     * A function that generates particles at the location of the support points
     * of a DoFHandler
     * The total number of particles that is added to the @p particle_handler object is
     * the number of dofs of the DoFHandler that is passed that are within the
     * triangulation and whose components are within the ComponentMask.
     *
     * @param[in] triangulation The triangulation associated with the @p particle_handler.
     *
     * @param[in] a DOF handler that may live on another triangulation
     *
     *
     * @param[in,out] particle_handler The particle handler that will take
     * ownership of the generated particles.
     *
     * @param[in] mapping An optional mapping object that is used to map
     * the DOF locations. If no mapping is provided a MappingQ1 is assumed.
     *
     * @param[in] Component mask of the dof_handler from which this originates
     *
     * @author Bruno Blais, Luca Heltai, 2019
     *
     *
     */
    template <int dim, int spacedim = dim>
    void
    dof_support_points(const Triangulation<dim, spacedim> &triangulation,
                       const DoFHandler<dim, spacedim> &   particle_dof_handler,
                       ParticleHandler<dim, spacedim> &    particle_handler,
                       const Mapping<dim, spacedim> &      mapping =
                         StaticMappingQ1<dim, spacedim>::mapping,
                       const ComponentMask &components = ComponentMask())
    {
      const auto &fe = particle_dof_handler.get_fe();

      // Take care of components
      const ComponentMask mask =
        (components.size() == 0 ? ComponentMask(fe.n_components(), true) :
                                  components);

      std::map<types::global_dof_index, Point<spacedim>> support_points_map;

      DoFTools::map_dofs_to_support_points(mapping,
                                           particle_dof_handler,
                                           support_points_map,
                                           mask);

      // Generate vector of points from the map
      // Memory is reserved for efficiency
      // is problematic
      std::vector<Point<spacedim>> support_points_vec;
      support_points_vec.reserve(support_points_map.size());
      for (auto const &element : support_points_map)
        support_points_vec.push_back(element.second);

      // Distribute the local points to the processor that owns them
      // on the triangulation
      auto my_bounding_box = GridTools::compute_mesh_predicate_bounding_box(
        triangulation, IteratorFilters::LocallyOwnedCell());

      auto global_bounding_boxes =
        Utilities::MPI::all_gather(MPI_COMM_WORLD, my_bounding_box);


      GridTools::Cache<dim, spacedim> cache(triangulation, mapping);

      auto distributed_tuple =
        GridTools::distributed_compute_point_locations(cache,
                                                       support_points_vec,
                                                       global_bounding_boxes);

      // Finally create the particles
      std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
                                                cell_iterators = std::get<0>(distributed_tuple);
      std::vector<std::vector<Point<dim>>>      dist_reference_points =
        std::get<1>(distributed_tuple);
      std::vector<std::vector<unsigned int>> dist_map =
        std::get<2>(distributed_tuple);
      std::vector<std::vector<Point<spacedim>>> dist_points =
        std::get<3>(distributed_tuple);
      std::vector<std::vector<unsigned int>> dist_procs =
        std::get<4>(distributed_tuple);

      // Count how many particles there are for current processor
      unsigned int n_particles = 0;
      for (unsigned int i = 0; i < dist_reference_points.size(); ++i)
        for (unsigned int j = 0; j < dist_reference_points[i].size(); ++j)
          n_particles++;

      // Gather the number per processor
      auto n_particles_per_proc =
        Utilities::MPI::all_gather(MPI_COMM_WORLD, n_particles);

      // Calculate all starting points locally
      std::vector<unsigned int> starting_points(
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD));
      for (unsigned int i = 0; i < starting_points.size(); ++i)
        std::accumulate(n_particles_per_proc.begin(),
                        n_particles_per_proc.begin() + i,
                        0u);



      // Create the multimap of particles
      std::multimap<typename Triangulation<dim, spacedim>::active_cell_iterator,
                    Particle<dim, spacedim>>
        particles;

      for (unsigned int i_cell = 0; i_cell < cell_iterators.size(); ++i_cell)
        {
          for (unsigned int i_particle = 0;
               i_particle < dist_points[i_cell].size();
               ++i_particle)
            {
              const unsigned int particle_id =
                dist_map[i_cell][i_particle] +
                starting_points[dist_procs[i_cell][i_particle]];

              particles.emplace(cell_iterators[i_cell],
                                Particle<dim, spacedim>(
                                  dist_points[i_cell][i_particle],
                                  dist_reference_points[i_cell][i_particle],
                                  particle_id));
            }
        }


      // Create particles from the list of point
      particle_handler.insert_particles(particles);
    }

  } // namespace Generators
} // namespace Particles

#endif // DEAL_II_WITH_P4EST

DEAL_II_NAMESPACE_CLOSE

#endif
