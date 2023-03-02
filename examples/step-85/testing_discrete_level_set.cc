/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */



#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/base/bounding_box.h>
#include <deal.II/numerics/rtree.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include "../exact_non_matching/NonMatching_utilities.h"



namespace Step85
{
  using namespace dealii;

  template <int dim>
  class LaplaceSolver
  {
  public:
    LaplaceSolver();

    void run();

  private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve();

    void output_results() const;

    double compute_L2_error() const;

    bool face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

    const unsigned int fe_degree;

    const Functions::ConstantFunction<dim> rhs_function;
    const Functions::ConstantFunction<dim> boundary_condition;

    Triangulation<dim> triangulation;

    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double>  level_set;

    hp::FECollection<dim> fe_collection;
    DoFHandler<dim>       dof_handler;
    Vector<double>        solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> stiffness_matrix;
    Vector<double>       rhs;
  };



  template <int dim>
  LaplaceSolver<dim>::LaplaceSolver()
    : fe_degree(1)
    , rhs_function(4.0)
    , boundary_condition(1.0)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , mesh_classifier(level_set_dof_handler, level_set)
  {}



  template <int dim>
  void LaplaceSolver<dim>::make_grid()
  {
    std::cout << "Creating background mesh" << std::endl;

    GridGenerator::hyper_cube(triangulation, -1.21, 1.21);
    triangulation.refine_global(2);
  }



  template <int dim>
  void LaplaceSolver<dim>::setup_discrete_level_set()
  {
    std::cout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;

    // GridGenerator::hyper_ball(embedded_tria, {Cx, Cy}, R);
    Triangulation<dim - 1, dim> embedded_triangulation;
    GridGenerator::hyper_sphere(embedded_triangulation, Point<dim>(), 1.);
    embedded_triangulation.refine_global(10);

    NonMatchingUtilities::CDT tr;
    using Point2 = typename NonMatchingUtilities::Point2;
    Point<dim> center{};
    for (const auto &cell : embedded_triangulation.active_cell_iterators())
      {
        tr.insert_constraint(
          CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(0)),
          CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(1)));

        tr.insert_constraint(
          CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(1)),
          CGALWrappers::dealii_point_to_cgal_point<Point2>(center));

        tr.insert_constraint(
          CGALWrappers::dealii_point_to_cgal_point<Point2>(center),
          CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(0)));
      }

    mark_domains(tr);


    GridTools::Cache<dim - 1, dim> cache(embedded_triangulation);
    auto                           tree = cache.get_cell_bounding_boxes_rtree();

    NonMatchingUtilities::DiscreteLevelSet<dim, decltype(tree)>
      discrete_level_set(&tree, tr);

    { // Sanity checks
      std::cout << signed_distance_sphere.value(Point<2>{}) << " and "
                << discrete_level_set.value(Point<2>{}) << std::endl;

      std::cout << discrete_level_set.value(Point<2>{1.1, 1.1}) << " and "
                << signed_distance_sphere.value(Point<2>{1.1, 1.1})
                << std::endl;

      std::cout << discrete_level_set.value(Point<2>{1.2, 1.2}) << " and "
                << signed_distance_sphere.value(Point<2>{1.2, 1.2})
                << std::endl;

      std::cout << discrete_level_set.value(Point<2>{0.5, 0.5}) << " and "
                << signed_distance_sphere.value(Point<2>{0.5, 0.5})
                << std::endl;

      std::cout << discrete_level_set.value(Point<2>{0.8, 0.8}) << " and "
                << signed_distance_sphere.value(Point<2>{0.8, 0.8})
                << std::endl;
    }

    VectorTools::interpolate(level_set_dof_handler,
                             discrete_level_set,
                             level_set);
  }



  enum ActiveFEIndex
  {
    lagrange = 0,
    nothing  = 1
  };

  template <int dim>
  void LaplaceSolver<dim>::distribute_dofs()
  {
    std::cout << "Distributing degrees of freedom" << std::endl;

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::outside)
          cell->set_active_fe_index(ActiveFEIndex::nothing);
        else
          cell->set_active_fe_index(ActiveFEIndex::lagrange);
      }

    dof_handler.distribute_dofs(fe_collection);
  }



  template <int dim>
  void LaplaceSolver<dim>::initialize_matrices()
  {
    std::cout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto &       cell,
                                            const unsigned int face_index) {
      return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }



  template <int dim>
  bool LaplaceSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  }



  template <int dim>
  void LaplaceSolver<dim>::assemble_system()
  {
    std::cout << "Assembling" << std::endl;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter   = 0.5;
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

    const QGauss<dim - 1>  face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                 update_JxW_values |
                                                 update_normal_vectors);


    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        local_stiffness = 0;
        local_rhs       = 0;

        const double cell_side_length = cell->minimum_vertex_distance();

        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  local_rhs(i) += rhs_function.value(point) *
                                  inside_fe_values->shape_value(i, q) *
                                  inside_fe_values->JxW(q);
                }
            }

        const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          (-normal * surface_fe_values->shape_grad(i, q) *
                             surface_fe_values->shape_value(j, q) +
                           -normal * surface_fe_values->shape_grad(j, q) *
                             surface_fe_values->shape_value(i, q) +
                           nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) *
                             surface_fe_values->shape_value(j, q)) *
                          surface_fe_values->JxW(q);
                      }
                    local_rhs(i) +=
                      boundary_condition.value(point) *
                      (nitsche_parameter / cell_side_length *
                         surface_fe_values->shape_value(i, q) -
                       normal * surface_fe_values->shape_grad(i, q)) *
                      surface_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        stiffness_matrix.add(local_dof_indices, local_stiffness);
        rhs.add(local_dof_indices, local_rhs);

        for (unsigned int f : cell->face_indices())
          if (face_has_ghost_penalty(cell, f))
            {
              const unsigned int invalid_subface =
                numbers::invalid_unsigned_int;

              fe_interface_values.reinit(cell,
                                         f,
                                         invalid_subface,
                                         cell->neighbor(f),
                                         cell->neighbor_of_neighbor(f),
                                         invalid_subface);

              const unsigned int n_interface_dofs =
                fe_interface_values.n_current_interface_dofs();
              FullMatrix<double> local_stabilization(n_interface_dofs,
                                                     n_interface_dofs);
              for (unsigned int q = 0;
                   q < fe_interface_values.n_quadrature_points;
                   ++q)
                {
                  const Tensor<1, dim> normal = fe_interface_values.normal(q);
                  for (unsigned int i = 0; i < n_interface_dofs; ++i)
                    for (unsigned int j = 0; j < n_interface_dofs; ++j)
                      {
                        local_stabilization(i, j) +=
                          .5 * ghost_parameter * cell_side_length * normal *
                          fe_interface_values.jump_in_shape_gradients(i, q) *
                          normal *
                          fe_interface_values.jump_in_shape_gradients(j, q) *
                          fe_interface_values.JxW(q);
                      }
                }

              const std::vector<types::global_dof_index>
                local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

              stiffness_matrix.add(local_interface_dof_indices,
                                   local_stabilization);
            }
      }
  }


  template <int dim>
  void LaplaceSolver<dim>::solve()
  {
    std::cout << "Solving system" << std::endl;

    const unsigned int max_iterations = solution.size();
    SolverControl      solver_control(max_iterations);
    SolverCG<>         solver(solver_control);
    solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
  }



  template <int dim>
  void LaplaceSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output("step-85.vtu");
    data_out.write_vtu(output);
  }



  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim> & point,
                 const unsigned int component = 0) const override;
  };



  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> & point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return 1. - 2. / dim * (point.norm_square() - 1.);
  }



  template <int dim>
  double LaplaceSolver<dim>::compute_L2_error() const
  {
    std::cout << "Computing L2 error" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    const AnalyticalSolution<dim> analytical_solution;
    double                        error_L2_squared = 0;

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
      {
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            fe_values->get_function_values(solution, solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values.at(q) - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
      }

    return std::sqrt(error_L2_squared);
  }



  template <int dim>
  void LaplaceSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 3;

    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        std::cout << "Refinement cycle " << cycle << std::endl;
        triangulation.refine_global(1);
        setup_discrete_level_set();
        std::cout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
        distribute_dofs();
        initialize_matrices();
        assemble_system();
        solve();
        if (cycle == 1)
          output_results();
        const double error_L2 = compute_L2_error();
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("Mesh size", cell_side_length);
        convergence_table.add_value("L2-Error", error_L2);

        convergence_table.evaluate_convergence_rates(
          "L2-Error", ConvergenceTable::reduction_rate_log2);
        convergence_table.set_scientific("L2-Error", true);

        std::cout << std::endl;
        convergence_table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

} // namespace Step85



int main()
{
  const int dim = 2;

  Step85::LaplaceSolver<dim> laplace_solver;
  laplace_solver.run();
}