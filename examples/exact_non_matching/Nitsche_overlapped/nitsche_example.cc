// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2022 by the deal.II authors
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



#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/non_matching/quadrature_overlapped_grids.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>



using namespace dealii;


const double R = .45;
// Functors-like classes to describe boundary values, right hand side,
// analytical solution, if any.
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};



template <>
double RightHandSide<3>::value(const Point<3> &   p,
                               const unsigned int component) const
{
  // (void)p;
  (void)component;
  // return 1.;
  return 12. * numbers::PI * numbers::PI *
         (std::sin(2. * numbers::PI * p[0]) *
          std::sin(2. * numbers::PI * p[1]) *
          std::sin(2. * numbers::PI * p[2]));
}



template <>
double RightHandSide<2>::value(const Point<2> &   p,
                               const unsigned int component) const
{
  // (void)p;
  (void)component;
  return 0.;
  // return 8. * numbers::PI * numbers::PI *
  //        (std::sin(2. * numbers::PI * p[0]) *
  //         std::sin(2. * numbers::PI * p[1]));
}



// template <int dim>
// class BoundaryCondition : public Function<dim>
// {
// public:
//   virtual double value(const Point<dim>  &p,
//                        const unsigned int component = 0) const override;
// };



// template <int dim>
// double BoundaryCondition<dim>::value(const Point<dim>  &p,
//                                      const unsigned int component) const
// {
//   (void)p;
//   (void)component;
//   return 1.5;
// }



template <int dim>
class Solution : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim>
  gradient(const Point<dim> & p,
           const unsigned int component = 0) const override;
};



template <>
double Solution<3>::value(const Point<3> &p, const unsigned int component) const
{
  (void)component;
  return std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]) *
         std::sin(2. * numbers::PI * p[2]);
}



template <>
double Solution<2>::value(const Point<2> &p, const unsigned int component) const
{
  (void)component;
  const double r = p.norm();
  return (r <= R) ? p[0] : ((R * R) / (r * r)) * p[0];
  // std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);
}


template <>
Tensor<1, 3> Solution<3>::gradient(const Point<3> &   p,
                                   const unsigned int component) const
{
  (void)component;
  Tensor<1, 3> gradient;
  gradient[0] = std::cos(2. * numbers ::PI * p[0]) *
                std::sin(2. * numbers::PI * p[1]) *
                std::sin(2. * numbers::PI * p[2]);

  gradient[1] = std::sin(2. * numbers ::PI * p[0]) *
                std::cos(2. * numbers::PI * p[1]) *
                std::sin(2. * numbers::PI * p[2]);

  gradient[2] = std::sin(2. * numbers ::PI * p[0]) *
                std::sin(2. * numbers::PI * p[1]) *
                std::cos(2. * numbers::PI * p[2]);

  return 2. * numbers::PI * gradient;
}



template <>
Tensor<1, 2> Solution<2>::gradient(const Point<2> &   p,
                                   const unsigned int component) const
{
  (void)component;
  const double r = p.norm();

  Tensor<1, 2> gradient;
  gradient[0] =
    (r <= R) ? 1. : (R * p[0] * p[0]) / (std::pow(p.norm_square(), 1.5));

  gradient[1] =
    (r <= R) ? 0. : (R * p[0] * p[1]) / (std::pow(p.norm_square(), 1.5));
  return gradient;
  // gradient[0] =
  //   std::cos(2. * numbers ::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);

  // gradient[1] =
  //   std::sin(2. * numbers ::PI * p[0]) * std::cos(2. * numbers::PI * p[1]);

  // return 2. * numbers::PI * gradient;
}



template <int dim, int spacedim = dim>
class PoissonNitscheInterface
{
public:
  PoissonNitscheInterface();
  void run();

private:
  void generate_grids();

  void setup_system();

  void assemble_system();

  void solve();

  void output_results(const unsigned cycle) const;

  /**
   * The actual triangulations. Here with "space_triangulation" we refer to
   * the original domain \Omega, also called the ambient space, while with
   * embedded we refer to the immersed domain, the one where we want to
   * impose a constraint.
   *
   */
  Triangulation<spacedim>      space_triangulation;
  Triangulation<dim, spacedim> embedded_triangulation;

  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
  std::unique_ptr<GridTools::Cache<dim, spacedim>>      embedded_cache;


  std::vector<
    std::tuple<typename dealii::Triangulation<spacedim>::cell_iterator,
               typename dealii::Triangulation<dim, spacedim>::cell_iterator,
               dealii::Quadrature<spacedim>>>
    cells_and_quads;


  FE_Q<spacedim> space_fe;

  DoFHandler<spacedim> space_dh;

  MappingQ1<spacedim> mapping;

  AffineConstraints<double> space_constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;
  Vector<double>            solution;
  Vector<double>            system_rhs;


  /**
   * The actual function to use as a forcing term.
   */
  // Function<spacedim> forcing_term;


  /**
   * This is the value we want to impose on the embedded domain.
   *
   */
  // Functions<spacedim> embedded_value;


  /**
   * The coefficient in front of the Nitsche contribution to the stiffness
   * matrix.
   *
   */
  // Function<spacedim> nitsche_coefficient;

  /**
   * The actual function to use as a exact solution when computing the
   * errors.
   * */
  // Function<spacedim> exact_solution;


  // BoundaryConditions<spacedim> boundary_conditions;

  // mutable TimerOutput timer;

  mutable ConvergenceTable convergence_table;

  mutable DataOut<spacedim> data_out;


  /**
   * The penalty parameter which multiplies Nitsche's terms. In this program
   * it is defaulted to 100.0
   */

  double penalty = 10.0;

  unsigned int n_refinement_cycles = 5;
};



template <int dim, int spacedim>
PoissonNitscheInterface<dim, spacedim>::PoissonNitscheInterface()
  : space_fe(1)
  , space_dh(space_triangulation)
{}


template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::generate_grids()
{
  // TimerOutput::Scope timer_section(timer, "Generate grids");

  GridGenerator::hyper_cube(space_triangulation, -1., 1.);

  if constexpr (dim == 3 && spacedim == 3)
    {
      GridGenerator::hyper_cube(embedded_triangulation, 0.42, 0.66);
      GridTools::rotate(Tensor<1, 3>({0, 1, 0}),
                        numbers::PI_4,
                        embedded_triangulation);
    }
  else if constexpr (dim == 1 && spacedim == 2)
    {
      GridGenerator::hyper_sphere(embedded_triangulation, {}, R);
      embedded_triangulation.refine_global(2); // 5
      space_triangulation.refine_global(2);    // 2
    }
  else if constexpr (dim == 2 && spacedim == 2)
    {
      GridGenerator::hyper_ball(embedded_triangulation, {}, R, false);
      embedded_triangulation.refine_global(2);
      space_triangulation.refine_global(1);
    }
  else if constexpr (dim == 2 && spacedim == 3)
    {
      GridGenerator::hyper_cube(embedded_triangulation, -0.45, .35);
      embedded_triangulation.refine_global(2);
      // GridTools::rotate(Tensor<1, 3>({0, 1, 0}),
      //                   numbers::PI_4,
      //                   embedded_triangulation);
    }
  space_triangulation.refine_global(2);
  // We create unique pointers to cached triangulations. This This objects
  // will be necessary to compute the the Quadrature formulas on the
  // intersection of the cells.
  space_cache =
    std::make_unique<GridTools::Cache<spacedim, spacedim>>(space_triangulation);
  embedded_cache =
    std::make_unique<GridTools::Cache<dim, spacedim>>(embedded_triangulation);
}



template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::setup_system()
{
  // TimerOutput::Scope timer_section(timer, "Setup system");
  std::cout << "System setup" << std::endl;


  // We propagate the information about the constants to all functions of
  // the problem, so that constants can be used within the functions

  space_dh.distribute_dofs(space_fe);
  std::cout << "Number of dofs in space: " << space_dh.n_dofs() << std::endl;

  space_constraints.clear();
  DoFTools::make_hanging_node_constraints(space_dh, space_constraints);

  // This is where we apply essential boundary conditions.
  VectorTools::interpolate_boundary_values(
    space_dh,
    0,
    /*Functions::ZeroFunction<spacedim>(),*/
    Solution<spacedim>(),
    space_constraints); // zero Dirichlet on the boundary

  space_constraints.close();
  DynamicSparsityPattern dsp(space_dh.n_dofs());
  DoFTools::make_sparsity_pattern(space_dh, dsp, space_constraints, false);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  solution.reinit(space_dh.n_dofs());
  system_rhs.reinit(space_dh.n_dofs());
}



template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::assemble_system()
{
  {
    // TimerOutput::Scope timer_section(timer, "Assemble system");
    std::cout << "Assemble system" << std::endl;


    QGauss<spacedim>             quadrature_formula(2 * space_fe.degree + 1);
    FEValues<spacedim, spacedim> fe_values(mapping,
                                           space_fe,
                                           quadrature_formula,
                                           update_values | update_gradients |
                                             update_quadrature_points |
                                             update_JxW_values);

    const unsigned int      dofs_per_cell = space_fe.n_dofs_per_cell();
    FullMatrix<double>      cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>          cell_rhs(dofs_per_cell);
    RightHandSide<spacedim> rhs;
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : space_dh.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix          = 0;
        cell_rhs             = 0;
        const auto &q_points = fe_values.get_quadrature_points();
        for (const unsigned int q_index : fe_values.quadrature_point_indices())
          {
            for (const unsigned int i : fe_values.dof_indices())
              for (const unsigned int j : fe_values.dof_indices())
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                   fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                   fe_values.JxW(q_index));           // dx
            for (const unsigned int i : fe_values.dof_indices())
              cell_rhs(i) +=
                (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                                     /*  forcing_term.value(
                                                         fe_values.quadrature_point(q_index)) * // f(x_q)*/
                 rhs.value(q_points[q_index]) * fe_values.JxW(q_index)); // dx
          }

        cell->get_dof_indices(local_dof_indices);
        space_constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
      }
  }


  std::cout << "Assemble Nitsche contributions" << std::endl;
  {
    // TimerOutput::Scope timer_section(timer, "Assemble Nitsche terms");

    // Add Nitsche's contribution to the system matrix.
    NonMatching::
      assemble_nitsche_with_exact_intersections<spacedim, dim, spacedim>(
        space_dh,
        cells_and_quads,
        system_matrix,
        space_constraints,
        ComponentMask(),
        MappingQ1<spacedim, spacedim>(),
        Functions::ConstantFunction<spacedim>(2.0),
        penalty);

    // Add the Nitsche's contribution to the rhs. The embedded value is
    // parsed from the parameter file, while we have again the constant 2.0
    // in front of that term, parsed as above from command line. Finally, we
    // have the penalty parameter as before.
    NonMatching::
      create_nitsche_rhs_with_exact_intersections<spacedim, dim, spacedim>(
        space_dh,
        cells_and_quads,
        system_rhs,
        space_constraints,
        MappingQ1<spacedim>(),
        Solution<spacedim>(),
        Functions::ConstantFunction<spacedim>(2.0),
        penalty);


    // FE_Q<dim, spacedim>       embedded_fe(1);
    // DoFHandler<dim, spacedim> embedded_dh(embedded_triangulation);
    // embedded_dh.distribute_dofs(embedded_fe);



    // NonMatching::create_coupling_mass_matrix_nitsche(*space_cache,
    //                                                  space_dh,
    //                                                  embedded_dh,
    //                                                  QGauss<dim>(
    //                                                    2 * space_fe.degree +
    //                                                    1),
    //                                                  system_matrix,
    //                                                  system_rhs,
    //                                                  Solution<spacedim>(),
    //                                                  mapping,
    //                                                  MappingQ1<dim,
    //                                                  spacedim>(),
    //                                                  space_constraints);
  }
}


// We solve the resulting system as done in the classical Poisson example.
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::solve()
{
  // TimerOutput::Scope timer_section(timer, "Solve system");
  std::cout << "Solve system" << std::endl;

  PreconditionJacobi<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix);
  const auto A = linear_operator<Vector<double>>(system_matrix);

  ReductionControl         reduction_control(2000, 1.0e-18, 1.0e-10);
  SolverCG<Vector<double>> solver(reduction_control);

  const auto Ainv = inverse_operator(A, solver, preconditioner);
  solution        = Ainv * system_rhs;

  std::cout << "Solver converged in: " << reduction_control.last_step()
            << " iterations" << std::endl;
  space_constraints.distribute(solution);
}



// Finally, we output the solution living in the embedding space, just
// like all the other programs.
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::output_results(
  const unsigned cycle) const
{
  // TimerOutput::Scope timer_section(timer, "Output results");
  std::cout << "Output results" << std::endl;
  data_out.clear();
  data_out.attach_dof_handler(space_dh);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution_nitsche" + std::to_string(dim) +
                       std::to_string(spacedim) + std::to_string(cycle) +
                       ".vtu");
  data_out.write_vtu(output);
  {
    Vector<double> difference_per_cell(space_triangulation.n_active_cells());
    VectorTools::integrate_difference(space_dh,
                                      solution,
                                      Solution<spacedim>(),
                                      difference_per_cell,
                                      QGauss<spacedim>(2 * space_fe.degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(space_triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);

    difference_per_cell.reinit(
      space_triangulation
        .n_active_cells()); // zero out again to store the H1 error
    VectorTools::integrate_difference(space_dh,
                                      solution,
                                      Solution<spacedim>(),
                                      difference_per_cell,
                                      QGauss<spacedim>(2 * space_fe.degree + 1),
                                      VectorTools::H1_norm);
    const double H1_error =
      VectorTools::compute_global_error(space_triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", space_triangulation.n_active_cells());
    convergence_table.add_value("dofs", space_dh.n_dofs());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
  }

  {
    if (cycle == 3)
      {
        std::ofstream output_test_space("space_grid_cycle3.vtk");
        GridOut().write_vtk(space_triangulation, output_test_space);
        std::ofstream output_test_embedded("embedded_grid_cycle3.vtk");
        GridOut().write_vtk(embedded_triangulation, output_test_embedded);
      }
  }
}


// The run() method here differs only in the call to
// NonMatching::compute_intersection().
template <int dim, int spacedim>
void PoissonNitscheInterface<dim, spacedim>::run()
{
  generate_grids();
  for (unsigned int cycle = 0; cycle < n_refinement_cycles; ++cycle)
    {
      std::cout << "Cycle: " << cycle << std::endl;

      // HCompute all the things we need to assemble the Nitsche's
      // contributions, namely the two cached triangulations and a degree to
      // integrate over the intersections.
      std::cout << "Start collecting quadratures" << std::endl;
      cells_and_quads = NonMatching::collect_quadratures_on_overlapped_grids(
        *space_cache, *embedded_cache, 2 * space_fe.degree + 1);
      std::cout << "Collected quadratures" << std::endl;

      double sum = 0.;
      for (const auto &p : cells_and_quads)
        {
          auto quad = std::get<2>(p);
          sum += std::accumulate(quad.get_weights().begin(),
                                 quad.get_weights().end(),
                                 0.);
        }
      std::cout << "Area/Measure: " << sum << std::endl;

      setup_system();
      assemble_system();
      solve();

      // error_table.error_from_exact(space_dh, solution, exact_solution);
      output_results(cycle);

      if (cycle < n_refinement_cycles - 1)
        space_triangulation.refine_global(1);
      embedded_triangulation.refine_global(1);
      cells_and_quads.clear();
    }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.evaluate_convergence_rates(
    "L2", ConvergenceTable::reduction_rate_log2);
  convergence_table.evaluate_convergence_rates(
    "H1", ConvergenceTable::reduction_rate_log2);
  convergence_table.write_text(std::cout);
}



int main()
{
  try
    {
      {
        std::cout << "Solving in 1D/2D" << std::endl;
        PoissonNitscheInterface<1, 2> problem;
        problem.run();
      }
      {
        // std::cout << "Solving in 2D/2D" << std::endl;
        // PoissonNitscheInterface<2> problem;
        // problem.run();
      } {
        // std::cout << "Solving in 2D/3D" << std::endl;
        // PoissonNitscheInterface<2, 3> problem;
        // problem.run();
      } {
        // std::cout << "Solving in 3D/3D" << std::endl;
        // PoissonNitscheInterface<3> problem;
        // problem.run();
      }
      return 0;
    }
  catch (const std::exception &e)
    {
      std::cerr << e.what() << '\n';
      return 1;
    }
}
