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

#include <deal.II/non_matching/quadrature_overlapped_grids.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

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


// Functors-like classes to describe boundary values, right hand side,
// analytical solution, if any.
template <int dim>
class RightHandSide : public Function<dim>
{
public:
  virtual double value(const Point<dim> & p,
                       const unsigned int component = 0) const override;
};



template <int dim>
double RightHandSide<dim>::value(const Point<dim> & p,
                                 const unsigned int component) const
{
  (void)p;
  (void)component;
  return 1.;
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

  /**
   * GridTools::Cache objects are used to cache all the necessary
   * information about a given triangulation, such as its Mapping, Bounding
   * Boxes, etc.
   *
   */
  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
  std::unique_ptr<GridTools::Cache<dim, spacedim>>      embedded_cache;

  /**
   * The coupling between the two grids is ultimately encoded in this
   * vector. Here the i-th entry stores a tuple for which the first two
   * elements are iterators to two cells from the space and embedded grid,
   * respectively, that intersect each other (up to a specified tolerance)
   * and a Quadrature object to integrate over that region.
   *
   *
   */
  std::vector<
    std::tuple<typename dealii::Triangulation<spacedim>::cell_iterator,
               typename dealii::Triangulation<dim, spacedim>::cell_iterator,
               dealii::Quadrature<spacedim>>>
    cells_and_quads;


  FE_Q<spacedim> space_fe;

  /**
   * The actual DoFHandler class.
   */
  DoFHandler<spacedim> space_dh;

  /**
   * According to the Triangulation type, we use a MappingFE or a MappingQ,
   * to make sure we can run the program both on a tria/tetra grid and on
   * quad/hex grids.
   */
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


  // ConvergenceTable error_table;



  /**
   * Choosing as embedded space the square $[-.0.45,0.45]^2$ and as
   * embedding space the square $[-1,1]^2$, with embedded value the
   * function $g(x,y)=1$, this is what we get
   * @image html Poisson_1_interface.png
   *
   *
   * Taking a manufactured smooth solution $u=\sin(2 \pi x) \sin(2 \pi y)$,
   * classical rates can be observed, as in the following table:
   * cells dofs   u_L2_norm    u_Linfty_norm    u_H1_norm
     256  289 5.851e-02    - 8.125e-02    - 2.015e+00    -
    1024 1089 1.436e-02 2.12 2.160e-02 2.00 1.007e+00 1.05
    4096 4225 3.605e-03 2.04 5.519e-03 2.01 5.037e-01 1.02
   */
  mutable DataOut<spacedim, spacedim> data_out;

  /**
   * Level of log verbosity.
   */
  unsigned int console_level = 1;

  /**
   * The penalty parameter which multiplies Nitsche's terms. In this program
   * it is defaulted to 100.0
   */

  double penalty = 100.0;

  unsigned int n_refinement_cycles = 4;
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

  GridGenerator::hyper_cube(embedded_triangulation, -.45, .35);
  GridTools::rotate(numbers::PI_4, 2, embedded_triangulation);
  space_triangulation.refine_global(1);
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
  // boundary_conditions.apply_essential_boundary_conditions(*mapping,
  //                                                         space_dh,
  //                                                         space_constraints);

  VectorTools::interpolate_boundary_values(
    space_dh,
    0,
    Functions::ZeroFunction<spacedim>(),
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

    // Add the Nitsche's contribution to the system matrix. The coefficient
    // that multiplies the inner product is equal to 2.0, and the penalty is
    // set to 100.0.
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
        RightHandSide<spacedim>(),
        Functions::ConstantFunction<spacedim>(2.0),
        penalty);
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
  std::ofstream output("solution_nitsche" + std::to_string(cycle) + ".vtu");
  data_out.write_vtu(output);

  {
    std::ofstream output_test_space("space_grid.vtk");
    GridOut().write_vtk(space_triangulation, output_test_space);
    std::ofstream output_test_embedded("embedded_grid.vtk");
    GridOut().write_vtk(embedded_triangulation, output_test_embedded);
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

      // Here we compute all the things we need to assemble the Nitsche's
      // contributions, namely the two cached triangulations and a degree to
      // integrate over the intersections.
      std::cout << "Start collecting quadratures" << std::endl;
      cells_and_quads = NonMatching::collect_quadratures_on_overlapped_grids(
        *space_cache, *embedded_cache, 2 * space_fe.degree + 1);
      std::cout << "Collected quadratures" << std::endl;

      setup_system();
      assemble_system();
      solve();

      // error_table.error_from_exact(space_dh, solution, exact_solution);
      output_results(cycle);

      if (cycle < n_refinement_cycles - 1)
        space_triangulation.refine_global(1);
    }
  // Make sure we output the error table after the last cycle
  // error_table.output_table(std::cout);
}



int main()
{
  PoissonNitscheInterface<3> problem;
  problem.run();
}
