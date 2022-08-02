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



#include <deal.II/lac/linear_operator_tools.h>

#include <deal.II/base/parsed_function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/non_matching/quadrature_overlapped_grids.h>
#include <deal.II/non_matching/coupling.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/grid/grid_out.h>

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

const double R = .45;

using namespace dealii;


// // Functors-like classes to describe boundary values, right hand side,
// // analytical solution, if any.
// template <int dim>
// class RightHandSide : public Function<dim>
// {
// public:
//   virtual double value(const Point<dim> & p,
//                        const unsigned int component = 0) const override;
// };



// template <>
// double RightHandSide<3>::value(const Point<3> &   p,
//                                const unsigned int component) const
// {
//   // (void)p;
//   (void)component;
//   // return 1.;
//   return 0.; // 12. * numbers::PI * numbers::PI *
//   //    (std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI *
//   p[1]) *
//   //     std::sin(2. * numbers::PI * p[2]));
// }



// template <>
// double RightHandSide<2>::value(const Point<2> &   p,
//                                const unsigned int component) const
// {
//   // (void)p;
//   (void)component;
//   // return 1.;
//   return /*0.;*/ 8. * numbers::PI * numbers::PI *
//          (std::sin(2. * numbers::PI * p[0]) *
//           std::sin(2. * numbers::PI * p[1]));
// }



// // template <int dim>
// // class BoundaryCondition : public Function<dim>
// // {
// // public:
// //   virtual double value(const Point<dim>  &p,
// //                        const unsigned int component = 0) const override;
// // };



// // template <int dim>
// // double BoundaryCondition<dim>::value(const Point<dim>  &p,
// //                                      const unsigned int component) const
// // {
// //   (void)p;
// //   (void)component;
// //   return 1.5;
// // }



// template <int dim>
// class Solution : public Function<dim>
// {
// public:
//   virtual double value(const Point<dim> & p,
//                        const unsigned int component = 0) const override;

//   virtual Tensor<1, dim>
//   gradient(const Point<dim> & p,
//            const unsigned int component = 0) const override;
// };



// template <>
// double Solution<3>::value(const Point<3> &p, const unsigned int component)
// const
// {
//   (void)component;
//   const double r = p.norm();
//   return /*(r <= R) ?
//            p[0] :
//            ((R * R) / (r * r)) * p[0];*/
//     std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]) *
//     std::sin(2. * numbers::PI * p[2]);
// }



// template <>
// double Solution<2>::value(const Point<2> &p, const unsigned int component)
// const
// {
//   (void)component;
//   const double r = p.norm();
//   return /*(r <= R) ?
//            p[0] :
//            ((R * R) / (r * r)) * p[0];*/
//     std::sin(2. * numbers::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);
// }


// template <>
// Tensor<1, 3> Solution<3>::gradient(const Point<3> &   p,
//                                    const unsigned int component) const
// {
//   (void)component;
//   Tensor<1, 3> gradient;
//   gradient[0] = std::cos(2. * numbers ::PI * p[0]) *
//                 std::sin(2. * numbers::PI * p[1]) *
//                 std::sin(2. * numbers::PI * p[2]);

//   gradient[1] = std::sin(2. * numbers ::PI * p[0]) *
//                 std::cos(2. * numbers::PI * p[1]) *
//                 std::sin(2. * numbers::PI * p[2]);

//   gradient[2] = std::sin(2. * numbers ::PI * p[0]) *
//                 std::sin(2. * numbers::PI * p[1]) *
//                 std::cos(2. * numbers::PI * p[2]);

//   return 2. * numbers::PI * gradient;
// }



// template <>
// Tensor<1, 2> Solution<2>::gradient(const Point<2> &   p,
//                                    const unsigned int component) const
// {
//   (void)component;
//   Tensor<1, 2> gradient;
//   gradient[0] =
//     std::cos(2. * numbers ::PI * p[0]) * std::sin(2. * numbers::PI * p[1]);

//   gradient[1] =
//     std::sin(2. * numbers ::PI * p[0]) * std::cos(2. * numbers::PI * p[1]);

//   return 2. * numbers::PI * gradient;
// }



template <int dim, int spacedim = dim>
class PoissonDLM
{
public:
  class Parameters : public ParameterAcceptor
  {
  public:
    Parameters();

    unsigned int n_refinement_cycles = 6;

    unsigned int delta_refinement_cycles = 2;

    unsigned int space_initial_global_refinements = 4;

    unsigned int embedded_initial_global_refinements = 2;

    bool initialized = false;

    std::string coupling_strategy;
  };

  PoissonDLM(const Parameters &parameters);

  void run();

private:
  const Parameters &parameters;

  void setup_grids_and_dofs();

  void setup_coupling();

  void setup_embedded_dofs();

  void setup_space_dofs();

  void adjust_grids();

  void assemble_system();

  void solve();

  void output_results(const unsigned cycle) const;


  Triangulation<spacedim>      space_triangulation;
  Triangulation<dim, spacedim> embedded_triangulation;

  std::unique_ptr<GridTools::Cache<spacedim, spacedim>> space_cache;
  std::unique_ptr<GridTools::Cache<dim, spacedim>>      embedded_cache;


  std::vector<
    std::tuple<typename dealii::Triangulation<spacedim>::cell_iterator,
               typename dealii::Triangulation<dim, spacedim>::cell_iterator,
               dealii::Quadrature<spacedim>>>
    cells_and_quads;


  FE_Q<spacedim>      space_fe;
  FE_Q<dim, spacedim> embedded_fe;

  /**
   * The actual DoFHandler class.
   */
  std::unique_ptr<DoFHandler<spacedim>>      space_dh;
  std::unique_ptr<DoFHandler<dim, spacedim>> embedded_dh;


  MappingQ1<spacedim>      space_mapping;
  MappingQ1<dim, spacedim> embedded_mapping;


  AffineConstraints<double> space_constraints;
  AffineConstraints<double> embedded_constraints;
  SparsityPattern           stiffness_sparsity_pattern;
  SparsityPattern           coupling_sparsity_pattern;
  SparseMatrix<double>      stiffness_matrix;
  SparseMatrix<double>      coupling_matrix;
  Vector<double>            space_rhs;
  Vector<double>            embedded_rhs;


  Vector<double> solution;
  Vector<double> lambda;


  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> rhs_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>> solution_function;

  ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
    boundary_condition_function;

  // mutable TimerOutput timer;

  mutable ConvergenceTable convergence_table;

  mutable DataOut<spacedim> data_out;

  unsigned int cycle;
};


template <int dim, int spacedim>
PoissonDLM<dim, spacedim>::Parameters::Parameters()
  : ParameterAcceptor("/Distributed Lagrange<" + Utilities::int_to_string(dim) +
                      "," + Utilities::int_to_string(spacedim) + ">/")
{
  add_parameter("Number of refinement cycles", n_refinement_cycles);

  add_parameter("Number of space initial refinement cycles",
                space_initial_global_refinements);

  add_parameter("Number of embedded initial refinement cycles",
                embedded_initial_global_refinements);

  add_parameter("Local refinements steps near embedded domain",
                delta_refinement_cycles);

  add_parameter("Coupling strategy", coupling_strategy);

  parse_parameters_call_back.connect([&]() -> void { initialized = true; });
}

template <int dim, int spacedim>
PoissonDLM<dim, spacedim>::PoissonDLM(const Parameters &parameters)
  : parameters(parameters)
  , space_fe(1)
  , embedded_fe(1)
  , rhs_function("Right hand side")
  , solution_function("Solution")
  , boundary_condition_function("Boundary condition")
{
  rhs_function.declare_parameters_call_back.connect(
    []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

  solution_function.declare_parameters_call_back.connect(
    []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });

  boundary_condition_function.declare_parameters_call_back.connect(
    []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });
}


template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::setup_grids_and_dofs()
{
  // TimerOutput::Scope timer_section(timer, "Generate grids");
  if (cycle == 0)
    {
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
          space_triangulation.refine_global(
            parameters.space_initial_global_refinements); // 4
          embedded_triangulation.refine_global(
            parameters.embedded_initial_global_refinements); // 2
        }
      else if constexpr (dim == 2 && spacedim == 2)
        {
          embedded_triangulation.refine_global(2);
          space_triangulation.refine_global(3);
        }
      else if constexpr (dim == 2 && spacedim == 3)
        {
          GridGenerator::hyper_sphere(embedded_triangulation, {}, R);
          embedded_triangulation.refine_global(2);
          space_triangulation.refine_global(3);
        }
    }

  space_cache =
    std::make_unique<GridTools::Cache<spacedim, spacedim>>(space_triangulation);
  embedded_cache =
    std::make_unique<GridTools::Cache<dim, spacedim>>(embedded_triangulation);

  setup_embedded_dofs();

  // adjust_grids();
  const double embedded_space_maximal_diameter =
    GridTools::maximal_cell_diameter(embedded_triangulation, embedded_mapping);
  double embedding_space_minimal_diameter =
    GridTools::minimal_cell_diameter(space_triangulation, space_mapping);

  std::cout << "Embedding minimal diameter: "
            << embedding_space_minimal_diameter
            << ", embedded maximal diameter: "
            << embedded_space_maximal_diameter << ", ratio: "
            << embedded_space_maximal_diameter /
                 embedding_space_minimal_diameter
            << std::endl;

  // AssertThrow(embedded_space_maximal_diameter <
  //               embedding_space_minimal_diameter,
  //             ExcMessage(
  //               "The embedding grid is too refined (or the embedded grid "
  //               "is too coarse). Adjust the parameters so that the minimal "
  //               "grid size of the embedding grid is larger "
  //               "than the maximal grid size of the embedded grid."));
  setup_space_dofs();
}



template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::adjust_grids()
{
  namespace bgi = boost::geometry::index;
  for (unsigned int i = 0; i < parameters.delta_refinement_cycles; ++i)
    {
      const auto &tree =
        space_cache->get_locally_owned_cell_bounding_boxes_rtree();

      const auto &embedded_tree =
        embedded_cache->get_cell_bounding_boxes_rtree();

      for (const auto &[embedded_box, embedded_cell] : embedded_tree)
        for (const auto &[space_box, space_cell] :
             tree | bgi::adaptors::queried(bgi::intersects(embedded_box)))
          space_cell->set_refine_flag();
      space_triangulation.execute_coarsening_and_refinement();
    }
}



template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::setup_space_dofs()
{
  // Setup space DoFs
  space_dh = std::make_unique<DoFHandler<spacedim>>(space_triangulation);
  space_dh->distribute_dofs(space_fe);
  std::cout << "Number of dofs in space: " << space_dh->n_dofs() << std::endl;
  space_constraints.clear();
  DoFTools::make_hanging_node_constraints(*space_dh, space_constraints);

  // This is where we apply essential boundary conditions.
  VectorTools::interpolate_boundary_values(
    *space_dh,
    0,
    solution_function,
    space_constraints); // zero Dirichlet on the boundary

  space_constraints.close();


  DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
  DoFTools::make_sparsity_pattern(*space_dh, dsp, space_constraints);
  stiffness_sparsity_pattern.copy_from(dsp);
  stiffness_matrix.reinit(stiffness_sparsity_pattern);
  solution.reinit(space_dh->n_dofs());
  space_rhs.reinit(space_dh->n_dofs());
}



template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::setup_embedded_dofs()
{
  embedded_dh =
    std::make_unique<DoFHandler<dim, spacedim>>(embedded_triangulation);
  embedded_dh->distribute_dofs(embedded_fe);
  embedded_rhs.reinit(embedded_dh->n_dofs());
  lambda.reinit(embedded_dh->n_dofs());
}


template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::setup_coupling()
{
  // TimerOutput::Scope timer_section(monitor, "Setup coupling");

  QGauss<dim> quad(2 * space_fe.degree + 1);

  DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

  // const double epsilon =
  //   2 * std::max(GridTools::maximal_cell_diameter(space_triangulation),
  //                GridTools::maximal_cell_diameter(embedded_triangulation));
  // std::cout << "Epsilon: " << epsilon << std::endl;
  if (parameters.coupling_strategy == "inexact")
    {
      NonMatching::create_coupling_sparsity_pattern(0.,
                                                    *space_cache,
                                                    *embedded_cache,
                                                    *space_dh,
                                                    *embedded_dh,
                                                    QGauss<dim>(
                                                      2 * space_fe.degree + 1),
                                                    dsp,
                                                    space_constraints);
    }
  else if (parameters.coupling_strategy == "exact")
    {
      NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
        cells_and_quads,
        *space_dh,
        *embedded_dh,
        dsp,
        space_constraints,
        ComponentMask(),
        ComponentMask(),
        embedded_constraints);
    }
  else
    {
      Assert(false, ExcMessage("Please select a valid strategy."));
    }

  coupling_sparsity_pattern.copy_from(dsp);
  coupling_matrix.reinit(coupling_sparsity_pattern);
}



template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::assemble_system()
{
  {
    // TimerOutput::Scope timer_section(timer, "Assemble system");
    std::cout << "Assemble system" << std::endl;

    QGauss<spacedim>             quadrature_formula(2 * space_fe.degree + 1);
    FEValues<spacedim, spacedim> fe_values(space_mapping,
                                           space_fe,
                                           quadrature_formula,
                                           update_values | update_gradients |
                                             update_quadrature_points |
                                             update_JxW_values);

    const unsigned int dofs_per_cell = space_fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : space_dh->active_cell_iterators())
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
                 rhs_function.value(q_points[q_index]) *
                 fe_values.JxW(q_index)); // dx
          }

        cell->get_dof_indices(local_dof_indices);
        space_constraints.distribute_local_to_global(cell_matrix,
                                                     cell_rhs,
                                                     local_dof_indices,
                                                     stiffness_matrix,
                                                     space_rhs);
      }

    VectorTools::create_right_hand_side(embedded_mapping,
                                        *embedded_dh,
                                        QGauss<dim>(2 * embedded_fe.degree + 1),
                                        solution_function,
                                        embedded_rhs);
  }


  std::cout << "Assemble coupling term" << std::endl;
  {
    // TimerOutput::Scope timer_section(timer, "Assemble Nitsche terms");
    Functions::CutOffFunctionC1<spacedim> dirac(
      1,
      Point<spacedim>(),
      1,
      Functions::CutOffFunctionBase<spacedim>::no_component,
      true);

    if (parameters.coupling_strategy == "inexact")
      {
        NonMatching::create_coupling_mass_matrix(*space_dh,
                                                 *embedded_dh,
                                                 QGauss<dim>(
                                                   2 * space_fe.degree + 1),
                                                 coupling_matrix,
                                                 space_constraints,
                                                 ComponentMask(),
                                                 ComponentMask(),
                                                 space_mapping,
                                                 embedded_mapping,
                                                 embedded_constraints);
      }
    else if (parameters.coupling_strategy == "exact")
      {
        // Coupling mass matrix
        NonMatching::create_coupling_mass_matrix_with_exact_intersections(
          *space_dh,
          *embedded_dh,
          cells_and_quads,
          coupling_matrix,
          space_constraints,
          ComponentMask(),
          ComponentMask(),
          space_mapping,
          embedded_mapping,
          embedded_constraints);
      }
  }
}



// We solve the resulting system as done in the classical Poisson example.
template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::solve()
{
  // TimerOutput::Scope timer_section(timer, "Solve system");
  std::cout << "Solve system" << std::endl;

  SparseDirectUMFPACK K_inv_umfpack;
  K_inv_umfpack.initialize(stiffness_matrix);

  auto K  = linear_operator(stiffness_matrix);
  auto Ct = linear_operator(coupling_matrix);
  auto C  = transpose_operator(Ct);

  auto K_inv = linear_operator(K, K_inv_umfpack);

  auto             S = C * K_inv * Ct;
  ReductionControl reduction_control(2000, 1.0e-12, 1.0e-10);
  // SolverCG<Vector<double>> solver_cg(reduction_control);
  SolverGMRES<Vector<double>> solver_cg(reduction_control);
  auto S_inv = inverse_operator(S, solver_cg, PreconditionIdentity());

  lambda   = S_inv * (C * K_inv * space_rhs - embedded_rhs);
  solution = K_inv * (space_rhs - Ct * lambda);

  std::cout << "Solved in : " << reduction_control.last_step() << "iterations."
            << std::endl;

  space_constraints.distribute(solution);
}



// Finally, we output the solution living in the embedding space, just
// like all the other programs.
template <int dim, int spacedim>
void PoissonDLM<dim, spacedim>::output_results(const unsigned cycle) const
{
  // TimerOutput::Scope timer_section(timer, "Output results");
  std::cout << "Output results" << std::endl;

  data_out.clear();
  std::ofstream data_out_file("space_solution.vtu");
  data_out.attach_dof_handler(*space_dh);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  data_out.write_vtu(data_out_file);

  {
    Vector<double> difference_per_cell(space_triangulation.n_active_cells());
    VectorTools::integrate_difference(*space_dh,
                                      solution,
                                      solution_function,
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
    VectorTools::integrate_difference(*space_dh,
                                      solution,
                                      solution_function,
                                      difference_per_cell,
                                      QGauss<spacedim>(2 * space_fe.degree + 1),
                                      VectorTools::H1_norm);
    const double H1_error =
      VectorTools::compute_global_error(space_triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_norm);

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", space_triangulation.n_active_cells());
    convergence_table.add_value("dofs", space_dh->n_dofs());
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
  }

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
void PoissonDLM<dim, spacedim>::run()
{
  for (cycle = 0; cycle < parameters.n_refinement_cycles; ++cycle)
    {
      std::cout << "Cycle: " << cycle << std::endl;
      setup_grids_and_dofs();

      // Compute all the things we need to assemble the Nitsche's
      // contributions, namely the two cached triangulations and a degree to
      // integrate over the intersections.
      if (parameters.coupling_strategy == "exact")
        {
          std::cout << "Start collecting quadratures" << std::endl;
          cells_and_quads =
            NonMatching::collect_quadratures_on_overlapped_grids(
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
          std::cout << "Error in intersection: "
                    << (sum - GridTools::volume(embedded_triangulation))
                    << std::endl;
        }

      setup_coupling();
      assemble_system();
      solve();

      // error_table.error_from_exact(space_dh, solution, exact_solution);
      output_results(cycle);
      // cells_and_quads.clear();
      if (cycle < parameters.n_refinement_cycles - 1)
        space_triangulation.refine_global(1);
      cells_and_quads.clear();
      // embedded_triangulation.refine_global(1);
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



int main(int argc, char **argv)
{
  try
    {
      {
        std::cout << "Solving in 1D/2D" << std::endl;
        PoissonDLM<1, 2>::Parameters parameters;
        PoissonDLM<1, 2>             problem(parameters);
        std::string                  parameter_file;
        if (argc > 1)
          parameter_file = argv[1];
        else
          parameter_file = "parameters.prm";

        ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");
        problem.run();
      }
      {
        // std::cout << "Solving in 2D/2D" << std::endl;
        // PoissonDLM<2> problem;
        // problem.run();
      } {
        // std::cout << "Solving in 2D/3D" << std::endl;
        // PoissonDLM<2, 3> problem;
        // problem.run();
      } {
        // std::cout << "Solving in 3D/3D" << std::endl;
        // PoissonDLM<3> problem;
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
