/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by the deal.II authors
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

 *
 * Authors: Luca Heltai, Bruno Blais, 2019
 */

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>


#include <deal.II/lac/generic_linear_algebra.h>

/* #define FORCE_USE_OF_TRILINOS */

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/parameter_acceptor.h>

#include <cmath>
#include <fstream>
#include <iostream>

namespace Step70
{
  using namespace dealii;

  class StokesImmersedProblemParameters : public ParameterAcceptor
  {
  public:
    StokesImmersedProblemParameters()
      : ParameterAcceptor("Stokes Immersed Problem")
    {
      add_parameter("Velocity degree",
                    velocity_degree,
                    "",
                    this->prm,
                    Patterns::Integer(1));
      add_parameter("Viscosity", viscosity);
    }

    unsigned int velocity_degree = 2;
    double       viscosity       = 1.0;
  };

  template <int dim, int spacedim = dim>
  class StokesImmersedProblem
  {
  public:
    StokesImmersedProblem(const StokesImmersedProblemParameters &par);

    void run();

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    const StokesImmersedProblemParameters &par;

    MPI_Comm mpi_communicator;

    std::unique_ptr<FESystem<spacedim>>      fe1;
    std::unique_ptr<FESystem<dim, spacedim>> fe2;

    parallel::distributed::Triangulation<spacedim>      tria1;
    parallel::distributed::Triangulation<dim, spacedim> tria2;

    DoFHandler<spacedim>      dh1;
    DoFHandler<dim, spacedim> dh2;

    std::unique_ptr<MappingFEField<dim, spacedim>> mapping2;

    std::vector<IndexSet> owned1;
    std::vector<IndexSet> owned2;

    std::vector<IndexSet> relevant1;
    std::vector<IndexSet> relevant2;


    AffineConstraints<double> constraints;

    LA::MPI::BlockSparseMatrix system_matrix;
    LA::MPI::BlockSparseMatrix coupling_matrix;

    LA::MPI::BlockSparseMatrix preconditioner_matrix;
    LA::MPI::BlockVector       locally_relevant_solution;
    LA::MPI::BlockVector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;
  };



  template <int dim, int spacedim>
  StokesImmersedProblem<dim, spacedim>::StokesImmersedProblem(
    const StokesImmersedProblemParameters &par)
    : par(par)
    , mpi_communicator(MPI_COMM_WORLD)
    , tria1(mpi_communicator,
            typename Triangulation<spacedim>::MeshSmoothing(
              Triangulation<spacedim>::smoothing_on_refinement |
              Triangulation<spacedim>::smoothing_on_coarsening))
    , tria2(mpi_communicator,
            typename Triangulation<dim, spacedim>::MeshSmoothing(
              Triangulation<dim, spacedim>::smoothing_on_refinement |
              Triangulation<dim, spacedim>::smoothing_on_coarsening))
    , dh1(tria1)
    , dh2(tria2)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
  {}


  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::make_grid()
  {
    GridGenerator::hyper_cube(tria1, -0.5, 1.5);
    tria1.refine_global(3);
  }

  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    fe1 =
      std::make_unique<FESystem<spacedim>>(FE_Q<spacedim>(par.velocity_degree),
                                           spacedim,
                                           FE_Q<spacedim>(par.velocity_degree -
                                                          1),
                                           1);

    fe2 = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(par.velocity_degree), spacedim);

    dh1.distribute_dofs(*fe1);

    std::vector<unsigned int> stokes_sub_blocks(dim + 1, 0);
    stokes_sub_blocks[dim] = 1;
    DoFRenumbering::component_wise(dh1, stokes_sub_blocks);

    std::vector<types::global_dof_index> dofs_per_block(2);
    DoFTools::count_dofs_per_block(dh1, dofs_per_block, stokes_sub_blocks);

    const unsigned int n_u = dofs_per_block[0], n_p = dofs_per_block[1];

    pcout << "   Number of degrees of freedom: " << dh1.n_dofs() << " (" << n_u
          << '+' << n_p << ')' << std::endl;

    owned1.resize(2);
    owned1[0] = dh1.locally_owned_dofs().get_view(0, n_u);
    owned1[1] = dh1.locally_owned_dofs().get_view(n_u, n_u + n_p);

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dh1, locally_relevant_dofs);
    relevant1.resize(2);
    relevant1[0] = locally_relevant_dofs.get_view(0, n_u);
    relevant1[1] = locally_relevant_dofs.get_view(n_u, n_u + n_p);

    {
      constraints.reinit(locally_relevant_dofs);

      FEValuesExtractors::Vector velocities(0);
      DoFTools::make_hanging_node_constraints(dh1, constraints);
      VectorTools::interpolate_boundary_values(dh1,
                                               0,
                                               ConstantFunction<spacedim>(0),
                                               constraints,
                                               fe1->component_mask(velocities));
      constraints.close();
    }

    {
      system_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::none;
          else if (c == dim || d == dim || c == d)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dh1, coupling, dsp, constraints, false);

      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dh1.compute_locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);

      system_matrix.reinit(owned1, dsp, mpi_communicator);
    }

    {
      preconditioner_matrix.clear();

      Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
      for (unsigned int c = 0; c < dim + 1; ++c)
        for (unsigned int d = 0; d < dim + 1; ++d)
          if (c == dim && d == dim)
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;

      BlockDynamicSparsityPattern dsp(dofs_per_block, dofs_per_block);

      DoFTools::make_sparsity_pattern(dh1, coupling, dsp, constraints, false);
      SparsityTools::distribute_sparsity_pattern(
        dsp,
        dh1.compute_locally_owned_dofs_per_processor(),
        mpi_communicator,
        locally_relevant_dofs);
      preconditioner_matrix.reinit(owned1, dsp, mpi_communicator);
    }

    locally_relevant_solution.reinit(owned1, relevant1, mpi_communicator);
    system_rhs.reinit(owned1, mpi_communicator);
  }



  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::assemble_system()
  {
    TimerOutput::Scope t(computing_timer, "assembly");

    system_matrix         = 0;
    preconditioner_matrix = 0;
    system_rhs            = 0;

    const QGauss<spacedim> quadrature_formula(par.velocity_degree + 1);

    FEValues<spacedim> fe_values(*fe1,
                                 quadrature_formula,
                                 update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);

    const unsigned int dofs_per_cell = fe1->dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> cell_matrix2(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);

    const ConstantFunction<spacedim> right_hand_side(spacedim + 1);
    std::vector<Vector<double>>      rhs_values(n_q_points,
                                                Vector<double>(spacedim + 1));

    std::vector<Tensor<2, spacedim>> grad_phi_u(dofs_per_cell);
    std::vector<double>              div_phi_u(dofs_per_cell);
    std::vector<double>              phi_p(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    const FEValuesExtractors::Vector     velocities(0);
    const FEValuesExtractors::Scalar     pressure(spacedim);

    for (const auto &cell : dh1.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell_matrix  = 0;
          cell_matrix2 = 0;
          cell_rhs     = 0;

          fe_values.reinit(cell);
          right_hand_side.vector_value_list(fe_values.get_quadrature_points(),
                                            rhs_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  grad_phi_u[k] = fe_values[velocities].gradient(k, q);
                  div_phi_u[k]  = fe_values[velocities].divergence(k, q);
                  phi_p[k]      = fe_values[pressure].value(k, q);
                }

              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                      cell_matrix(i, j) +=
                        (par.viscosity *
                           scalar_product(grad_phi_u[i], grad_phi_u[j]) -
                         div_phi_u[i] * phi_p[j] - phi_p[i] * div_phi_u[j]) *
                        fe_values.JxW(q);

                      cell_matrix2(i, j) += 1.0 / par.viscosity * phi_p[i] *
                                            phi_p[j] * fe_values.JxW(q);
                    }

                  const unsigned int component_i =
                    fe1->system_to_component_index(i).first;
                  cell_rhs(i) += fe_values.shape_value(i, q) *
                                 rhs_values[q](component_i) * fe_values.JxW(q);
                }
            }


          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 cell_rhs,
                                                 local_dof_indices,
                                                 system_matrix,
                                                 system_rhs);

          constraints.distribute_local_to_global(cell_matrix2,
                                                 local_dof_indices,
                                                 preconditioner_matrix);
        }

    system_matrix.compress(VectorOperation::add);
    preconditioner_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);
  }



  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::solve()
  {
    TimerOutput::Scope t(computing_timer, "solve");

    LA::MPI::PreconditionAMG prec_A;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      prec_A.initialize(system_matrix.block(0, 0), data);
    }

    LA::MPI::PreconditionAMG prec_S;
    {
      LA::MPI::PreconditionAMG::AdditionalData data;

#ifdef USE_PETSC_LA
      data.symmetric_operator = true;
#endif
      prec_S.initialize(preconditioner_matrix.block(1, 1), data);
    }

    //    using mp_inverse_t =
    //    LinearSolvers::InverseMatrix<LA::MPI::SparseMatrix,
    //                                                      LA::MPI::PreconditionAMG>;
    //    const mp_inverse_t mp_inverse(preconditioner_matrix.block(1, 1),
    //    prec_S);

    //    const
    //    LinearSolvers::BlockDiagonalPreconditioner<LA::MPI::PreconditionAMG,
    //                                                     mp_inverse_t>
    //      preconditioner(prec_A, mp_inverse);

    SolverControl solver_control(system_matrix.m(),
                                 1e-10 * system_rhs.l2_norm());

    SolverMinRes<LA::MPI::BlockVector> solver(solver_control);

    LA::MPI::BlockVector distributed_solution(owned1, mpi_communicator);

    constraints.set_zero(distributed_solution);

    solver.solve(system_matrix,
                 distributed_solution,
                 system_rhs,
                 PreconditionIdentity());
    // preconditioner);

    pcout << "   Solved in " << solver_control.last_step() << " iterations."
          << std::endl;

    constraints.distribute(distributed_solution);

    locally_relevant_solution = distributed_solution;
    const double mean_pressure =
      VectorTools::compute_mean_value(dh1,
                                      QGauss<spacedim>(par.velocity_degree + 2),
                                      locally_relevant_solution,
                                      spacedim);
    distributed_solution.block(1).add(-mean_pressure);
    locally_relevant_solution.block(1) = distributed_solution.block(1);
  }



  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::refine_grid()
  {
    TimerOutput::Scope t(computing_timer, "refine");

    tria1.refine_global();
  }



  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::output_results(
    const unsigned int cycle) const
  {
    {
      const ComponentSelectFunction<spacedim> pressure_mask(spacedim,
                                                            spacedim + 1);
      const ComponentSelectFunction<spacedim> velocity_mask(
        std::make_pair(0, spacedim), spacedim + 1);

      Vector<double>   cellwise_errors(tria1.n_active_cells());
      QGauss<spacedim> quadrature(par.velocity_degree + 2);

      VectorTools::integrate_difference(
        dh1,
        locally_relevant_solution,
        ConstantFunction<spacedim>(1, spacedim + 1),
        cellwise_errors,
        quadrature,
        VectorTools::L2_norm,
        &velocity_mask);

      const double error_u_l2 =
        VectorTools::compute_global_error(tria1,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(
        dh1,
        locally_relevant_solution,
        ConstantFunction<spacedim>(1.0, spacedim + 1),
        cellwise_errors,
        quadrature,
        VectorTools::L2_norm,
        &pressure_mask);

      const double error_p_l2 =
        VectorTools::compute_global_error(tria1,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      pcout << "error: u_0: " << error_u_l2 << " p_0: " << error_p_l2
            << std::endl;
    }


    std::vector<std::string> solution_names(spacedim, "velocity");
    solution_names.emplace_back("pressure");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    DataOut<spacedim> data_out;
    data_out.attach_dof_handler(dh1);
    data_out.add_data_vector(locally_relevant_solution,
                             solution_names,
                             DataOut<spacedim>::type_dof_data,
                             data_component_interpretation);

    LA::MPI::BlockVector interpolated;
    interpolated.reinit(owned1, MPI_COMM_WORLD);
    VectorTools::interpolate(dh1,
                             ConstantFunction<spacedim>(1.0, spacedim + 1),
                             interpolated);

    LA::MPI::BlockVector interpolated_relevant(owned1,
                                               relevant1,
                                               MPI_COMM_WORLD);
    interpolated_relevant = interpolated;
    {
      std::vector<std::string> solution_names(dim, "ref_u");
      solution_names.emplace_back("ref_p");
      data_out.add_data_vector(interpolated_relevant,
                               solution_names,
                               DataOut<spacedim>::type_dof_data,
                               data_component_interpretation);
    }


    Vector<float> subdomain(tria1.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = tria1.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches();

    const std::string filename =
      ("solution-" + Utilities::int_to_string(cycle, 2) + "." +
       Utilities::int_to_string(tria1.locally_owned_subdomain(), 4));
    std::ofstream output((filename + ".vtu"));
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back("solution-" + Utilities::int_to_string(cycle, 2) +
                              "." + Utilities::int_to_string(i, 4) + ".vtu");

        std::ofstream master_output(
          "solution-" + Utilities::int_to_string(cycle, 2) + ".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }
  }



  template <int dim, int spacedim>
  void StokesImmersedProblem<dim, spacedim>::run()
  {
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif
    const unsigned int n_cycles = 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          make_grid();
        else
          refine_grid();

        setup_system();

        assemble_system();
        solve();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results(cycle);
          }

        computing_timer.print_summary();
        computing_timer.reset();

        pcout << std::endl;
      }
  }
} // namespace Step70



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step70;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      StokesImmersedProblemParameters par;
      par.initialize("parameters.prm", "used_parameters.prm");

      StokesImmersedProblem<2> problem(par);
      problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
