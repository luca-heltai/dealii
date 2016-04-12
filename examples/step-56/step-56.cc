/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Authors: Andrea Bonito, Sebastian Pauletti.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>
namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
# define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
}

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <fstream>
#include <iostream>
namespace Step38
{
  using namespace dealii;
  template <int spacedim>
  class LaplaceBeltramiProblem
  {
  public:
    LaplaceBeltramiProblem (const unsigned degree = 2);
    ~LaplaceBeltramiProblem ();
    void run ();
  private:
    static const unsigned int dim = spacedim-1;
    void make_grid ();
    void setup_dofs (std::ofstream &errors);
    void assemble_system ();
    void solve ();
    void output_results (const unsigned int cycle) const;
    void compute_error (std::ofstream &errors) const;
    void refine_grid ();

    MPI_Comm                      mpi_communicator;
    parallel::distributed::Triangulation<dim,spacedim> triangulation;
    ConstraintMatrix              constraints;
    FE_Q<dim,spacedim>            fe;
    DoFHandler<dim,spacedim>      dof_handler;
    IndexSet                      locally_owned_dofs;
    IndexSet                      locally_relevant_dofs;
    MappingQ<dim, spacedim>       mapping;
    SparsityPattern               sparsity_pattern;
    LA::MPI::SparseMatrix         system_matrix;
    LA::MPI::Vector               locally_relevant_solution;
    LA::MPI::Vector               system_rhs;
    ConditionalOStream            pcout;
    TimerOutput                   computing_timer;
  };
  template <int dim>
  class Solution  : public Function<dim>
  {
  public:
    Solution () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
  };
  template <>
  double
  Solution<2>::value (const Point<2> &p,
                      const unsigned int) const
  {
    return ( -2. * p(0) * p(1) );
  }
  template <>
  Tensor<1,2>
  Solution<2>::gradient (const Point<2>   &p,
                         const unsigned int) const
  {
    Tensor<1,2> return_value;
    return_value[0] = -2. * p(1) * (1 - 2. * p(0) * p(0));
    return_value[1] = -2. * p(0) * (1 - 2. * p(1) * p(1));
    return return_value;
  }
  template <>
  double
  Solution<3>::value (const Point<3> &p,
                      const unsigned int) const
  {
    return (std::sin(numbers::PI * p(0)) *
            std::cos(numbers::PI * p(1))*exp(p(2)));
  }
  template <>
  Tensor<1,3>
  Solution<3>::gradient (const Point<3>   &p,
                         const unsigned int) const
  {
    using numbers::PI;
    Tensor<1,3> return_value;
    return_value[0] = PI *cos(PI * p(0))*cos(PI * p(1))*exp(p(2));
    return_value[1] = -PI *sin(PI * p(0))*sin(PI * p(1))*exp(p(2));
    return_value[2] = sin(PI * p(0))*cos(PI * p(1))*exp(p(2));
    return return_value;
  }
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };
  template <>
  double
  RightHandSide<2>::value (const Point<2> &p,
                           const unsigned int /*component*/) const
  {
    return ( -8. * p(0) * p(1) );
  }
  template <>
  double
  RightHandSide<3>::value (const Point<3> &p,
                           const unsigned int /*component*/) const
  {
    using numbers::PI;
    Tensor<2,3> hessian;
    hessian[0][0] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[1][1] = -PI*PI*sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[2][2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[0][1] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));
    hessian[1][0] = -PI*PI*cos(PI*p(0))*sin(PI*p(1))*exp(p(2));
    hessian[0][2] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[2][0] = PI*cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
    hessian[1][2] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
    hessian[2][1] = -PI*sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
    Tensor<1,3> gradient;
    gradient[0] = PI * cos(PI*p(0))*cos(PI*p(1))*exp(p(2));
    gradient[1] = - PI * sin(PI*p(0))*sin(PI*p(1))*exp(p(2));
    gradient[2] = sin(PI*p(0))*cos(PI*p(1))*exp(p(2));
    Point<3> normal = p;
    normal /= p.norm();
    return (- trace(hessian)
            + 2 * (gradient * normal)
            + (hessian * normal) * normal);
  }
  template <int spacedim>
  LaplaceBeltramiProblem<spacedim>::
  LaplaceBeltramiProblem (const unsigned degree)
    :
    mpi_communicator (MPI_COMM_WORLD),
    triangulation (mpi_communicator,
                   typename Triangulation<dim,spacedim>::MeshSmoothing
                   (Triangulation<dim,spacedim>::smoothing_on_refinement |
                    Triangulation<dim,spacedim>::smoothing_on_coarsening)),
    fe (degree),
    dof_handler(triangulation),
    mapping (degree),
    pcout (std::cout,
           (Utilities::MPI::this_mpi_process(mpi_communicator)
            == 0)),
    computing_timer (mpi_communicator,
                     pcout,
                     TimerOutput::summary,
                     TimerOutput::wall_times)

  {}

  template <int spacedim>
  LaplaceBeltramiProblem<spacedim>::~LaplaceBeltramiProblem ()
  {
    dof_handler.clear ();
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::make_grid ()
  {
    static SphericalManifold<dim,spacedim> surface_description;
    {
      parallel::distributed::Triangulation<spacedim> volume_mesh(mpi_communicator);
      GridGenerator::half_hyper_ball(volume_mesh);
      std::set<types::boundary_id> boundary_ids;
      boundary_ids.insert (0);
      GridGenerator::extract_boundary_mesh (volume_mesh, triangulation,
                                            boundary_ids);
    }
    triangulation.set_all_manifold_ids(0);
    triangulation.set_manifold (0, surface_description);
    triangulation.refine_global(5);
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::setup_dofs (std::ofstream &errors)
  {
    TimerOutput::Scope t(computing_timer, "setup dofs");
    dof_handler.distribute_dofs (fe);
    errors <<dof_handler.n_dofs() <<"\t";
    locally_owned_dofs = dof_handler.locally_owned_dofs ();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    locally_relevant_solution.reinit (locally_owned_dofs,
                                      locally_relevant_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);
    constraints.clear ();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<spacedim>(),
                                              constraints);
    constraints.close ();
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);

    SparsityTools::distribute_sparsity_pattern (dsp,
                                                dof_handler.n_locally_owned_dofs_per_processor(),
                                                mpi_communicator,
                                                locally_relevant_dofs);
    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);
  }

  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::assemble_system ()
  {
    TimerOutput::Scope t(computing_timer, "assembly");
    system_matrix = 0;
    system_rhs = 0;
    const QGauss<dim>  quadrature_formula(2*fe.degree);
    FEValues<dim,spacedim> fe_values (mapping, fe, quadrature_formula,
                                      update_values              |
                                      update_gradients           |
                                      update_quadrature_points   |
                                      update_JxW_values);
    const unsigned int        dofs_per_cell = fe.dofs_per_cell;
    const unsigned int        n_q_points    = quadrature_formula.size();
    FullMatrix<double>        cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>            cell_rhs (dofs_per_cell);
    std::vector<double>       rhs_values(n_q_points);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    const RightHandSide<spacedim> rhs;
    for (typename DoFHandler<dim,spacedim>::active_cell_iterator
         cell = dof_handler.begin_active(),
         endc = dof_handler.end();
         cell!=endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          cell_rhs = 0;
          fe_values.reinit (cell);
          rhs.value_list (fe_values.get_quadrature_points(), rhs_values);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                cell_matrix(i,j) += fe_values.shape_grad(i,q_point) *
                                    fe_values.shape_grad(j,q_point) *
                                    fe_values.JxW(q_point);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
              cell_rhs(i) += fe_values.shape_value(i,q_point) *
                             rhs_values[q_point]*
                             fe_values.JxW(q_point);
          cell->get_dof_indices (local_dof_indices);
          constraints.distribute_local_to_global (cell_matrix,
                                                  cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix,
                                                  system_rhs);
        }
    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::solve ()
  {
    TimerOutput::Scope t(computing_timer, "solve");
    LA::MPI::Vector
    completely_distributed_solution (locally_owned_dofs, mpi_communicator);
    SolverControl solver_control (dof_handler.n_dofs(),
                                  1e-7 * system_rhs.l2_norm());
#ifdef USE_PETSC_LA
    LA::SolverCG solver(solver_control, mpi_communicator);
#else
    LA::SolverCG solver(solver_control);
#endif
    LA::MPI::PreconditionAMG preconditioner;
    LA::MPI::PreconditionAMG::AdditionalData data;
#ifdef USE_PETSC_LA
    data.symmetric_operator = true;
#else
    / * Trilinos defaults are good */
#endif
    preconditioner.initialize(system_matrix, data);
    solver.solve (system_matrix, completely_distributed_solution, system_rhs,
                  preconditioner);
    pcout << "   Solved in " << solver_control.last_step()
          << " iterations." << std::endl;
    constraints.distribute (completely_distributed_solution);
    locally_relevant_solution = completely_distributed_solution;
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::refine_grid ()
  {
    TimerOutput::Scope t(computing_timer, "refine");
    Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
    KellyErrorEstimator<dim,spacedim>::estimate (dof_handler,
                                                 QGauss<dim-1>(2*fe.degree),
                                                 typename FunctionMap<spacedim>::type(),
                                                 locally_relevant_solution,
                                                 estimated_error_per_cell);
    parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number (triangulation,
        estimated_error_per_cell,
        0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();

  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::output_results (const unsigned int cycle) const
  {
    DataOut<dim,DoFHandler<dim,spacedim> > data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (locally_relevant_solution,"u");
    Vector<float> subdomain (triangulation.n_active_cells());
    for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector (subdomain, "subdomain");

    data_out.build_patches (mapping,
                            mapping.get_degree());
    std::string filename = ("solution-" +
                            Utilities::int_to_string (cycle, 2) +
                            "." +
                            Utilities::int_to_string
                            (triangulation.locally_owned_subdomain(), 4));
    std::ofstream output ((filename + ".vtu").c_str());
    data_out.write_vtu (output);
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0;
             i<Utilities::MPI::n_mpi_processes(mpi_communicator);
             ++i)
          filenames.push_back ("solution-" +
                               Utilities::int_to_string (cycle, 2) +
                               "." +
                               Utilities::int_to_string (i, 4) +
                               ".vtu");
        std::ofstream master_output (("solution-" +
                                      Utilities::int_to_string (cycle, 2) +
                                      ".pvtu").c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::compute_error (std::ofstream &errors) const
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping, dof_handler, locally_relevant_solution,
                                       Solution<spacedim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe.degree+1),
                                       VectorTools::L2_norm);
    errors << difference_per_cell.l2_norm()
           << "\t";
    VectorTools::integrate_difference (mapping, dof_handler, locally_relevant_solution,
                                       Solution<spacedim>(),
                                       difference_per_cell,
                                       QGauss<dim>(2*fe.degree+1),
                                       VectorTools::H1_norm);
    errors << difference_per_cell.l2_norm()
           << "\n";
  }
  template <int spacedim>
  void LaplaceBeltramiProblem<spacedim>::run ()
  {
    std::ofstream errors;
    errors.open("errors_refined.dat");
    errors << "DOFS\tL2 error\tH1 error \n";
    for (unsigned int cycle=0; cycle<8; ++cycle)
      {
        pcout << "Cycle " << cycle << ':' << std::endl;
        if (cycle == 0)
          make_grid ();
        else
          refine_grid ();
        setup_dofs(errors);
        pcout << "   Number of active cells:       "
              << triangulation.n_global_active_cells()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl
              << "   Number of processes: "
              << Utilities::MPI::n_mpi_processes(mpi_communicator)
              << std::endl;
        assemble_system ();
        solve ();

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) <= 32)
          {
            TimerOutput::Scope t(computing_timer, "output");
            output_results (cycle);
          }
        if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
          compute_error (errors);
        computing_timer.print_summary ();
        computing_timer.reset ();
        pcout << std::endl;

      }
    errors.close();
  }
}
int main (int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step38;
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      LaplaceBeltramiProblem<3> laplace_beltrami;
      laplace_beltrami.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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

