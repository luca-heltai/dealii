/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 by the deal.II authors
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
 * Author: Luca Heltai, International School for Advanced Studies, Trieste, 2018
 */

// @sect3{Include files}

#include <iostream>
#include <deal.II/base/logstream.h>

#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/base/timer.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/parsed_function.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/non_matching/coupling.h>

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

namespace Step60
{
  using namespace dealii;

  template <int dim, int spacedim=dim>
  class DistributedLagrangeProblem
  {
  public:

    class DistributedLagrangeProblemParameters : public ParameterAcceptor
    {
    public:
      DistributedLagrangeProblemParameters();

      unsigned int initial_refinement = 4;
      unsigned int delta_refinement = 3;
      unsigned int initial_embedded_refinement = 4;
      std::list<types::boundary_id> homogeneous_dirichlet_ids {0};
      unsigned int embedding_space_finite_element_degree = 1;
      unsigned int embedded_space_finite_element_degree = 1;
      unsigned int embedded_configuration_finite_element_degree = 1;
      unsigned int coupling_quadrature_order = 3;
      bool use_displacement = false;
    };

    DistributedLagrangeProblem(const DistributedLagrangeProblemParameters &parameters);

    void run();

  private:
    const DistributedLagrangeProblemParameters &parameters;

    void setup_grids_and_dofs();

    void setup_embedding_dofs();

    void setup_embedded_dofs();

    void setup_coupling();

    void assemble_system();

    void solve();

    void output_results();


    // Embedding space geometry
    std::unique_ptr<Triangulation<spacedim> > space_grid;

    // A Cache for the computation of some space grid related stuff.
    std::unique_ptr<GridTools::Cache<spacedim, spacedim> > space_grid_tools_cache;

    // Embedding finite element space
    std::unique_ptr<FiniteElement<spacedim> > space_fe;

    // Embedding dof handler
    std::unique_ptr<DoFHandler<spacedim> > space_dh;

    // Lagrange multiplier geometry
    std::unique_ptr<Triangulation<dim, spacedim> > embedded_grid;


    // Embedded finite element space
    std::unique_ptr<FiniteElement<dim, spacedim> > embedded_fe;

    // Embedded dof handler
    std::unique_ptr<DoFHandler<dim, spacedim> > embedded_dh;

    // Embedded finite element space
    std::unique_ptr<FiniteElement<dim, spacedim> > embedded_configuration_fe;

    // Embedded dof handler
    std::unique_ptr<DoFHandler<dim, spacedim> > embedded_configuration_dh;

    // The configuration vector
    Vector<double> embedded_configuration;

    std::unique_ptr<Mapping<dim,spacedim> > embedded_mapping;

    // Embedded configuration function
    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim> >
    embedded_configuration_function;

    // Embedded value
    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim> >
    embedded_value_function;

    ParameterAcceptorProxy<ReductionControl> schur_solver_control;

    std::vector<unsigned int> dofs_per_block;

    SparsityPattern           stiffness_sparsity;
    SparsityPattern           coupling_sparsity;
    SparsityPattern           embedded_mass_sparsity;

    SparseMatrix<double>      stiffness_matrix;
    SparseMatrix<double>      coupling_matrix;
    SparseMatrix<double>      embedded_mass_matrix;

    ConstraintMatrix          constraints;

    Vector<double>            solution;
    Vector<double>            rhs;

    Vector<double>            lambda;
    Vector<double>            lambda_rhs;
    Vector<double>            embedded_value;

    // TimerOuput
    TimerOutput monitor;
  };

  template<int dim, int spacedim>
  DistributedLagrangeProblem<dim,spacedim>::DistributedLagrangeProblemParameters::
  DistributedLagrangeProblemParameters() :
    ParameterAcceptor("/Distributed Lagrange<" + Utilities::int_to_string(dim)
                      + "," + Utilities::int_to_string(spacedim) +">")
  {
    add_parameter("Initial embedding space refinement",
                  initial_refinement);

    add_parameter("Initial embedded space refinement",
                  initial_embedded_refinement);

    add_parameter("Local refinements steps near embedded domain",
                  delta_refinement);

    add_parameter("Homogeneous Dirichlet boundary ids",
                  homogeneous_dirichlet_ids);

    add_parameter("Use displacement in embedded interface",
                  use_displacement);

    add_parameter("Embedding space finite element degree",
                  embedding_space_finite_element_degree);

    add_parameter("Embedded space finite element degree",
                  embedded_space_finite_element_degree);

    add_parameter("Embedded configuration finite element degree",
                  embedded_configuration_finite_element_degree);

    add_parameter("Coupling quadrature order",
                  coupling_quadrature_order);
  }

  template<int dim, int spacedim>
  DistributedLagrangeProblem<dim,spacedim>::DistributedLagrangeProblem(
    const DistributedLagrangeProblemParameters &parameters) :
    parameters(parameters),
    embedded_configuration_function("Embedded configuration", spacedim),
    embedded_value_function("Embedded value"),
    schur_solver_control("Schur solver control"),
    dofs_per_block(2),
    monitor(std::cout,
            TimerOutput::summary,
            TimerOutput::cpu_and_wall_times)
  {
    auto myf = [&] () -> void
    {
      // embedded_configuration_function.enter_my_subsection(ParameterAcceptor::prm);
      ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4, Cy=.4");
      ParameterAcceptor::prm.set("Function expression", "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      // embedded_configuration_function.leave_my_subsection(ParameterAcceptor::prm);
    };
    embedded_configuration_function.declare_parameters_call_back.connect(myf);
  }

  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::run()
  {
    setup_grids_and_dofs();
    setup_coupling();
    assemble_system();
    solve();
    output_results();
  }

  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::setup_grids_and_dofs()
  {
    TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

    space_grid = std_cxx14::make_unique<Triangulation<spacedim> >();
    GridGenerator::hyper_cube(*space_grid);
    space_grid->refine_global(parameters.initial_refinement);
    space_grid_tools_cache = std_cxx14::make_unique<GridTools::Cache<spacedim, spacedim> >(*space_grid);

    embedded_grid = std_cxx14::make_unique<Triangulation<dim,spacedim> >();
    GridGenerator::hyper_cube(*embedded_grid);
    embedded_grid->refine_global(parameters.initial_embedded_refinement);

    // At this point we need to configure the deformation of the embedded grid
    embedded_configuration_fe =
      std_cxx14::make_unique<FESystem<dim,spacedim> >
      (FE_Q<dim,spacedim>(parameters.embedded_configuration_finite_element_degree), spacedim);

    embedded_configuration_dh =
      std_cxx14::make_unique<DoFHandler<dim, spacedim> >(*embedded_grid);

    embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);
    embedded_configuration.reinit(embedded_configuration_dh->n_dofs());
    VectorTools::interpolate(*embedded_configuration_dh,
                             embedded_configuration_function,
                             embedded_configuration);

    if (parameters.use_displacement == true)
      embedded_mapping =
        std_cxx14::make_unique<MappingQEulerian<dim, Vector<double>, spacedim> >
        (parameters.embedded_configuration_finite_element_degree,
         *embedded_configuration_dh, embedded_configuration);
    else
      embedded_mapping =
        std_cxx14::make_unique<MappingFEField<dim, spacedim, Vector<double>, DoFHandler<dim,spacedim> > >
        (*embedded_configuration_dh,
         embedded_configuration);

    double embedded_space_maximal_diameter = GridTools::maximal_cell_diameter(*embedded_grid, *embedded_mapping);

    setup_embedded_dofs();
    for (unsigned int i=0; i<parameters.delta_refinement; ++i)
      {
        std::vector<Point<spacedim> > support_points(embedded_dh->n_dofs());
        DoFTools::map_dofs_to_support_points(*embedded_mapping,
                                             *embedded_dh,
                                             support_points);
        const auto point_locations = GridTools::compute_point_locations(*space_grid_tools_cache,
                                     support_points);
        const auto &cells = std::get<0>(point_locations);
        for (auto cell : cells)
          cell->set_refine_flag();
        space_grid->execute_coarsening_and_refinement();
        double embedding_space_minimal_diameter = GridTools::minimal_cell_diameter(*space_grid);
        AssertThrow(embedded_space_maximal_diameter < embedding_space_minimal_diameter,
                    ExcMessage("The embedding grid is too refined (or the embedded grid is too coarse). Adjust the "
                               "parameters so that the minimal grid size of the embedding grid is larger "
                               "than the maximal grid size of the embedded grid."))
      }
    setup_embedding_dofs();
  }

  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::setup_embedding_dofs()
  {
    TimerOutput::Scope timer_section(monitor, "Setup embedding dofs");

    space_dh = std_cxx14::make_unique<DoFHandler<spacedim> >(*space_grid);
    space_fe = std_cxx14::make_unique<FE_Q<spacedim> >(parameters.embedding_space_finite_element_degree);
    space_dh->distribute_dofs(*space_fe);

    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    for (auto id:parameters.homogeneous_dirichlet_ids)
      {
        VectorTools::interpolate_boundary_values(*space_dh, id,
                                                 Functions::ZeroFunction<spacedim>(),
                                                 constraints);
      }
    constraints.close();

    DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
    DoFTools::make_sparsity_pattern(*space_dh, dsp, constraints);

    stiffness_sparsity.copy_from(dsp);
    stiffness_matrix.reinit(stiffness_sparsity);
    solution.reinit(space_dh->n_dofs());
    rhs.reinit(space_dh->n_dofs());
  }

  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::setup_embedded_dofs()
  {
    TimerOutput::Scope timer_section(monitor, "Setup embedded dofs");

    embedded_dh = std_cxx14::make_unique<DoFHandler<dim,spacedim> >(*embedded_grid);
    embedded_fe = std_cxx14::make_unique<FE_Q<dim,spacedim> >(parameters.embedded_space_finite_element_degree);
    embedded_dh->distribute_dofs(*embedded_fe);

    DynamicSparsityPattern dsp(embedded_dh->n_dofs(), embedded_dh->n_dofs());
    DoFTools::make_sparsity_pattern(*embedded_dh, dsp);
    embedded_mass_sparsity.copy_from(dsp);
    embedded_mass_matrix.reinit(embedded_mass_sparsity);

    lambda.reinit(embedded_dh->n_dofs());
    lambda_rhs.reinit(embedded_dh->n_dofs());
    embedded_value.reinit(embedded_dh->n_dofs());
  }


  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::setup_coupling()
  {
    TimerOutput::Scope timer_section(monitor, "Setup coupling");

    QGauss<dim> quad(parameters.coupling_quadrature_order);

    DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

    NonMatching::create_coupling_sparsity_pattern(*space_dh,
                                                  *embedded_dh,
                                                  quad,
                                                  dsp, ComponentMask(), ComponentMask(),
                                                  StaticMappingQ1<spacedim>::mapping,
                                                  *embedded_mapping);
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);
  }


  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::assemble_system()
  {
    TimerOutput::Scope timer_section(monitor, "Assemble system");

    // Embedding stiffness matrix
    MatrixTools::create_laplace_matrix(*space_dh, QGauss<spacedim>(2*space_fe->degree+1),
                                       stiffness_matrix, (const Function<spacedim> *) nullptr, constraints);

    // Embedded mass matrix
    MatrixTools::create_mass_matrix(*embedded_dh, QGauss<dim>(2*embedded_fe->degree+1),
                                    embedded_mass_matrix);

    // Coupling matrix
    QGauss<dim> quad(parameters.coupling_quadrature_order);
    NonMatching::create_coupling_mass_matrix(*space_dh,
                                             *embedded_dh,
                                             quad,
                                             coupling_matrix, ConstraintMatrix(),
                                             ComponentMask(), ComponentMask(),
                                             StaticMappingQ1<spacedim>::mapping,
                                             *embedded_mapping);

    VectorTools::interpolate(*embedded_dh, embedded_value_function, embedded_value);
  }


  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::solve()
  {
    TimerOutput::Scope timer_section(monitor, "Solve system");

    // Start by creating the inverse stiffness matrix, and the inverse mass matrix
    SparseDirectUMFPACK A_inv_umfpack;
    A_inv_umfpack.initialize(stiffness_matrix);

    auto A = linear_operator(stiffness_matrix);
    auto M = linear_operator(embedded_mass_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C = transpose_operator(Ct);

    auto A_inv = linear_operator(A, A_inv_umfpack);

    auto S = C*A_inv*Ct;

    SolverCG<Vector<double> > solver_cg(schur_solver_control);
    auto S_inv = inverse_operator(S, solver_cg, identity_operator(M));

    lambda = S_inv * M * embedded_value;

    solution = A_inv * Ct * lambda;

    constraints.distribute(solution);
  }


  template<int dim, int spacedim>
  void DistributedLagrangeProblem<dim,spacedim>::output_results()
  {
    TimerOutput::Scope timer_section(monitor, "Output results");

    DataOut<spacedim> embedding_out;

    std::ofstream embedding_out_file("embedding.vtu");

    embedding_out.attach_dof_handler(*space_dh);
    embedding_out.add_data_vector(solution, "solution");
    embedding_out.build_patches(parameters.embedding_space_finite_element_degree);
    embedding_out.write_vtu(embedding_out_file);

    DataOut<dim, DoFHandler<dim,spacedim> > embedded_out;

    std::ofstream embedded_out_file("embedded.vtu");

    embedded_out.attach_dof_handler(*embedded_dh);
    embedded_out.add_data_vector(lambda, "lambda");
    embedded_out.add_data_vector(embedded_value, "lambda_values");
    embedded_out.build_patches(*embedded_mapping,
                               parameters.embedded_space_finite_element_degree);
    embedded_out.write_vtu(embedded_out_file);
  }
}



int main()
{
  try
    {
      using namespace dealii;
      using namespace Step60;

      const unsigned int dim=1, spacedim=2;

      DistributedLagrangeProblem<dim, spacedim>::DistributedLagrangeProblemParameters parameters;
      DistributedLagrangeProblem<dim, spacedim> problem(parameters);
      ParameterAcceptor::initialize("parameters.prm", "used_parameters.prm");
      problem.run();
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
