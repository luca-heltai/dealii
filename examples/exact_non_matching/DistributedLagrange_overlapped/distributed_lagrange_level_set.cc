/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 - 2021 by the deal.II authors
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
 * Authors: Luca Heltai, Giovanni Alzetta,
 * International School for Advanced Studies, Trieste, 2018
 */


#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/parameter_acceptor.h>
#include <deal.II/cgal/additional_data.h>


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

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

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <iostream>
#include <fstream>

namespace Step60
{
  using namespace dealii;


  template <int dim, int spacedim = dim>
  class DistributedLagrangeProblem
  {
  public:
    class Parameters : public ParameterAcceptor
    {
    public:
      Parameters();


      unsigned int initial_refinement = 4;

      unsigned int delta_refinement = 3;

      unsigned int initial_embedded_refinement = 8;

      std::list<types::boundary_id> dirichlet_ids{0, 1, 2, 3};

      unsigned int embedding_space_finite_element_degree = 1;

      unsigned int embedded_space_finite_element_degree = 1;

      unsigned int embedded_configuration_finite_element_degree = 1;

      unsigned int coupling_quadrature_order = 3;

      bool use_displacement = false;

      unsigned int verbosity_level = 10;

      bool initialized = false;
    };

    DistributedLagrangeProblem(const Parameters &parameters);

    void run();

  private:
    const Parameters &parameters;


    void setup_grids_and_dofs();

    void setup_embedding_dofs();

    void setup_embedded_dofs();


    void setup_coupling();

    void assemble_system();

    void solve();

    void output_results();



    std::unique_ptr<Triangulation<spacedim>> space_grid;
    std::unique_ptr<GridTools::Cache<spacedim, spacedim>>
                                             space_grid_tools_cache;
    std::unique_ptr<FiniteElement<spacedim>> space_fe;
    std::unique_ptr<DoFHandler<spacedim>>    space_dh;


    std::unique_ptr<Triangulation<dim, spacedim>> embedded_grid;
    std::unique_ptr<FiniteElement<dim, spacedim>> embedded_fe;
    std::unique_ptr<DoFHandler<dim, spacedim>>    embedded_dh;

    std::unique_ptr<FiniteElement<dim, spacedim>> embedded_configuration_fe;
    std::unique_ptr<DoFHandler<dim, spacedim>>    embedded_configuration_dh;
    Vector<double>                                embedded_configuration;


    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_configuration_function;

    std::unique_ptr<Mapping<dim, spacedim>> embedded_mapping;

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedding_rhs_function;

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedded_value_function;

    ParameterAcceptorProxy<Functions::ParsedFunction<spacedim>>
      embedding_dirichlet_boundary_function;

    ParameterAcceptorProxy<ReductionControl> schur_solver_control;

    SparsityPattern stiffness_sparsity;
    SparsityPattern coupling_sparsity;

    SparseMatrix<double> stiffness_matrix;
    SparseMatrix<double> coupling_matrix;

    AffineConstraints<double> constraints;

    Vector<double> solution;
    Vector<double> embedding_rhs;

    Vector<double> lambda;
    Vector<double> embedded_rhs;
    Vector<double> embedded_value;

    TimerOutput monitor;
  };

  template <int dim, int spacedim>
  DistributedLagrangeProblem<dim, spacedim>::Parameters::Parameters()
    : ParameterAcceptor("/Distributed Lagrange<" +
                        Utilities::int_to_string(dim) + "," +
                        Utilities::int_to_string(spacedim) + ">/")
  {
    add_parameter("Initial embedding space refinement", initial_refinement);

    add_parameter("Initial embedded space refinement",
                  initial_embedded_refinement);

    add_parameter("Local refinements steps near embedded domain",
                  delta_refinement);

    add_parameter("Dirichlet boundary ids", dirichlet_ids);

    add_parameter("Use displacement in embedded interface", use_displacement);

    add_parameter("Embedding space finite element degree",
                  embedding_space_finite_element_degree);

    add_parameter("Embedded space finite element degree",
                  embedded_space_finite_element_degree);

    add_parameter("Embedded configuration finite element degree",
                  embedded_configuration_finite_element_degree);

    add_parameter("Coupling quadrature order", coupling_quadrature_order);

    add_parameter("Verbosity level", verbosity_level);

    parse_parameters_call_back.connect([&]() -> void { initialized = true; });
  }

  template <int dim, int spacedim>
  DistributedLagrangeProblem<dim, spacedim>::DistributedLagrangeProblem(
    const Parameters &parameters)
    : parameters(parameters)
    , embedded_configuration_function("Embedded configuration", spacedim)
    , embedding_rhs_function("Embedding rhs function")
    , embedded_value_function("Embedded value")
    , embedding_dirichlet_boundary_function(
        "Embedding Dirichlet boundary conditions")
    , schur_solver_control("Schur solver control")
    , monitor(std::cout, TimerOutput::summary, TimerOutput::cpu_and_wall_times)
  {
    embedded_configuration_function.declare_parameters_call_back.connect(
      []() -> void {
        ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4, Cy=.4");


        ParameterAcceptor::prm.set("Function expression",
                                   "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
      });

    embedding_rhs_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

    embedded_value_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "1"); });

    embedding_dirichlet_boundary_function.declare_parameters_call_back.connect(
      []() -> void { ParameterAcceptor::prm.set("Function expression", "0"); });

    schur_solver_control.declare_parameters_call_back.connect([]() -> void {
      ParameterAcceptor::prm.set("Max steps", "1000");
      ParameterAcceptor::prm.set("Reduction", "1.e-12");
      ParameterAcceptor::prm.set("Tolerance", "1.e-12");
    });
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_grids_and_dofs()
  {
    TimerOutput::Scope timer_section(monitor, "Setup grids and dofs");

    space_grid = std::make_unique<Triangulation<spacedim>>();

    GridGenerator::hyper_cube(*space_grid, -1, 1, true);


    space_grid->refine_global(parameters.initial_refinement);
    space_grid_tools_cache =
      std::make_unique<GridTools::Cache<spacedim, spacedim>>(*space_grid);

    embedded_grid = std::make_unique<Triangulation<dim, spacedim>>();
    if constexpr (dim == 2 && spacedim == 3)
      {
        FunctionParser<3> cheese(
          "0.1*(x^2+y^2-4)^2 + 0.1*(z^2-1)+0.1*(y^2+z^2-4)^2 + 0.1*(x^2-1)^2 + 0.1*(x^2+z^2-4)^2 + 0.1*(y^2-1)^2 - 0.1*15");
        // FunctionParser<3>     my_function("(1-sqrt(x^2+y^2))^2+z^2-.25");
        CGALWrappers::AdditionalData<2> data;
        data.angular_bound  = 25.;
        data.radius_bound   = .1;
        data.distance_bound = .1;
        GridGenerator::implicit_function(
          *embedded_grid, cheese, data, Point<3>(1., 0., 2.), 20.0);
        std::ofstream out("test_implicit.vtu");
        GridOut       grid_out;
        grid_out.write_vtu(*embedded_grid, out);
        std::cout << "griglia embedded implicita written" << std::endl;
      }
    else
      {
        GridGenerator::hyper_cube(*embedded_grid, 0., 1.);
      }

    embedded_grid->refine_global(parameters.initial_embedded_refinement);



    embedded_configuration_fe = std::make_unique<FESystem<dim, spacedim>>(
      FE_Q<dim, spacedim>(
        parameters.embedded_configuration_finite_element_degree),
      spacedim);

    embedded_configuration_dh =
      std::make_unique<DoFHandler<dim, spacedim>>(*embedded_grid);

    embedded_configuration_dh->distribute_dofs(*embedded_configuration_fe);
    embedded_configuration.reinit(embedded_configuration_dh->n_dofs());

    VectorTools::interpolate(*embedded_configuration_dh,
                             embedded_configuration_function,
                             embedded_configuration);


    if (parameters.use_displacement == true)
      embedded_mapping =
        std::make_unique<MappingQEulerian<dim, Vector<double>, spacedim>>(
          parameters.embedded_configuration_finite_element_degree,
          *embedded_configuration_dh,
          embedded_configuration);
    else
      embedded_mapping =
        std::make_unique<MappingFEField<dim, spacedim, Vector<double>>>(
          *embedded_configuration_dh, embedded_configuration);

    setup_embedded_dofs();

    std::vector<Point<spacedim>> support_points(embedded_dh->n_dofs());
    if (parameters.delta_refinement != 0)
      DoFTools::map_dofs_to_support_points(*embedded_mapping,
                                           *embedded_dh,
                                           support_points);

    for (unsigned int i = 0; i < parameters.delta_refinement; ++i)
      {
        const auto point_locations =
          GridTools::compute_point_locations(*space_grid_tools_cache,
                                             support_points);
        const auto &cells = std::get<0>(point_locations);
        for (auto &cell : cells)
          {
            cell->set_refine_flag();
            for (const auto face_no : cell->face_indices())
              if (!cell->at_boundary(face_no))
                cell->neighbor(face_no)->set_refine_flag();
          }
        space_grid->execute_coarsening_and_refinement();
      }

    const double embedded_space_maximal_diameter =
      GridTools::maximal_cell_diameter(*embedded_grid, *embedded_mapping);
    double embedding_space_minimal_diameter =
      GridTools::minimal_cell_diameter(*space_grid);


    std::ofstream out("griglia_space.vtu");
    GridOut       grid_out;
    grid_out.write_vtu(*space_grid, out);
    std::cout << "griglia_space written" << std::endl;

    std::ofstream out_emb("griglia_emb.vtu");

    {
      DataOut<dim, spacedim> embedding_out;
      embedding_out.attach_dof_handler(*embedded_configuration_dh);
      embedding_out.build_patches(
        *embedded_mapping, parameters.embedded_space_finite_element_degree);
      embedding_out.write_vtu(out_emb);
      std::cout << "griglia_emb written" << std::endl;
    }


    deallog << "Embedding minimal diameter: "
            << embedding_space_minimal_diameter
            << ", embedded maximal diameter: "
            << embedded_space_maximal_diameter << ", ratio: "
            << embedded_space_maximal_diameter /
                 embedding_space_minimal_diameter
            << std::endl;

    AssertThrow(embedded_space_maximal_diameter <
                  embedding_space_minimal_diameter,
                ExcMessage(
                  "The embedding grid is too refined (or the embedded grid "
                  "is too coarse). Adjust the parameters so that the minimal "
                  "grid size of the embedding grid is larger "
                  "than the maximal grid size of the embedded grid."));

    setup_embedding_dofs();
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_embedding_dofs()
  {
    space_dh = std::make_unique<DoFHandler<spacedim>>(*space_grid);
    space_fe = std::make_unique<FE_Q<spacedim>>(
      parameters.embedding_space_finite_element_degree);
    space_dh->distribute_dofs(*space_fe);

    DoFTools::make_hanging_node_constraints(*space_dh, constraints);
    for (auto id : parameters.dirichlet_ids)
      {
        VectorTools::interpolate_boundary_values(
          *space_dh, id, embedding_dirichlet_boundary_function, constraints);
      }
    constraints.close();

    DynamicSparsityPattern dsp(space_dh->n_dofs(), space_dh->n_dofs());
    DoFTools::make_sparsity_pattern(*space_dh, dsp, constraints);
    stiffness_sparsity.copy_from(dsp);
    stiffness_matrix.reinit(stiffness_sparsity);
    solution.reinit(space_dh->n_dofs());
    embedding_rhs.reinit(space_dh->n_dofs());

    deallog << "Embedding dofs: " << space_dh->n_dofs() << std::endl;
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_embedded_dofs()
  {
    embedded_dh = std::make_unique<DoFHandler<dim, spacedim>>(*embedded_grid);
    embedded_fe = std::make_unique<FE_Q<dim, spacedim>>(
      parameters.embedded_space_finite_element_degree);
    embedded_dh->distribute_dofs(*embedded_fe);

    lambda.reinit(embedded_dh->n_dofs());
    embedded_rhs.reinit(embedded_dh->n_dofs());
    embedded_value.reinit(embedded_dh->n_dofs());

    deallog << "Embedded dofs: " << embedded_dh->n_dofs() << std::endl;
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::setup_coupling()
  {
    TimerOutput::Scope timer_section(monitor, "Setup coupling");

    QGauss<dim> quad(parameters.coupling_quadrature_order);

    DynamicSparsityPattern dsp(space_dh->n_dofs(), embedded_dh->n_dofs());

    NonMatching::create_coupling_sparsity_pattern(*space_grid_tools_cache,
                                                  *space_dh,
                                                  *embedded_dh,
                                                  quad,
                                                  dsp,
                                                  AffineConstraints<double>(),
                                                  ComponentMask(),
                                                  ComponentMask(),
                                                  *embedded_mapping);
    coupling_sparsity.copy_from(dsp);
    coupling_matrix.reinit(coupling_sparsity);
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::assemble_system()
  {
    {
      TimerOutput::Scope timer_section(monitor, "Assemble system");

      MatrixTools::create_laplace_matrix(
        *space_dh,
        QGauss<spacedim>(2 * space_fe->degree + 1),
        stiffness_matrix,
        embedding_rhs_function,
        embedding_rhs,
        static_cast<const Function<spacedim> *>(nullptr),
        constraints);

      VectorTools::create_right_hand_side(*embedded_mapping,
                                          *embedded_dh,
                                          QGauss<dim>(2 * embedded_fe->degree +
                                                      1),
                                          embedded_value_function,
                                          embedded_rhs);
    }
    {
      TimerOutput::Scope timer_section(monitor, "Assemble coupling system");

      QGauss<dim> quad(parameters.coupling_quadrature_order);
      NonMatching::create_coupling_mass_matrix(*space_grid_tools_cache,
                                               *space_dh,
                                               *embedded_dh,
                                               quad,
                                               coupling_matrix,
                                               AffineConstraints<double>(),
                                               ComponentMask(),
                                               ComponentMask(),
                                               *embedded_mapping);

      VectorTools::interpolate(*embedded_mapping,
                               *embedded_dh,
                               embedded_value_function,
                               embedded_value);
    }
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::  solve()
  {
    TimerOutput::Scope timer_section(monitor, "Solve system");

    SparseDirectUMFPACK K_inv_umfpack;
    K_inv_umfpack.initialize(stiffness_matrix);

    auto K  = linear_operator(stiffness_matrix);
    auto Ct = linear_operator(coupling_matrix);
    auto C  = transpose_operator(Ct);

    auto K_inv = linear_operator(K, K_inv_umfpack);

    auto                     S = C * K_inv * Ct;
    SolverCG<Vector<double>> solver_cg(schur_solver_control);
    auto S_inv = inverse_operator(S, solver_cg, PreconditionIdentity());

    lambda = S_inv * (C * K_inv * embedding_rhs - embedded_rhs);

    solution = K_inv * (embedding_rhs - Ct * lambda);

    constraints.distribute(solution);
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::output_results()
  {
    TimerOutput::Scope timer_section(monitor, "Output results");

    std::ofstream embedding_out_file("embedding.vtu");

    DataOut<spacedim> embedding_out;
    embedding_out.attach_dof_handler(*space_dh);
    embedding_out.add_data_vector(solution, "solution");
    embedding_out.build_patches(
      parameters.embedding_space_finite_element_degree);
    embedding_out.write_vtu(embedding_out_file);


    DataOut<dim, spacedim> embedded_out;

    std::ofstream embedded_out_file("embedded.vtu");

    embedded_out.attach_dof_handler(*embedded_dh);
    embedded_out.add_data_vector(lambda, "lambda");
    embedded_out.add_data_vector(embedded_value, "g");
    embedded_out.build_patches(*embedded_mapping,
                               parameters.embedded_space_finite_element_degree);
    embedded_out.write_vtu(embedded_out_file);
  }

  template <int dim, int spacedim>
  void DistributedLagrangeProblem<dim, spacedim>::run()
  {
    AssertThrow(parameters.initialized, ExcNotInitialized());
    deallog.depth_console(parameters.verbosity_level);

    setup_grids_and_dofs();
    setup_coupling();
    assemble_system();
    solve();
    output_results();
  }
} // namespace Step60



int main(int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace Step60;

      const unsigned int dim = 1, spacedim = 2;
      // const unsigned int dim = 2, spacedim = 3;


      DistributedLagrangeProblem<dim, spacedim>::Parameters parameters;
      DistributedLagrangeProblem<dim, spacedim>             problem(parameters);

      std::string parameter_file;
      if (argc > 1)
        parameter_file = argv[1];
      else
        parameter_file = "parameters.prm";

      ParameterAcceptor::initialize(parameter_file, "used_parameters.prm");

      // Triangulation<dim, 3> tria;
      // FunctionParser<3>     cheese(
      //   "(x^2+y^2-4)^2 + (z^2-1)+(y^2+z^2-4)^2 + (x^2-1)^2 + (x^2+z^2-4)^2 +
      //   (y^2-1)^2 - 15");
      // // FunctionParser<3>     my_function("(1-sqrt(x^2+y^2))^2+z^2-.25");
      // CGALWrappers::AdditionalData<2> data;
      // data.angular_bound  = 25.;
      // data.radius_bound   = .1;
      // data.distance_bound = .1;
      // GridGenerator::implicit_function(
      //   tria, cheese, data, Point<3>(1., 0., 2.), 20.0);
      // std::ofstream out("test_implicit.vtu");
      // GridOut       grid_out;
      // grid_out.write_vtu(tria, out);
      // std::cout << "test_implicit written" << std::endl;

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