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
// Most of these have been introduced elsewhere, we'll comment only on the new ones.

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>

#include <deal.II/base/parameter_acceptor.h>

// The parameter acceptor class is the first novelty of this tutorial program.
//
// This class is used to define a public interface for classes that want to use
// a single global ParameterHandler to handle parameters. The class provides a
// static ParameterHandler member, namely ParameterAcceptor::prm, and
// implements the "Command design patter" (see, for example, E. Gamma, R. Helm,
// R. Johnson, J. Vlissides, Design Patterns: Elements of Reusable
// Object-Oriented Software, Addison-Wesley Professional, 1994.
// https://goo.gl/FNYByc).
//
// ParameterAcceptor provides a global subscription mechanism. Whenever an
// object of a class derived from ParameterAcceptor is constructed, a pointer
// to that object-of-derived-type is registered, together with a section entry
// in the parameter file. Such registry is traversed upon invocation of the
// single function ParameterAcceptor::initialize(file.prm) which in turn makes
// sure that all classes stored in the global register declare the parameters
// they will be using, and after having declared them, it reads the content of
// `file.prm` to parse the actual parameters.
//
// If you call the method ParameterHandler::add_parameter for each of the
// parameters you want to use in your code, there is nothing else you need to
// do. If you are using an already existing class that provides the two
// functions `declare_parameters` and `parse_parameters`, you can still use
// ParameterAcceptor, by encapsulating the existing class into a
// ParameterAcceptorProxy class.
//
// In this example, we'll use both strategies, using ParameterAcceptorProxy for
// deal.II classes, and deriving our own parameter classes directly from
// ParameterAcceptor.

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_tools_cache.h>
// The other new include file is the one that contains the GridTools::Cache class.
// This class is used when you need to compute data structures that refer to a
// Triangulation that are usually not stored in the Triangulation itself, like,
// for example, a map from the vertices of a Triangulation to all cells that share
// that vertex. Since these data structures are usually not needed in a finite element
// code, deal.II provides function in GridTools to compute them, but it does not store
// this information in the Triangulation itself.
//
// Some methods, for example GridTools::find_active_cell_around_point, make
// heavy usage of these non-standard data structures. If you need to call these
// methods more than once, it makes sense to store those data structures
// somewhere. GridTools::Cache does exactly this, giving you access to previously
// computed objects, or computing them on the fly (and then storing them inside the
// class for later use), and making sure that whenever the Triangulation is updated,
// also the relevant data strucutres are recomputed.

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>
// In this example, we will be using a reference domain to describe an embedded
// Triangulation, deformed through a finite element vector field.
//
// The two include files above contain the definition of two classes that can
// be used in these cases. MappingQEulerian allows one to describe a domain
// through a *deformation* field, based on a FESystem[FE_Q(p)^spacedim] finite
// element space. The second is a little more generic, and allows you to use
// arbitrary vector FiniteElement spaces, as long as they provide a
// *continuous* description of your domain. In this case, the description is
// done through the actual *configuration* field, rather than a *deformation*
// field.
//
// Which one is used depends on how the user wants to specify the reference
// domain, and/or the actual configuration. We'll provide both options, and
// experiment a little in the results section of this tutorial program.

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/parsed_function.h>
// The parsed function class is another new entry. It allows one to create a
// Function object, starting from a string in a parameter file which is parsed
// into an actual Function object, that you can use anywhere deal.II accepts a
// Function (for example, for interpolation, boundary conditions, etc.).

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <deal.II/non_matching/coupling.h>
// This is the last new entry for this tutorial program. The namespace
// NonMatching contains a few methods that are useful when performing
// computations on non-matching grids, or on curves that are not aligned with
// the underlying mesh.
//
// We'll discuss its use in details later on in the `setup_coupling` method.

#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>

#include <iostream>
#include <fstream>

namespace Step60
{
  using namespace dealii;

  // In the DistributedLagrangeProblem, we need two parameters describing the
  // dimensions of the domain $\Gamma$ (`dim`) and of the domain $\Omega$
  // (`spacedim`).
  //
  // These will be used to initialize a Triangulation<dim,spacedim> (for
  // $\Gamma$) and a Triangulation<spacedim,spacedim> (for $\Omega$).
  //
  // A novelty w.r.t. to other tutorial programs is the heavy use of
  // std::unique_ptr. These behave like classical pointers, with the advantage
  // of doing automatic house-keeping. We do this, because we want to be able to
  // i) construct the problem, ii) read the parameters, and iii) initialize all
  // objects according to what is specified in a parameter file.
  //
  // We construct the parameters of our problem in the internal class
  // DistributedLagrangeProblemParameters, derived from ParameterAcceptor. The
  // DistributedLagrangeProblem class takes a const reference to a
  // DistributedLagrangeProblemParameters object, so that it is not possible to
  // modify the parameters from within the DistributedLagrangeProblem class
  // itself.
  //
  // We could have initialized the parameters first, and then pass the
  // parameters to the DistributedLagrangeProblem assuming all entries are set to
  // the desired values, but this has two disadvantages:
  //
  // - we should not make assumptions on how the user initializes a class that
  // is not under our direct control. If the user fails to initialize the
  // class, we should notice and throw an exception;
  //
  // - not all objects that need to read parameters from a parameter file may
  // be available when we construct the DistributedLagrangeProblemParameters;
  // this is often the case for complex programs, with multiple physics, or
  // where we reuse existing code in some external classes. We simulate this by
  // keeping some "complex" objects, like ParsedFunction objects, inside the
  // DistributedLagrangeProblem instead of inside the
  // DistributedLagrangeProblemParameters.
  //
  // Here we assume that upon construction, the classes that build up our
  // problem are not usable yet. Parsing the parameter file is what ensures we have
  // all ingredients to build up our classes, and we design them so that if parsing
  // was not executed, the run is aborted.

  template <int dim, int spacedim=dim>
  class DistributedLagrangeProblem
  {
  public:

    // The DistributedLagrangeProblemParameters is derived from
    // ParameterAcceptor. This allows us to use the
    // ParameterAcceptor::add_parameter methods in its constructor.
    //
    // The members of this function are all non-const, but the
    // DistributedLagrangeProblem class takes a const reference to a
    // DistributedLagrangeProblemParameters object, so that it is not possible
    // to modify the parameters from within the DistributedLagrangeProblem
    // class itself.

    class DistributedLagrangeProblemParameters : public ParameterAcceptor
    {
    public:
      DistributedLagrangeProblemParameters();

      // Initial refinement for the embedding grid, corresponding to the domain
      // $\Omega$.
      unsigned int initial_refinement                           = 4;

      // We allow also a local refinement in the domain $\Omega$, where there
      // is overlap between the embedded grid and the embedding grid. If
      // `delta_refinement` is greater than zero, then we mark each cell of
      // the space grid that contains a vertex of the embedded grid, execute
      // the refinement, and repeat the process `delta_refinement` times.
      unsigned int delta_refinement                             = 3;

      // Starting refinement of the embedding grid, corresponding to the domain
      // $\Omega$.
      unsigned int initial_embedded_refinement                  = 7;

      // A list of boundary ids where we impose homogeneous Dirichlet boundary
      // conditions. On the remaining boundary ids (if any), we impose
      // homogeneous Neumann boundary conditions
      std::list<types::boundary_id> homogeneous_dirichlet_ids   {0};

      // FiniteElement degree of the embedding space
      unsigned int embedding_space_finite_element_degree        = 1;

      // FiniteElement degree of the embedded space
      unsigned int embedded_space_finite_element_degree         = 1;

      // FiniteElement degree of the space used to describe the deformation
      // of the embedded domain
      unsigned int embedded_configuration_finite_element_degree = 1;

      // Order of the quadrature formula used to integrate the coupling
      unsigned int coupling_quadrature_order                    = 3;

      // If set to true, then the embedded configuration function is
      // interpreted as a displacement function
      bool use_displacement                                     = false;

      // A flag to keep track if we were initialized or not
      bool initialized                                          = false;
    };

    DistributedLagrangeProblem(const DistributedLagrangeProblemParameters &parameters);

    // Entry point for the DistributedLagrangeProblem
    void run();

  private:
    // The actual parameters
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

  // At construction time, we initialize also the ParameterAcceptor class, with
  // the section name we want our problem to use when parsing the parameter
  // file.
  //
  // This gives us a nice opportunity: ParameterAcceptor allows you to specify
  // the section name using unix conventions on paths. If the section name
  // starts with a slash ("/"), then the section is interpreted as an *absolute
  // path*, ParameterAcceptor enters a subsection for each directory in the
  // path, using the last name it encountered as the landing subsection for the
  // current class.
  //
  // For example, if you construct your class using
  // ParameterAcceptor("/first/second/third/My Class"), the parameters will be
  // organized as follows:
  //
  // @code
  // subsection first
  //   subsection second
  //     subsection third
  //       subsection My Class
  //        ... # all the parameters
  //       end
  //     end
  //   end
  // end
  // @endcode
  //
  // Internally, the *current path* stored in ParameterAcceptor, is now
  // considered to be "/first/second/third/", i.e., when you specify an
  // absolute path, ParameterAcceptor *changes* the current section to the
  // current path, i.e., to the path of the section name until the *last* "/".
  //
  // If you now construct another class derived from ParameterAcceptor using a
  // relative path (e.g., ParameterAcceptor("My Other class")), you'll end up
  // with
  // @code
  // subsection first
  //   subsection second
  //     subsection third
  //       subsection MyClass
  //        ... # all the parameters
  //       end
  //       subsection My Other Class
  //        ... # all the parameters of MyOtherClass
  //       end
  //     end
  //   end
  // end
  // @endcode
  //
  // If the section name *ends* with a slash, say, similar to the example above
  // we have two classes, one initialized with "/first/second/third/My Class/"
  // and the other with "My Other class", then the resulting parameter file will
  // look like:
  //
  // @code
  // subsection first
  //   subsection second
  //     subsection third
  //       subsection MyClass
  //        ... # all the parameters
  //        subsection My Other Class
  //         ... # all the parameters of MyOtherClass
  //        end
  //       end
  //     end
  //   end
  // end
  // @endcode
  //
  // We are going to exploit this, by making our
  // DistributedLagrangeProblemParameters
  // the *parent* of all subsequently constructed classes. Since most of the other
  // classes are members of DistributedLagrangeProblem, this allows one to construct,
  // for example, two DistributedLagrangeProblem for two different dimensions, without
  // having conflicts in the parameters for the two problems.

  template<int dim, int spacedim>
  DistributedLagrangeProblem<dim,spacedim>::DistributedLagrangeProblemParameters::
  DistributedLagrangeProblemParameters() :
    ParameterAcceptor("/Distributed Lagrange<" + Utilities::int_to_string(dim)
                      + "," + Utilities::int_to_string(spacedim) +">/")
  {

    // The ParameterAcceptor::add_parameter does a few things:
    //
    // - enters the subsection specified at construction time to ParameterAcceptor
    //
    // - calls the ParameterAcceptor::prm.add_parameter
    //
    // - calls any signal you may have attached to
    // ParameterAcceptor::declare_parameters_call_back
    //
    // - leaves the subsection
    //
    // In turns, ParameterAcceptor::prm.add_parameter
    //
    // - declares an entry in the parameter handler for the given variable;
    //
    // - reads the value of the variable,
    //
    // - transforms it to a string, used as the default value for the parameter
    // file
    //
    // - attaches an *action* to ParameterAcceptor::prm that monitors when a file
    // is parsed, or when an entry is set, and when this happens, it updates the
    // content of the given variable to the value parsed by the string

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
    // Here is a way to set default values for a ParameterAcceptor class
    // that was constructed using ParameterAcceptorProxy.
    //
    // In this case, we set the default deformation of the embedded grid to be
    // a circle with radius `R` and center (Cx, Cy).

    embedded_configuration_function.declare_parameters_call_back.connect(
          [&] () -> void
          {
            ParameterAcceptor::prm.set("Function constants", "R=.3, Cx=.4, Cy=.4");
            ParameterAcceptor::prm.set("Function expression", "R*cos(2*pi*x)+Cx; R*sin(2*pi*x)+Cy");
          });

    embedded_value_function.declare_parameters_call_back.connect(
          [&] () -> void
          {
            ParameterAcceptor::prm.set("Function expression", "1");
          });

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
