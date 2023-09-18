/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2023 by the deal.II authors
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
 * Author: Marco Feder, (SISSA), 2023
 *         Peter Munch, University of Augsburg, 2023
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples. We
// include headers related to the FiniteElement space we want to work with, to
// the TrilinosWrappers for the coarse grid solver, MatrixFree tools for
// operator evaluation and Multigrid infrastructure.
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/distributed/shared_tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

// The following include file is the one containing the implementation of actual
// transfer between non-nested levels:
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// This is just C+:
#include <fstream>


// As we want to use as coarse grid solver the Trilinos implementation of AMG,
// the following class is just a wrapper that transforms a given preconditioner
// into a coarse grid solver.
namespace dealii
{
  template <class VectorType, class PreconditionerType>
  class MGCoarseGridApplyPreconditioner : public MGCoarseGridBase<VectorType>
  {
  public:
    /**
     * Default constructor.
     */
    MGCoarseGridApplyPreconditioner();

    /**
     * Constructor. Store a pointer to the preconditioner for later use.
     */
    MGCoarseGridApplyPreconditioner(const PreconditionerType &precondition);

    /**
     * Clear the pointer.
     */
    void clear();

    /**
     * Initialize new data.
     */
    void initialize(const PreconditionerType &precondition);

    /**
     * Implementation of the abstract function.
     */
    virtual void operator()(const unsigned int level,
                            VectorType        &dst,
                            const VectorType  &src) const override;

  private:
    /**
     * Reference to the preconditioner.
     */
    SmartPointer<
      const PreconditionerType,
      MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>>
      preconditioner;
  };



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner()
    : preconditioner(0, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::
    MGCoarseGridApplyPreconditioner(const PreconditionerType &preconditioner)
    : preconditioner(&preconditioner, typeid(*this).name())
  {}



  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::initialize(
    const PreconditionerType &preconditioner_)
  {
    preconditioner = &preconditioner_;
  }



  template <class VectorType, class PreconditionerType>
  void MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::clear()
  {
    preconditioner = 0;
  }


  namespace internal
  {
    namespace MGCoarseGridApplyPreconditioner
    {
      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void solve(const PreconditionerType preconditioner,
                 VectorType              &dst,
                 const VectorType        &src)
      {
        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst, src);
      }

      template <class VectorType,
                class PreconditionerType,
                typename std::enable_if<
                  !std::is_same<typename VectorType::value_type, double>::value,
                  VectorType>::type * = nullptr>
      void solve(const PreconditionerType preconditioner,
                 VectorType              &dst,
                 const VectorType        &src)
      {
        LinearAlgebra::distributed::Vector<double> src_;
        LinearAlgebra::distributed::Vector<double> dst_;

        src_ = src;
        dst_ = dst;

        // to allow the case that the preconditioner was only set up on a
        // subset of processes
        if (preconditioner != nullptr)
          preconditioner->vmult(dst_, src_);

        dst = dst_;
      }
    } // namespace MGCoarseGridApplyPreconditioner
  }   // namespace internal


  template <class VectorType, class PreconditionerType>
  void
  MGCoarseGridApplyPreconditioner<VectorType, PreconditionerType>::operator()(
    const unsigned int /*level*/,
    VectorType       &dst,
    const VectorType &src) const
  {
    internal::MGCoarseGridApplyPreconditioner::solve(preconditioner, dst, src);
  }
} // namespace dealii



namespace Step88
{
  using namespace dealii;

  // The next struct contains parameters such as the polynomial degree of the
  // FiniteElement, the spatial dimension on the problem, the number of global
  // refinements as well as parameters for the geometric multigrid scheme.
  struct Parameters
  {
    // As said above, by default we will work with grids generated externally.
    std::string  mesh_type            = "mesh_file";
    unsigned int dim                  = 2;
    unsigned int n_global_refinements = 3;
    unsigned int fe_degree            = 2;

    // The following parameters are related to geometric multigrid.
    unsigned int solver_max_iterations = 100;
    double       solver_abs_tolerance  = 1e-20;
    double       solver_rel_tolerance  = 1e-4;

    unsigned int mg_smoothing_range              = 20;
    unsigned int mg_smoother_degree              = 5;
    unsigned int mg_smoother_eig_cg_n_iterations = 20;
    bool         mg_non_nested                   = true;

    void parse(const std::string file_name)
    {
      dealii::ParameterHandler prm;
      add_parameters(prm);

      std::ifstream file;
      file.open(file_name);
      prm.parse_input_from_json(file, true);
    }

    // This function simply reads the grid from a given level. The sequence of
    // external grids has the form <code>piston_0.inp,piston_1.inp, ...</code>,
    // where the integer suffix denotes the level (with the convention that the
    // greter the index, the finer the level).
    std::string get_mesh_file_name(const unsigned int level) const
    {
      char buffer[100];

      std::snprintf(buffer, 100, mesh_file_format.c_str(), level);

      return {buffer};
    }

  private:
    std::string mesh_file_format = "";

    // As the name suggests, the following member function simply adds all
    // parameters discussed above using ParameterAcceptor::add_parameter().
    void add_parameters(ParameterHandler &prm)
    {
      prm.add_parameter(
        "MeshType",
        mesh_type,
        "",
        Patterns::Selection(
          "hyper_cube|hyper_cube_with_simplices|mesh_file|gmesh_journal"));
      prm.add_parameter("MeshFileFormat", mesh_file_format);
      prm.add_parameter("Dimension", dim);
      prm.add_parameter("NGlobalRefinements", n_global_refinements);
      prm.add_parameter("Degree", fe_degree);

      prm.add_parameter("SolverMaxIterations", solver_max_iterations);
      prm.add_parameter("SolverAbsTolerance", solver_abs_tolerance);
      prm.add_parameter("SolverRelTolerance", solver_rel_tolerance);

      prm.add_parameter("MGSmoothingScheme", mg_smoothing_range);
      prm.add_parameter("MGSmootherDegree", mg_smoother_degree);
      prm.add_parameter("MGSmootherEigNIterations",
                        mg_smoother_eig_cg_n_iterations);
      prm.add_parameter("MGNonNested", mg_non_nested);
    }
  };


  // The following class only implements the evaluation of the Laplace operator.
  // We refer to step-37 for an extended discussion on how this is done.
  // Alternatively, one could also use directly the
  // MatrixFreeOperators::LaplaceOperator class.
  template <int dim, typename number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    using VectorType       = LinearAlgebra::distributed::Vector<number>;
    using FECellIntegrator = FEEvaluation<dim, -1, 0, 1, number>;

    void reinit(const Mapping<dim>              &mapping,
                const DoFHandler<dim>           &dof_handler,
                const AffineConstraints<number> &constraints,
                const Quadrature<dim>           &quad);

    types::global_dof_index m() const;

    number el(unsigned int, unsigned int) const;

    void initialize_dof_vector(VectorType &vec) const;

    void vmult(VectorType &dst, const VectorType &src) const;

    void Tvmult(VectorType &dst, const VectorType &src) const;

    void compute_inverse_diagonal(VectorType &diagonal) const;

    const TrilinosWrappers::SparseMatrix &get_system_matrix() const;

    std::shared_ptr<const Utilities::MPI::Partitioner> get_partitioner() const;

  private:
    void do_cell_integral_local(FECellIntegrator &integrator) const;

    void do_cell_integral_global(FECellIntegrator &integrator,
                                 VectorType       &dst,
                                 const VectorType &src) const;


    void do_cell_integral_range(
      const MatrixFree<dim, number>               &matrix_free,
      VectorType                                  &dst,
      const VectorType                            &src,
      const std::pair<unsigned int, unsigned int> &range) const;

    MatrixFree<dim, number>                matrix_free;
    mutable TrilinosWrappers::SparseMatrix system_matrix;
  };



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::reinit(
    const Mapping<dim>              &mapping,
    const DoFHandler<dim>           &dof_handler,
    const AffineConstraints<number> &constraints,
    const Quadrature<dim>           &quad)
  {
    this->system_matrix.clear();

    typename MatrixFree<dim, number>::AdditionalData data;
    data.mapping_update_flags = update_gradients;

    matrix_free.reinit(mapping, dof_handler, constraints, quad, data);
  }


  template <int dim, typename number>
  types::global_dof_index LaplaceOperator<dim, number>::m() const
  {
    return matrix_free.get_dof_handler().n_dofs();
  }



  template <int dim, typename number>
  number LaplaceOperator<dim, number>::el(unsigned int, unsigned int) const
  {
    Assert(false, ExcNotImplemented());
    return 0;
  }



  template <int dim, typename number>
  void
  LaplaceOperator<dim, number>::initialize_dof_vector(VectorType &vec) const
  {
    matrix_free.initialize_dof_vector(vec);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::vmult(VectorType       &dst,
                                           const VectorType &src) const
  {
    this->matrix_free.cell_loop(
      &LaplaceOperator::do_cell_integral_range, this, dst, src, true);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::Tvmult(VectorType       &dst,
                                            const VectorType &src) const
  {
    this->vmult(dst, src);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::compute_inverse_diagonal(
    VectorType &diagonal) const
  {
    this->matrix_free.initialize_dof_vector(diagonal);
    MatrixFreeTools::compute_diagonal(matrix_free,
                                      diagonal,
                                      &LaplaceOperator::do_cell_integral_local,
                                      this);

    for (auto &i : diagonal)
      i = (std::abs(i) > 1.0e-10) ? (1.0 / i) : 1.0;
  }



  template <int dim, typename number>
  const TrilinosWrappers::SparseMatrix &
  LaplaceOperator<dim, number>::get_system_matrix() const
  {
    if (system_matrix.m() == 0 && system_matrix.n() == 0)
      {
        const auto &dof_handler = this->matrix_free.get_dof_handler();

        TrilinosWrappers::SparsityPattern dsp(
          dof_handler.locally_owned_dofs(),
          dof_handler.get_triangulation().get_communicator());

        DoFTools::make_sparsity_pattern(dof_handler,
                                        dsp,
                                        matrix_free.get_affine_constraints());

        dsp.compress();
        system_matrix.reinit(dsp);

        MatrixFreeTools::compute_matrix(
          matrix_free,
          matrix_free.get_affine_constraints(),
          system_matrix,
          &LaplaceOperator::do_cell_integral_local,
          this);
      }

    return this->system_matrix;
  }



  template <int dim, typename number>
  std::shared_ptr<const Utilities::MPI::Partitioner>
  LaplaceOperator<dim, number>::get_partitioner() const
  {
    return matrix_free.get_vector_partitioner();
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_local(
    FECellIntegrator &integrator) const
  {
    integrator.evaluate(EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate(EvaluationFlags::gradients);
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_range(
    const MatrixFree<dim, number>               &matrix_free,
    VectorType                                  &dst,
    const VectorType                            &src,
    const std::pair<unsigned int, unsigned int> &range) const
  {
    FECellIntegrator integrator(matrix_free, range);

    for (unsigned cell = range.first; cell < range.second; ++cell)
      {
        integrator.reinit(cell);

        do_cell_integral_global(integrator, dst, src);
      }
  }



  template <int dim, typename number>
  void LaplaceOperator<dim, number>::do_cell_integral_global(
    FECellIntegrator &integrator,
    VectorType       &dst,
    const VectorType &src) const
  {
    integrator.gather_evaluate(src, EvaluationFlags::gradients);

    for (unsigned int q = 0; q < integrator.n_q_points; ++q)
      integrator.submit_gradient(integrator.get_gradient(q), q);

    integrator.integrate_scatter(EvaluationFlags::gradients, dst);
  }



  // @sect3{The <code>Step88</code> class template}

  // The main class has the same structure as in all other tutorial
  // programs, with the obvious exception that no matrix needs to be assembled
  // in this case. Notice that the create_grids() function return a boolean that
  // is used to switch at runtime to a nested multigrid scheme if levels are
  // nested.
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem(const Parameters &params);

    void run();

  private:
    bool create_grids();

    void setup_system();

    void solve();

    void output_results();

    using Number     = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    const MPI_Comm     comm;
    const Parameters   params;
    const unsigned int min_level;
    const unsigned int max_level;
    ConditionalOStream pcout;

    // Each level is described by an independent Triangulation object that will
    // have to be accessed during the setup phase of the multigrid method.
    // Hence, we store them in a standard vector of shared pointers.
    std::vector<std::shared_ptr<Triangulation<dim>>> triangulations;

    std::unique_ptr<FiniteElement<dim>> fe;
    std::unique_ptr<Mapping<dim>>       mapping;
    Quadrature<dim>                     quadrature;

    // The following objects are used to store, on every level of the hierarchy,
    // its associated DoFHandler, AffineConstraints and the underlying
    // differential operator.
    MGLevelObject<DoFHandler<dim>>              dof_handlers;
    MGLevelObject<AffineConstraints<Number>>    constraints;
    MGLevelObject<LaplaceOperator<dim, Number>> operators;

    // The next vectors store right-hand-side and solution.
    VectorType rhs;
    VectorType solution;
  };



  // @sect4{Step88::LaplaceProblem}
  // This constructor is pretty straightforward: all it does it to initialize
  // some parameters, the number of levels and parallel output stream.
  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem(const Parameters &params)
    : comm(MPI_COMM_WORLD)
    , params(params)
    , min_level(0)
    , max_level(params.n_global_refinements)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(comm) == 0))
  {}


  // @sect4{Step88::create_grids}
  // Next, we fill the hierarchy of levels for the multigrid method. Depending
  // on the selected type of mesh, we loop through the levels and for each one
  // we create (or import) a different Triangulation that discretizes the same
  // geoemtry. This member function returns true if the grid is created with the
  // GridGenerator namespace, false otherwise.
  template <int dim>
  bool LaplaceProblem<dim>::create_grids()
  {
    // If the geometry is a simple one such a <code>hyper_cube</code>, we can
    // just use the GridGenerator namespace. Since the present infrastructure
    // also support simplex meshes, we allow their usage by using
    // GridGenerator::convert_hypercube_to_simplex_mesh.
    if (params.mesh_type == "hyper_cube")
      {
        for (unsigned int l = min_level; l <= max_level; ++l)
          {
            auto triangulation =
              std::make_shared<parallel::shared::Triangulation<dim>>(comm);
            GridGenerator::hyper_cube(*triangulation);
            triangulation->refine_global(l);
            triangulations.push_back(triangulation);
          }

        return true;
      }
    else if (params.mesh_type == "hyper_cube_with_simplices")
      {
        Triangulation<dim> dummy;
        GridGenerator::hyper_cube(dummy);

        for (unsigned int l = min_level; l <= max_level; ++l)
          {
            auto triangulation =
              std::make_shared<parallel::shared::Triangulation<dim>>(comm);
            GridGenerator::convert_hypercube_to_simplex_mesh(dummy,
                                                             *triangulation);
            triangulation->refine_global(l);
            triangulations.push_back(triangulation);
          }

        return true;
      }

    // By default we will import each level separately by reading it from
    // externally generated grids. Notice that each triangulation here is
    // a parallel::distributed::Triangulation independently partitioned.
    else if (params.mesh_type == "mesh_file")
      {
        for (unsigned int l = min_level; l <= max_level; ++l)
          {
            auto triangulation =
              std::make_shared<parallel::distributed::Triangulation<dim>>(comm);

            GridIn<dim> grid_in(*triangulation);
            // We chose the Abaqus format just for convenience, but adapting
            // the present code to other formats is easy, as long as the format
            // is supported by the GridIn class.
            grid_in.read(params.get_mesh_file_name(l), GridIn<dim>::abaqus);

            triangulations.push_back(triangulation);
          }

        return false;
      }
    else if (params.mesh_type == "gmesh_journal")
      {
        AssertThrow(false, ExcNotImplemented());
        return false;
      }

    AssertThrow(false, ExcNotImplemented());
    return true;
  }


  // @sect4{Step88::setup_system}
  // After generating all levels, we can define on each one of them a suitable
  // FiniteElement space and distribute the DoFs.
  template <int dim>
  void LaplaceProblem<dim>::setup_system()
  {
    // Depending on the types of cells of the triangulation, we use the
    // appropriate FiniteElement spaces, Mapping and Quadrature rules:
    if (triangulations.back()->all_reference_cells_are_hyper_cube())
      fe = std::make_unique<FE_Q<dim>>(params.fe_degree);
    else if (triangulations.back()->all_reference_cells_are_simplex())
      fe = std::make_unique<FE_SimplexP<dim>>(params.fe_degree);
    else
      AssertThrow(false, ExcNotImplemented());

    const auto reference_cell = triangulations.back()->get_reference_cells()[0];

    mapping    = reference_cell.template get_default_mapping<dim>(1);
    quadrature = reference_cell.template get_gauss_type_quadrature<dim>(
      params.fe_degree + 1);

    // We set each MGLevelObject to its correct size before filling it:
    dof_handlers.resize(min_level, max_level);
    constraints.resize(min_level, max_level);
    operators.resize(min_level, max_level);

    // Next, we start looping over all levels:
    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        // We distribute the DoFs on each level in the standard way:
        dof_handlers[l].reinit(*triangulations[l]);
        dof_handlers[l].distribute_dofs(*fe);

        pcout << "Number of DoFs on level " + std::to_string(l) + ": "
              << dof_handlers[l].n_dofs() << std::endl;

        // Constraints are handled in the usual way. Since we chose homogeneous
        // Dirichlet boundary conditions, we directly call
        // DoFTools::make_zero_boundary_constraints() in order to enforce the
        // solution to be zero on all of the boundary.
        constraints[l].reinit(
          DoFTools::extract_locally_relevant_dofs(dof_handlers[l]));
        DoFTools::make_zero_boundary_constraints(dof_handlers[l],
                                                 constraints[l]);
        constraints[l].close();

        // Finally, we conclude the setup with the evaluation of the
        // LaplaceOperator on the present level.
        operators[l].reinit(*mapping,
                            dof_handlers[l],
                            constraints[l],
                            quadrature);
      }

    // We then initialize solution and right-hand-side vector on the finest
    // level.
    operators.back().initialize_dof_vector(solution);
    operators.back().initialize_dof_vector(rhs);

    VectorTools::create_right_hand_side(*mapping,
                                        dof_handlers.back(),
                                        quadrature,
                                        Functions::ConstantFunction<dim>(1.0),
                                        rhs,
                                        constraints.back());
  }



  // @sect4{Step88::solve}
  // We are now ready to initialize the intergrid transfer operators,
  // smoothers and preconditioner.
  template <int dim>
  void LaplaceProblem<dim>::solve()
  {
    // We define some useful aliases for better code readability:
    using LevelMatrixType            = LaplaceOperator<dim, Number>;
    using SmootherPreconditionerType = DiagonalMatrix<VectorType>;
    using SmootherType               = PreconditionChebyshev<LevelMatrixType,
                                               VectorType,
                                               SmootherPreconditionerType>;
    using MGTransferType             = MGTransferMF<dim, Number>;
    using PreconditionerType = PreconditionMG<dim, VectorType, MGTransferType>;

    // We then store a vector of pointers to two-level transfer operators.
    // Since we only know at runtime the actual type of transfer (depending on
    // the nested or non-nested nature of the hierarchy), we use the base
    // class MGTwoLevelTransferBase. In order to setup the transfer operator
    // from one level to the next, it is sufficient to call the
    // MGTwoLevelTransferNonNested::reinit() (or
    // MGTwoLevelTransfer::reinit_geometric_transfer() in case levels are
    // nested). The only difference between the two interfaces is that the
    // latter does not take Mapping arguments.
    MGLevelObject<std::shared_ptr<const MGTwoLevelTransferBase<VectorType>>>
      transfers(min_level, max_level);

    for (unsigned int l = min_level; l < max_level; ++l)
      if (params.mg_non_nested)
        {
          auto transfer =
            std::make_shared<MGTwoLevelTransferNonNested<dim, VectorType>>();

          transfer->reinit(dof_handlers[l + 1],
                           dof_handlers[l],
                           *mapping,
                           *mapping,
                           constraints[l + 1],
                           constraints[l]);

          transfers[l + 1] = transfer;
        }
      else
        {
          auto transfer =
            std::make_shared<MGTwoLevelTransfer<dim, VectorType>>();

          transfer->reinit_geometric_transfer(dof_handlers[l + 1],
                                              dof_handlers[l],
                                              constraints[l + 1],
                                              constraints[l]);

          transfers[l + 1] = transfer;
        }

    // After the setup of transfer operators, we can initialize multigrid
    // operators, smoothers and related data structures:
    std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>>
      partitioners;

    for (unsigned int l = min_level; l <= max_level; ++l)
      partitioners.push_back(operators[l].get_partitioner());

    MGTransferMF<dim, Number> mg_transfer;
    mg_transfer.intitialize_two_level_transfers(transfers);
    mg_transfer.build(partitioners);

    mg::Matrix<VectorType> mg_matrix(operators);

    MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
      min_level, max_level);

    for (unsigned int l = min_level; l <= max_level; ++l)
      {
        smoother_data[l].preconditioner =
          std::make_shared<SmootherPreconditionerType>();
        operators[l].compute_inverse_diagonal(
          smoother_data[l].preconditioner->get_vector());
        smoother_data[l].smoothing_range = params.mg_smoothing_range;
        smoother_data[l].degree          = params.mg_smoother_degree;
        smoother_data[l].eig_cg_n_iterations =
          params.mg_smoother_eig_cg_n_iterations;
      }

    MGSmootherPrecondition<LevelMatrixType, SmootherType, VectorType>
      mg_smoother;
    mg_smoother.initialize(operators, smoother_data);

    // As coarse-grid solver we use the algebraic multigrid (AMG)
    // preconditioner based on the Trilinos ML implementation.
    TrilinosWrappers::PreconditionAMG precondition_amg;
    precondition_amg.initialize(operators[min_level].get_system_matrix());

    MGCoarseGridApplyPreconditioner<VectorType,
                                    TrilinosWrappers::PreconditionAMG>
      mg_coarse(precondition_amg);

    // Finally, we can initialize the Multigrid object and use it to
    // precondition our conjugate-gradient solver. Parameters related to the
    // CG algorithm are read from the parameter files.
    Multigrid<VectorType> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);

    PreconditionerType preconditioner(dof_handlers.back(), mg, mg_transfer);

    ReductionControl solver_control(params.solver_max_iterations,
                                    params.solver_abs_tolerance,
                                    params.solver_rel_tolerance);

    SolverCG<VectorType>(solver_control)
      .solve(operators.back(), solution, rhs, preconditioner);

    pcout << "Solved in " << solver_control.last_step() << " steps."
          << std::endl;
  }



  // @sect4{Step88::output_results}
  // The following function simply generates standard result output.
  template <int dim>
  void LaplaceProblem<dim>::output_results()
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handlers.back());
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches(*mapping);

    data_out.write_vtu_in_parallel("solution.vtu", comm);

    for (unsigned int l = 0; l < triangulations.size(); ++l)
      {
        DataOut<dim> data_out;
        data_out.attach_triangulation(*triangulations[l]);
        data_out.build_patches();

        data_out.write_vtu_in_parallel("grid_" + std::to_string(l) + ".vtu",
                                       comm);
      }
  }



  // The run function is standard: it simply calls all other methods in the
  // correct order.
  template <int dim>
  void LaplaceProblem<dim>::run()
  {
    const bool nested_mesh = create_grids();
    AssertThrow(nested_mesh || params.mg_non_nested, ExcNotImplemented());

    setup_system();
    solve();
    output_results();
  }

} // namespace Step88



int main(int argc, char *argv[])
{
  try
    {
      using namespace dealii;
      using namespace Step88;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      Parameters prm;
      if (argc > 1)
        prm.parse(std::string(argv[1]));

      // Since the external grids are three-dimensional, we make sure this
      // program is run in 3D only.
      AssertThrow(prm.dim == 3, ExcNotImplemented());
      LaplaceProblem<3> laplace_problem(prm);
      laplace_problem.run();
    }
  catch (const std::exception &exc)
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
