// ---------------------------------------------------------------------
//
// Copyright (C) 2020 - 2022 by the deal.II authors
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


/**
 * Test global-coarsening multigrid for a uniformly refined mesh both for
 * simplex and hypercube mesh.
 */

#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include "mg_transfer_matrix_based.h"
// #include "multigrid_util.h"

template <int dim, typename Number = double>
void
test(const unsigned int n_refinements,
     const unsigned int fe_degree_fine,
     const bool         do_simplex_mesh)
{
  using VectorTypeMB = Vector<Number>;

  const unsigned int min_level = 0;
  const unsigned int max_level = n_refinements;

  MGLevelObject<Triangulation<dim>> triangulations(min_level, max_level);
  MGLevelObject<DoFHandler<dim>>    dof_handlers(min_level, max_level);
  MGLevelObject<SparsityPattern>    sparsity_patterns(min_level, max_level);
  MGLevelObject<std::unique_ptr<GridTools::Cache<dim>>> caches(min_level,
                                                               max_level);
  MGLevelObject<AffineConstraints<Number>> constraints(min_level, max_level);
  MGLevelObject<MGTwoLevelTransfer<dim, VectorTypeMB>> transfers_mb(min_level,
                                                                    max_level);
  MGLevelObject<SparseMatrix<Number>> operators(min_level, max_level);

  std::unique_ptr<Mapping<dim>> mapping_;

  // set up levels
  for (auto l = min_level; l <= max_level; ++l)
    {
      auto &tria        = triangulations[l];
      auto &dof_handler = dof_handlers[l];
      auto &cache       = caches[l];
      auto &constraint  = constraints[l];
      auto &op          = operators[l];
      auto &sp          = sparsity_patterns[l];

      std::unique_ptr<FiniteElement<dim>> fe;
      std::unique_ptr<Quadrature<dim>>    quad;
      std::unique_ptr<Mapping<dim>>       mapping;

      if (do_simplex_mesh)
        {
          fe      = std::make_unique<FE_SimplexP<dim>>(fe_degree_fine);
          quad    = std::make_unique<QGaussSimplex<dim>>(fe_degree_fine + 1);
          mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
        }
      else
        {
          fe      = std::make_unique<FE_Q<dim>>(fe_degree_fine);
          quad    = std::make_unique<QGauss<dim>>(2 * fe_degree_fine + 1);
          mapping = std::make_unique<MappingFE<dim>>(FE_Q<dim>(1));
        }

      if (l == max_level)
        mapping_ = mapping->clone();

      // set up triangulation
      if (do_simplex_mesh)
        GridGenerator::subdivided_hyper_cube_with_simplices(tria, 2);
      else
        GridGenerator::subdivided_hyper_cube(tria, 2);
      tria.refine_global(l);

      // set up dofhandler
      dof_handler.reinit(tria);
      dof_handler.distribute_dofs(*fe);
      std::cout << dof_handler.n_dofs() << std::endl;

      // set up caches
      cache = std::make_unique<GridTools::Cache<dim>>(tria);

      // set up constraints
      // IndexSet locally_relevant_dofs;
      // DoFTools::extract_locally_relevant_dofs(dof_handler,
      //                                         locally_relevant_dofs);
      // constraint.reinit(locally_relevant_dofs);
      constraint.clear();
      DoFTools::make_hanging_node_constraints(dof_handler, constraint);
      VectorTools::interpolate_boundary_values(
        *mapping, dof_handler, 0, Functions::ZeroFunction<dim>(), constraint);
      constraint.close();

      // set up operator
      // op.reinit(*mapping, dof_handler, *quad, constraint);
      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sp.copy_from(dsp);
      op.reinit(sp);
      MatrixTools::create_laplace_matrix(*mapping,
                                         dof_handler,
                                         *quad,
                                         op,
                                         (const Function<dim> *const) nullptr,
                                         constraint);
    }

  // set up transfer operator
  for (unsigned int l = min_level; l < max_level; ++l)
    {
      transfers_mb[l + 1].reinit(*caches[l + 1],
                                 *caches[l],
                                 dof_handlers[l + 1],
                                 dof_handlers[l]);
      std::cout << "l=" << l << dof_handlers[l].n_dofs() << std::endl;
      std::cout << "l+1=" << l + 1 << dof_handlers[l + 1].n_dofs() << std::endl;
    }


  // MGTransferGlobalCoarsening<dim, VectorTypeMB> transfer(
  //   transfers,
  //   [&](const auto l, auto &vec) { operators[l].initialize_dof_vector(vec);
  //   });
  MGTransferGlobalCoarsening<dim, VectorTypeMB> transfer_mb(transfers_mb, true);

  GMGParameters mg_data; // TODO

  VectorTypeMB dst, src;
  dst.reinit(dof_handlers[max_level].n_dofs());
  src.reinit(dof_handlers[max_level].n_dofs());
  // operators[max_level].initialize_dof_vector(dst);
  // operators[max_level].initialize_dof_vector(src);
  // operators[max_level].rhs(src);

  VectorTools::create_right_hand_side(*mapping_,
                                      dof_handlers[max_level],
                                      QGauss<dim>(2 * fe_degree_fine + 1),
                                      Functions::ConstantFunction<dim>(1.),
                                      src);

  {
    // Just testing.
    SolverControl          solver_control(1000, 1e-12);
    SolverCG<VectorTypeMB> cg(solver_control);
    cg.solve(operators[max_level], dst, src, PreconditionIdentity());
    constraints[max_level].distribute(dst);

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handlers[max_level]);
    data_out.add_data_vector(
      dst,
      "solution",
      DataOut_DoFData<dim, dim>::DataVectorType::type_dof_data);
    data_out.build_patches(*mapping_, 2);

    std::ofstream output("test_consistency.vtk");
    data_out.write_vtk(output);
  }

  ReductionControl solver_control(
    mg_data.maxiter, mg_data.abstol, mg_data.reltol, false, false);

  mg_solve(solver_control,
           dst,
           src,
           mg_data,
           dof_handlers[max_level],
           operators[max_level],
           operators,
           transfer_mb);

  deallog << dim << ' ' << fe_degree_fine << ' ' << n_refinements << ' '
          << (do_simplex_mesh ? "tri " : "quad") << ' '
          << solver_control.last_step() << std::endl;

  // MGLevelObject<VectorTypeMB> results(min_level, max_level);

  // transfer.interpolate_to_mg(dof_handlers[max_level], results, dst);

  for (unsigned int l = max_level; l <= max_level; ++l)
    {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(dof_handlers[max_level]);
      data_out.add_data_vector(
        dst,
        "solution",
        DataOut_DoFData<dim, dim>::DataVectorType::type_dof_data);
      data_out.build_patches(*mapping_, 2);

      std::ofstream output("test." + std::to_string(dim) + "." +
                           std::to_string(l) + ".vtk");
      data_out.write_vtk(output);
    }
}

int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    all;

  deallog.precision(8);

  for (unsigned int n_refinements = 2; n_refinements <= 4; ++n_refinements)
    for (unsigned int degree = 2; degree <= 2; ++degree)
      test<2>(n_refinements, degree, false /*quadrilateral*/);

  // for (unsigned int n_refinements = 2; n_refinements <= 4; ++n_refinements)
  //   for (unsigned int degree = 2; degree <= 2; ++degree)
  //     test<2>(n_refinements, degree, true /*triangle*/);
}
