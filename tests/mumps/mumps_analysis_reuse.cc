// -----------------------------------------------------------------------------
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception OR LGPL-2.1-or-later
// Copyright (C) 2001 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Detailed license information governing the source code and contributions
// can be found in LICENSE.md and CONTRIBUTING.md at the top level directory.
//
// -----------------------------------------------------------------------------

// Check split analysis/factorization, zero crossings, matrix lifetime,
// collective pattern/distribution checks, and recovery after failure.

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#  include <deal.II/lac/trilinos_vector.h>
#endif

#include <cmath>

#include "../tests.h"

// Exceptions must be tested in Release as well.
template <typename Function>
void
expect_exception(const Function &function)
{
  bool caught = false;
  try
    {
      function();
    }
  catch (const ExceptionBase &)
    {
      caught = true;
    }
  AssertThrow(caught, ExcInternalError());
}



void
serial_test()
{
  SparsityPattern pattern(3, 3, 2);
  for (unsigned int i = 0; i < 3; ++i)
    pattern.add(i, i);
  pattern.add(0, 1);
  pattern.compress();
  SparseMatrix<double> matrix(pattern);
  for (unsigned int i = 0; i < 3; ++i)
    matrix.set(i, i, 4.);

  SparseDirectMUMPS solver({}, MPI_COMM_SELF);
  Vector<double>    rhs(3), solution(3);
  expect_exception([&]() { solver.factorize(matrix); });
  {
    SparseMatrix<double> temporary(pattern);
    temporary.copy_from(matrix);
    solver.analyze(temporary);
  }
  AssertThrow(solver.get_infog()[0] == 0, ExcInternalError());
  AssertThrow(solver.get_infog()[15] >= 0, ExcInternalError());
  solver.get_icntl()[22] = 32; // ICNTL(23): memory limit in megabytes.
  expect_exception([&]() { solver.vmult(solution, rhs); });
  expect_exception([&]() { solver.Tvmult(solution, rhs); });

  // The same stored entry goes from zero to nonzero and back to zero.
  for (const double value : {0., 1., 0., -1.})
    {
      matrix.set(0, 1, value);
      solver.factorize(matrix);
      // Independent exact solution x=(1,2,3).
      rhs[0] = 4. + 2. * value;
      rhs[1] = 8.;
      rhs[2] = 12.;
      solver.vmult(solution, rhs);
      for (unsigned int i = 0; i < 3; ++i)
        AssertThrow(std::abs(solution[i] - (i + 1.)) < 1.e-12,
                    ExcInternalError());
      AssertThrow(solver.get_infog()[0] == 0, ExcInternalError());
      AssertThrow(solver.get_infog()[21] >= 0, ExcInternalError());
    }

  // Same dimensions and number of entries, but different coordinates.
  SparsityPattern changed(3, 3, 2);
  for (unsigned int i = 0; i < 3; ++i)
    changed.add(i, i);
  changed.add(0, 2);
  changed.compress();
  SparseMatrix<double> other(changed);
  for (unsigned int i = 0; i < 3; ++i)
    other.set(i, i, 4.);
  other.set(0, 2, 1.);
  expect_exception([&]() { solver.factorize(other); });
  // Rejection must happen before MUMPS, rather than as a numerical failure.
  AssertThrow(solver.get_infog()[0] == 0, ExcInternalError());
  expect_exception([&]() { solver.vmult(solution, rhs); });

  solver.initialize(matrix);
  solver.factorize(matrix);
  matrix = 0.;
  expect_exception([&]() { solver.factorize(matrix); });
  AssertThrow(solver.get_infog()[0] < 0, ExcInternalError());
  expect_exception([&]() { solver.vmult(solution, rhs); });
  for (unsigned int i = 0; i < 3; ++i)
    matrix.set(i, i, 4.);
  // A failed numerical factorization leaves the analysis reusable.
  solver.factorize(matrix);
}



IndexSet
owned_rows(const bool all_on_root)
{
  const auto rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  const auto size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
  IndexSet   owned(4);
  if (all_on_root)
    {
      if (rank == 0)
        owned.add_range(0, 4);
    }
  else
    owned.add_range(4 * rank / size, 4 * (rank + 1) / size);
  return owned;
}



template <typename MatrixType>
void
fill(MatrixType     &matrix,
     const IndexSet &owned,
     const double    value,
     const bool      changed = false)
{
  DynamicSparsityPattern pattern(4, 4);
  for (unsigned int i = 0; i < 4; ++i)
    {
      pattern.add(i, i);
      pattern.add(i, (i + 1) % 4);
    }
  if (changed)
    pattern.add(0, 2);
  matrix.reinit(owned, owned, pattern, MPI_COMM_WORLD);
  for (const auto i : owned)
    {
      matrix.set(i, i, 4.);
      matrix.set(i, (i + 1) % 4, value);
    }
  matrix.compress(VectorOperation::insert);
}



void
centralized_test(SparseDirectMUMPS &solver)
{
  SparsityPattern pattern(3, 3, 1);
  for (unsigned int i = 0; i < 3; ++i)
    pattern.add(i, i);
  pattern.compress();
  SparseMatrix<double> matrix(pattern);
  for (unsigned int i = 0; i < 3; ++i)
    matrix.set(i, i, 4.);
  solver.analyze(matrix);
  solver.factorize(matrix);
  Vector<double> rhs(3), solution(3);
  for (unsigned int i = 0; i < 3; ++i)
    rhs[i] = 4. * (i + 1.);
  solver.vmult(solution, rhs);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    for (unsigned int i = 0; i < 3; ++i)
      AssertThrow(std::abs(solution[i] - (i + 1.)) < 1.e-12,
                  ExcInternalError());
}



template <typename VectorType>
void
solve_and_check(SparseDirectMUMPS &solver,
                const IndexSet    &owned,
                const double       value)
{
  VectorType rhs(owned, MPI_COMM_WORLD);
  VectorType solution(owned, MPI_COMM_WORLD);
  // Independent exact solution x=(1,2,3,4).
  for (const auto i : owned)
    rhs[i] = 4. * (i + 1.) + value * ((i + 1) % 4 + 1.);
  rhs.compress(VectorOperation::insert);
  solver.vmult(solution, rhs);
  for (const auto i : owned)
    AssertThrow(std::abs(solution[i] - (i + 1.)) < 1.e-12, ExcInternalError());
}



template <typename MatrixType, typename VectorType>
void
distributed_test()
{
  const auto owned = owned_rows(false);
  MatrixType matrix;
  fill(matrix, owned, 1.);
  SparseDirectMUMPS solver({}, matrix.get_mpi_communicator());
  solver.analyze(matrix);
  for (const double value : {1., 0., -1.})
    {
      for (const auto i : owned)
        matrix.set(i, (i + 1) % 4, value);
      matrix.compress(VectorOperation::insert);
      solver.factorize(matrix);
      solve_and_check<VectorType>(solver, owned, value);
    }

  // Only rank zero's coordinates change: every rank must reject the input.
  MatrixType changed;
  fill(changed, owned, 1., true);
  expect_exception([&]() { solver.factorize(changed); });
  solver.analyze(matrix);
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
    {
      // Also exercise an empty rank during analysis, factorization, and solve.
      MatrixType redistributed;
      fill(redistributed, owned_rows(true), 1.);
      expect_exception([&]() { solver.factorize(redistributed); });
      solver.analyze(redistributed);
      solver.factorize(redistributed);
      solve_and_check<VectorType>(solver, owned_rows(true), 1.);
      expect_exception([&]() { solver.factorize(matrix); });
    }
  // Reanalysis may switch back to a centralized matrix and RHS.
  centralized_test(solver);
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  serial_test();
  {
    SparseDirectMUMPS solver;
    centralized_test(solver);
  }
  distributed_test<PETScWrappers::MPI::SparseMatrix,
                   PETScWrappers::MPI::Vector>();
// The MUMPS interface for Trilinos MPI right-hand sides requires Epetra.
#if defined(DEAL_II_WITH_TRILINOS) && defined(DEAL_II_TRILINOS_WITH_EPETRA)
  distributed_test<TrilinosWrappers::SparseMatrix,
                   TrilinosWrappers::MPI::Vector>();
#endif
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      initlog();
      deallog << "OK" << std::endl;
    }
}
