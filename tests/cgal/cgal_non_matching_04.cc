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

// Create Nitsche mass matrix for differend dimension and check the matrix norm
// to show that it's equal to the measure of the object over which is
// integrating.

#include <deal.II/base/function.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/non_matching/coupling.h>
#include <deal.II/non_matching/quadrature_overlapped_grids.h>

#include "../tests.h"
using namespace dealii;


template <int dim, int spacedim>
void
test()
{
  constexpr int                     degree = 1;
  constexpr double                  radius = .5;
  constexpr double                  left   = -0.44;
  constexpr double                  right  = .3;
  Triangulation<spacedim, spacedim> space_tria;
  Triangulation<dim, spacedim>      embedded_tria;

  GridGenerator::hyper_cube(space_tria, -1., 1.);
  if constexpr (dim == 1 && spacedim == 2)
    {
      GridGenerator::hyper_sphere(embedded_tria, {.2, .2}, radius);
      space_tria.refine_global(2);
      embedded_tria.refine_global(6);
    }
  else if constexpr (dim == 2 && spacedim == 2)
    {
      GridGenerator::hyper_cube(embedded_tria, left, right);
      GridTools::rotate(M_PI_4 / 2., embedded_tria);
      space_tria.refine_global(4);
      embedded_tria.refine_global(2);
    }
  else if constexpr (dim == 2 && spacedim == 3)
    {
      // Buggy case, see CGAL issue #6777
      /*GridTools::rotate(Tensor<1, 3>({1. / sqrt(2.), 1. / sqrt(2.), 0.}),
                        numbers::PI_4,
                        embedded_tria);*/
      // embedded_tria.refine_global(2);
      // space_tria.refine_global(1);
    }
  else if constexpr (dim == 3 && spacedim == 3)
    {
      GridGenerator::hyper_cube(embedded_tria, left, right);
      embedded_tria.refine_global(2);
      GridTools::rotate(Tensor<1, 3>({0, 1, 0}), numbers::PI_4, embedded_tria);
      space_tria.refine_global(2);
    }

  DoFHandler<spacedim>      space_dh(space_tria);
  DoFHandler<dim, spacedim> embedded_dh(embedded_tria);

  FE_Q<spacedim>      fe_space(1);
  FE_Q<dim, spacedim> fe_embedded(1);

  space_dh.distribute_dofs(fe_space);
  embedded_dh.distribute_dofs(fe_embedded);

  auto space_cache =
    std::make_unique<GridTools::Cache<spacedim>>(space_tria); // Q1 mapping
  auto embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
    embedded_tria); // Q1 mapping

  // Compute Quadrature formulas on the intersections of the two
  const auto vec_info = NonMatching::collect_quadratures_on_overlapped_grids(
    *space_cache, *embedded_cache, degree, 1e-14);
  deallog << "Done collecting" << std::endl;

  AffineConstraints<double> space_constraints;


  SparsityPattern        sparsity_pattern;
  DynamicSparsityPattern dsp(space_dh.n_dofs());
  DoFTools::make_sparsity_pattern(space_dh, dsp);
  sparsity_pattern.copy_from(dsp);
  SparseMatrix<double> nitsche_matrix(sparsity_pattern);

  const double h = space_dh.begin_active()->diameter();
  deallog << "h = " << h << '\n';
  NonMatching::
    assemble_nitsche_with_exact_intersections<spacedim, dim, spacedim>(
      space_dh,
      vec_info,
      nitsche_matrix,
      space_constraints,
      ComponentMask(),
      MappingQ1<spacedim>(),
      Functions::ConstantFunction<spacedim>(
        h)); // multiply by h so that "h/penalty" equals 1. Just for testing
             // purposes

  Vector<double> ones(space_dh.n_dofs());
  ones                = 1.0;
  const double result = nitsche_matrix.matrix_norm_square(ones);
  deallog << "Result with Nitsche matrix: " << std::setprecision(10) << result
          << std::endl;

  if constexpr (dim == 1 && spacedim == 2)
    {
      deallog << "Expected : " << std::setprecision(10)
              << 2. * numbers::PI * radius << std::endl;
    }
  else
    {
      deallog << "Expected : " << std::setprecision(10)
              << std::pow(right - left, dim) << std::endl;
    }
}

int
main()
{
  initlog();


  test<1, 2>();
  test<2, 2>();
  // test<2, 3>();
  test<3, 3>();
}
