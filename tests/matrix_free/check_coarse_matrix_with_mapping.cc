// ---------------------------------------------------------------------
//
// Copyright (C) 2013 - 2020 by the deal.II authors
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


// correctness of matrix-free initialization with multigrid, and MappingFEField
// on deformed grids (here we use a grid rotated by 90 degree and stretched)
// along one direction

#include <deal.II/base/function_parser.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "../tests.h"

#include "/var/lib/nova/dealii/dealii/tests/matrix_free/get_functions_common.h"

template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> tria(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(3);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  dof.distribute_mg_dofs();

  AffineConstraints<double> constraints;
  constraints.close();

  deallog << "Testing " << dof.get_fe().get_name() << std::endl;
  // use this for info on problem
  // std::cout << "Number of cells: " <<
  // dof.get_triangulation().n_active_cells()
  //          << std::endl;
  // std::cout << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;
  // std::cout << "Number of constraints: " << constraints.n_constraints() <<
  // std::endl;

  using VectorType = LinearAlgebra::distributed::Vector<double>;

  FESystem<dim>   fe_sys(dof.get_fe(), dim);
  DoFHandler<dim> deformation_dofs(dof.get_triangulation());
  deformation_dofs.distribute_dofs(fe_sys);
  deformation_dofs.distribute_mg_dofs();

  FunctionParser<dim> domain_configuration(dim == 2 ? "y;-.5*x" : "y;-.5*x;z");
  VectorType          domain_mapping_fine(deformation_dofs.n_dofs());
  VectorTools::interpolate(deformation_dofs,
                           domain_configuration,
                           domain_mapping_fine);

  MatrixFree<dim, double> mf_data;

  MGConstrainedDoFs                 mapping_mg_constrained_dofs;
  MGTransferMatrixFree<dim, double> mapping_transfer;
  MGLevelObject<VectorType>         domain_mapping_levels;

  mapping_mg_constrained_dofs.initialize(deformation_dofs);

  mapping_transfer.initialize_constraints(mapping_mg_constrained_dofs);

  mapping_transfer.build(deformation_dofs);
  domain_mapping_levels.resize(0, tria.n_global_levels() - 1);

  mapping_transfer.interpolate_to_mg(
    deformation_dofs, // deformation_solver->u_dofs(),
    domain_mapping_levels,
    domain_mapping_fine);

  MappingFEField<dim, dim, VectorType, DoFHandler<dim>> domain_mapping(
    deformation_dofs, domain_mapping_levels);

  for (unsigned int i = 0; i < tria.n_global_levels(); ++i)
    {
      // As a solution vector, still use serial vectors.
      Vector<double> solution(dof.n_dofs(i));
      // create vector with random entries
      for (unsigned int j = 0; j < dof.n_dofs(i); ++j)
        {
          const double entry = random_value<double>();
          solution(j)        = entry;
        }


      {
        const QGauss<1>                                  quad(fe_degree + 1);
        typename MatrixFree<dim, double>::AdditionalData data;
        data.tasks_parallel_scheme =
          MatrixFree<dim, double>::AdditionalData::none;
        data.mapping_update_flags = update_gradients;
        if (i == 0)
          data.initialize_indices = true;
        else
          data.initialize_indices = false;
        data.mg_level = i;
        mf_data.reinit(domain_mapping, dof, constraints, quad, data);
      }

      MatrixFreeTestNoHessians<dim, fe_degree, fe_degree + 1> mf(
        mf_data, domain_mapping);
      mf.test_functions(solution);
    }
}
