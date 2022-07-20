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


template <int dim0, int dim1, int spacedim>
std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                       typename Triangulation<dim1, spacedim>::cell_iterator,
                       Quadrature<spacedim>>>
collect_quadratures_on_overlapped_grids_test(
  const GridTools::Cache<dim0, spacedim> &space_cache,
  const GridTools::Cache<dim1, spacedim> &immersed_cache,
  const unsigned int                      degree,
  const double                            tol)
{
  AssertThrow(
    dim1 <= dim0,
    ExcMessage(
      "Intrinsic dimension of the immersed object must be smaller than dim0."));
  AssertThrow(degree > 0, ExcMessage("Invalid quadrature degree."));
  Assert((dim1 <= dim0) && (dim0 <= spacedim),
         ExcMessage("This function can only work if dim1<=dim0<=spacedim"));
  std::vector<std::tuple<typename Triangulation<dim0, spacedim>::cell_iterator,
                         typename Triangulation<dim1, spacedim>::cell_iterator,
                         Quadrature<spacedim>>>
    cells_with_quadratures;

  const auto &space_tree =
    space_cache.get_locally_owned_cell_bounding_boxes_rtree();

  // The immersed tree *must* contain all cells, also the non-locally owned
  // ones.
  const auto &immersed_tree = immersed_cache.get_cell_bounding_boxes_rtree();

  // references to triangulations' info (cp cstrs marked as delete)
  const auto &mapping0 = space_cache.get_mapping();
  const auto &mapping1 = immersed_cache.get_mapping();
  namespace bgi        = boost::geometry::index;
  // Whenever the BB space_cell intersects the BB of an embedded cell,
  // store the space_cell in the set of intersected_cells
  double sum = 0;
  for (const auto &[immersed_box, immersed_cell] : immersed_tree)
    {
      deallog << std::endl;
      for (const auto &[space_box, space_cell] :
           space_tree | bgi::adaptors::queried(bgi::intersects(immersed_box)))
        {
          const auto test_intersection =
            NonMatching::compute_quadrature_on_intersection(
              space_cell, immersed_cell, degree, mapping0, mapping1);

          const auto  &weights = test_intersection.get_weights();
          const double area =
            std::accumulate(weights.begin(), weights.end(), 0.0);
          sum += area;
          // if (area > tol) // non-trivial intersection
          //   {
          cells_with_quadratures.push_back(
            std::make_tuple(space_cell, immersed_cell, test_intersection));
          // }

          deallog << "Con Hexa: " << space_cell->active_cell_index()
                  << std::endl;
          deallog << space_cell->vertex(0) << std::endl;
          deallog << space_cell->vertex(1) << std::endl;
          deallog << space_cell->vertex(2) << std::endl;
          deallog << space_cell->vertex(3) << std::endl;
          deallog << space_cell->vertex(4) << std::endl;
          deallog << space_cell->vertex(5) << std::endl;
          deallog << space_cell->vertex(6) << std::endl;
          deallog << space_cell->vertex(7) << std::endl;

          deallog << "Ha area: " << area << std::endl;
        }


      deallog << "Square" << immersed_cell->active_cell_index() << std::endl;
      deallog << immersed_cell->vertex(0) << std::endl;
      deallog << immersed_cell->vertex(1) << std::endl;
      deallog << immersed_cell->vertex(2) << std::endl;
      deallog << immersed_cell->vertex(3) << std::endl;
      deallog << "Sum: " << sum << std::endl;
      sum = 0.;
    }
  return cells_with_quadratures;
}

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
      GridGenerator::hyper_cube(embedded_tria, left, right);
      GridTools::rotate(Tensor<1, 3>({1. / sqrt(2.), 1. / sqrt(2.), 0.}),
                        numbers::PI_4,
                        embedded_tria);
      embedded_tria.refine_global(1);
      space_tria.refine_global(1);

      std::ofstream output_test_space("space_test_quad.vtk");
      std::ofstream output_test_embedded("embedded_test_quad.vtk");
      GridOut().write_vtk(space_tria, output_test_space);
      GridOut().write_vtk(embedded_tria, output_test_embedded);
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

  {
    std::string codim;
    (spacedim - dim == 0) ? codim = "codim_0" : codim = "codim_1";
    std::ofstream output_test_space("space_test_" + codim + ".vtk");
    std::ofstream output_test_embedded("embedded_test_" + codim + ".vtk");
    GridOut().write_vtk(space_tria, output_test_space);
    GridOut().write_vtk(embedded_tria, output_test_embedded);
  }
  auto space_cache =
    std::make_unique<GridTools::Cache<spacedim>>(space_tria); // Q1 mapping
  auto embedded_cache = std::make_unique<GridTools::Cache<dim, spacedim>>(
    embedded_tria); // Q1 mapping

  // Compute Quadrature formulas on the intersections of the two
  const auto vec_info = collect_quadratures_on_overlapped_grids_test(
    *space_cache, *embedded_cache, degree, 1e-14);
  deallog << "Finito di collezionare" << std::endl;
  if constexpr (dim == 2 && spacedim == 3)
    {
      double sum = 0.;
      for (const auto &p : vec_info)
        {
          auto first_cell  = std::get<0>(p);
          auto second_cell = std::get<1>(p);
          auto quad        = std::get<2>(p);
          deallog << "space" << first_cell->active_cell_index() << std::endl;
          deallog << "immersed" << second_cell->active_cell_index()
                  << std::endl;

          sum += std::accumulate(quad.get_weights().begin(),
                                 quad.get_weights().end(),
                                 0.);
        }

      deallog << "Sommando: " << sum << std::endl;
    }

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
        h)); // multiply by h so that "h/penalty" equals 1.Just for testing
             // purposes

  Vector<double> ones(space_dh.n_dofs());
  ones                = 1.0;
  const double result = nitsche_matrix.matrix_norm_square(ones);
  deallog << "Result with Nitsche matrix: " << std::setprecision(10) << result
          << std::endl;
  if constexpr (dim == 1 && spacedim == 2)
    {
      deallog << "Expected : " << std::setprecision(10)
              << GridTools::volume(embedded_tria) << std::endl;
    }
  else if constexpr (dim == 2 && spacedim == 2)
    {
      deallog << "Expected : " << std::setprecision(10)
              << std::pow(right - left, 2) << std::endl;
    }
  else if constexpr (dim == 2 && spacedim == 3)
    {
      deallog << "Expected : " << std::setprecision(10)
              << std::pow(right - left, 2) << std::endl;
    }
  else if constexpr (dim == 3 && spacedim == 3)
    {
      deallog << "Expected : " << std::setprecision(10)
              << std::pow(right - left, 3) << std::endl;
    }
}

int
main()
{
  initlog();


  // test<1, 2>();
  // test<2, 2>();
  test<2, 3>();
  // test<3, 3>();
}
