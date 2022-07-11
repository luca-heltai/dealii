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

// Compute the coupling mass matrix <v_i,q_j> and check its correct by computing
// the measure of the embedded grid.

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/non_matching/coupling.h>
#include <deal.II/non_matching/quadrature_overlapped_grids.h>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Surface_mesh/IO.h>
#include <deal.II/cgal/intersections.h>
#include <deal.II/cgal/triangulation.h>

#include "../tests.h"
using namespace dealii;

template <int dim, int spacedim>
void
test()
{
  deallog << "dim: " << dim << "\t"
          << "spacedim: " << spacedim << std::endl;
  constexpr int                degree = 1;
  constexpr double             left   = 0.28;
  constexpr double             right  = 0.69;
  Triangulation<spacedim>      space_tria;
  Triangulation<dim, spacedim> embedded_tria;

  GridGenerator::hyper_cube(space_tria, -1., 1.);
  if constexpr (spacedim == 3)
    {
      GridGenerator::hyper_ball(embedded_tria, {0.5, 0.5, 0.5}, 0.2);
      // GridTools::rotate(numbers::PI_4, 2, embedded_tria);
    }
  else
    {
      GridGenerator::hyper_cube(embedded_tria, left, right);
    }
  space_tria.refine_global(1);
  embedded_tria.refine_global(1);
  if constexpr (dim == 3 && spacedim == 3)
    {
      for (const auto &cell : embedded_tria.active_cell_iterators())
        {
          for (const auto &f : cell->face_indices())

            if (cell->face(f)->at_boundary())
              {
                Point<3> pt =
                  cell->face(f)->get_manifold().get_new_point_on_face(
                    cell->face(f));
              }
        }
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
  const auto vec_info =
    NonMatching::collect_quadratures_on_overlapped_grids(*space_cache,
                                                         *embedded_cache,
                                                         degree);
  double sum   = 0.;
  int    count = 0;
  for (const auto &infos : vec_info)
    {
      const auto &[cell0, cell1, quad_form] = infos;
      deallog << "Idx Cell0: " << cell0->active_cell_index() << std::endl;
      deallog << "Idx Cell1: " << cell1->active_cell_index() << std::endl;
      deallog << "intersection between : " << cell0->active_cell_index()
              << " and " << cell1->active_cell_index() << " has measure: ";

      deallog << std::accumulate(quad_form.get_weights().begin(),
                                 quad_form.get_weights().end(),
                                 0.)
              << std::endl;
      deallog << "It should be: " << cell1->measure() << std::endl;
      if (std::abs(cell1->measure() -
                   std::accumulate(quad_form.get_weights().begin(),
                                   quad_form.get_weights().end(),
                                   0.)) < 1e-12)
        {
          ++count;
        }

      sum += cell1->measure();

      if constexpr (spacedim == 3)
        {
          Surface_mesh sm;
          CGALWrappers::dealii_cell_to_cgal_surface_mesh(
            cell1, embedded_cache->get_mapping(), sm);

          CGAL::Polygon_mesh_processing::triangulate_faces(sm);
          deallog << "With CGAL:" << CGAL::Polygon_mesh_processing::volume(sm)
                  << std::endl;
          // sum += CGAL::Polygon_mesh_processing::volume(sm);
          if (cell1->active_cell_index() == 8)
            {
              std::ofstream filename("test_face.ply");
              CGAL::write_ply(filename, sm);
              for (const auto &v_deal :
                   (embedded_cache->get_mapping()).get_vertices(cell1))
                {
                  deallog << v_deal << std::endl;
                }

              for (const auto &v_cgal : sm.points())
                {
                  deallog << v_cgal << std::endl;
                }
              Triangulation<2, 3> tria;
              CGALWrappers::cgal_surface_mesh_to_dealii_triangulation(sm, tria);


              std::ofstream out("test_to_understand_different_area.vtk");
              GridOut().write_vtk(tria, out);

              Triangulation<3>       tria_q;
              Triangulation3_inexact tr_q;

              tr_q.insert(sm.points().begin(), sm.points().end());
              deallog << "NUMERO DI CELLE:" << tr_q.number_of_finite_cells()
                      << std::endl;
              deallog << "NUMERO DI CELLE INF:" << tr_q.number_of_cells()
                      << std::endl;


              double test = 0.;
              for (const auto &c : tr_q.finite_cell_handles())
                {
                  const auto &tet = tr_q.tetrahedron(c);
                  deallog << tet.is_degenerate() << std::endl;
                  test += std::abs(tet.volume());
                  deallog << "volume una dopo l'altra:"
                          << std::abs(tet.volume()) << std::endl;
                }
              deallog << "TOTALE:" << test << std::endl;

              CGALWrappers::cgal_triangulation_to_dealii_triangulation(tr_q,
                                                                       tria_q);
              std::ofstream out_q("test_to_understand_different_area_quad.vtk");
              GridOut().write_vtk(tria_q, out_q);
              for (const auto &f : cell1->face_indices())
                {
                  // if (cell1->face(f)->at_boundary())
                  //   {
                  deallog
                    << "New point: "
                    << cell1->face(f)->get_manifold().get_new_point_on_face(
                         cell1->face(f))
                    << std::endl;
                  // }
                }
            }
          else if (cell1->active_cell_index() == 64)
            {
              for (const auto &v_deal :
                   (embedded_cache->get_mapping()).get_vertices(cell1))
                {
                  deallog << v_deal << std::endl;
                }

              for (const auto &v_cgal : sm.points())
                {
                  deallog << v_cgal << std::endl;
                }
              Triangulation<2, 3> tria_d;
              CGALWrappers::cgal_surface_mesh_to_dealii_triangulation(sm,
                                                                      tria_d);

              std::ofstream out_d("test_to_understand_same_area.vtk");
              GridOut().write_vtk(tria_d, out_d);


              Triangulation<3>       tria_sq;
              Triangulation3_inexact tr;
              // CGAL::Polygon_mesh_processing::stitch_borders(sm);
              tr.insert(sm.points().begin(), sm.points().end());
              for (const auto &c : tr.finite_cell_handles())
                {
                  const auto &tet = tr.tetrahedron(c);
                  deallog << tet.is_degenerate() << std::endl;
                }
              deallog << "NUMERO DI CELLE:" << tr.number_of_finite_cells()
                      << std::endl;
              deallog << "NUMERO DI CELLE INF:" << tr.number_of_cells()
                      << std::endl;


              CGALWrappers::cgal_triangulation_to_dealii_triangulation(tr,
                                                                       tria_sq);
              std::ofstream out_dq("test_to_understand_same_area_quad.vtk");


              GridOut().write_vtk(tria_sq, out_dq);
            }
          sm.clear();
        }
    }


  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> coupling_matrix(sparsity_pattern);

  AffineConstraints<double> constraints;
  AffineConstraints<double> embedded_constraints;
  DynamicSparsityPattern    dsp(space_dh.n_dofs(), embedded_dh.n_dofs());
  NonMatching::create_coupling_sparsity_pattern_with_exact_intersections(
    vec_info,
    space_dh,
    embedded_dh,
    dsp,
    constraints,
    ComponentMask(),
    ComponentMask(),
    embedded_constraints);

  sparsity_pattern.copy_from(dsp);
  coupling_matrix.reinit(sparsity_pattern);

  NonMatching::create_coupling_mass_matrix_with_exact_intersections(
    space_dh,
    embedded_dh,
    vec_info,
    coupling_matrix,
    constraints,
    ComponentMask(),
    ComponentMask(),
    MappingQ1<spacedim>(),
    embedded_cache->get_mapping(),
    embedded_constraints);

  Vector<double> ones_space(space_dh.n_dofs());
  Vector<double> ones_embedded(embedded_dh.n_dofs());
  ones_space    = 1.0;
  ones_embedded = 1.0;
  const double result =
    coupling_matrix.matrix_scalar_product(ones_space, ones_embedded);
  deallog << "Result with coupling matrix: " << std::setprecision(10) << result
          << std::endl;

  deallog << "Expected : " << std::setprecision(10)
          << GridTools::volume(embedded_tria, embedded_cache->get_mapping())
          << std::endl;


  deallog << "Sum computed: " << sum << std::endl;


  if (dim == 3 && spacedim == 3)
    {
      std::ofstream output_test_space("space_test_non_matching.vtk");
      std::ofstream output_test_embedded("embedded_test_non_matching.vtk");
      GridOut().write_vtk(space_tria, output_test_space);
      GridOut().write_vtk(embedded_tria, output_test_embedded);
      deallog << "Esatte: " << count << std::endl;
    }
}

int
main()
{
  initlog();



  test<1, 2>();
  test<2, 2>();
  test<3, 3>();
}
