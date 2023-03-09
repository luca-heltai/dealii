// ---------------------------------------------------------------------

// Copyright (C) 2018 - 2021 by the deal.II authors

// This file is part of the deal.II library.

// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.

// ---------------------------------------------------------------------


// evaluate jump_in_shape_values(), average_of_shape_values(), shape_value() of
// FEInterfaceValues

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include <deal.II/non_matching/fe_immersed_values.h>

#include <fstream>
#include <iostream>

#include "../tests.h"

template <int dim>
void
make_2_cells(Triangulation<dim> &tria);

template <>
void
make_2_cells<2>(Triangulation<2> &tria)
{
  const unsigned int        dim         = 2;
  std::vector<unsigned int> repetitions = {2, 1};
  Point<dim>                p1;
  Point<dim>                p2(2.0, 1.0);

  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
}

template <>
void
make_2_cells<3>(Triangulation<3> &tria)
{
  const unsigned int        dim         = 3;
  std::vector<unsigned int> repetitions = {2, 1, 1};
  Point<dim>                p1;
  Point<dim>                p2(2.0, 1.0, 1.0);

  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2);
}


template <int dim>
std::pair<std::vector<Point<dim>>, std::vector<double>>
rescale(const typename Triangulation<dim>::active_cell_iterator &cell,
        const MappingQ<dim> &                                    mapping,
        std::vector<Point<dim>> &                                q_points,
        std::vector<double> &                                    JxWs,
        std::vector<Tensor<1, dim>> &                            normals)
{
  const auto &            bbox         = mapping.get_bounding_box(cell);
  const double            bbox_measure = bbox.volume();
  std::vector<Point<dim>> unit_q_points;

  std::transform(q_points.begin(),
                 q_points.end(),
                 std::back_inserter(unit_q_points),
                 [&](const Point<dim> &p) {
                   return mapping.transform_real_to_unit_cell(cell, p);
                 });

  // Weights must be scaled with det(J)*|J^-t n| for each quadrature point.
  // Use the fact that we are using a BBox, so the jacobi entries are the
  // side_length in each direction and normals are already available at this
  // point.
  std::vector<double> scale_factors(q_points.size());
  std::vector<double> scaled_weights(q_points.size());
  Tensor<1, dim>      scale;

  for (unsigned int q = 0; q < q_points.size(); ++q)
    {
      for (unsigned int direction = 0; direction < dim; ++direction)
        {
          scale[direction] =
            normals[q][direction] / (bbox.side_length(direction));
        }

      scaled_weights[q] = JxWs[q] / (bbox_measure * scale.norm());
    }

  std::pair<std::vector<Point<dim>>, std::vector<double>> my_p(unit_q_points,
                                                               scaled_weights);
  return my_p;
}


template <int dim>
void
inspect_fiv(FEInterfaceValues<dim> &fiv)
{
  deallog << "at_boundary(): " << fiv.at_boundary() << "\n"
          << "n_current_interface_dofs(): " << fiv.n_current_interface_dofs()
          << "\n";

  std::vector<types::global_dof_index> indices =
    fiv.get_interface_dof_indices();
  Assert(indices.size() == fiv.n_current_interface_dofs(), ExcInternalError());

  deallog << "interface_dof_indices: ";
  for (auto i : indices)
    deallog << i << ' ';
  deallog << "\n";

  unsigned int idx = 0;
  for (auto v : indices)
    {
      deallog << "  index " << idx << " global_dof_index:" << v << ":\n";

      const auto pair = fiv.interface_dof_to_dof_indices(idx);
      deallog << "    dof indices: " << static_cast<int>(pair[0]) << " | "
              << static_cast<int>(pair[1]) << "\n";

      ++idx;
    }

  deallog << std::endl;
}


template <int dim>
void
test(unsigned int fe_degree)
{
  Triangulation<dim> tria;
  make_2_cells(tria);

  DoFHandler<dim> dofh(tria);
  FE_DGQ<dim>     fe(fe_degree);
  deallog << fe.get_name() << std::endl;
  dofh.distribute_dofs(fe);

  MappingQ<dim> mapping(1);
  UpdateFlags   update_flags = update_values | update_gradients |
                             update_quadrature_points | update_normal_vectors |
                             update_JxW_values;
  FEFaceValues<dim> no_values(mapping,
                              fe,
                              QGauss<dim - 1>(fe_degree + 1),
                              update_flags); // only for quadrature

  auto cell = dofh.begin();

  for (const unsigned int f : GeometryInfo<dim>::face_indices())
    if (!cell->at_boundary(f))
      {
        no_values.reinit(cell, f);
        auto q_points = no_values.get_quadrature_points();
        auto JxWs     = no_values.get_JxW_values();
        auto normals  = no_values.get_normal_vectors();

        NonMatching::FEImmersedSurfaceValues<dim> fe0(
          mapping,
          fe,
          NonMatching::ImmersedSurfaceQuadrature<dim>(q_points, JxWs, normals),
          update_flags);

        no_values.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));
        q_points = no_values.get_quadrature_points();
        JxWs     = no_values.get_JxW_values();
        normals  = no_values.get_normal_vectors();

        NonMatching::FEImmersedSurfaceValues<dim> fe1(
          mapping,
          fe,
          NonMatching::ImmersedSurfaceQuadrature<dim>(q_points, JxWs, normals),
          update_flags);

        fe0.reinit(cell);
        fe1.reinit(cell->neighbor(f));

        FEInterfaceValues<dim> fiv(&fe0, &fe1);
        fiv.reinit(cell,
                   f,
                   numbers::invalid_unsigned_int,
                   cell->neighbor(f),
                   cell->neighbor_of_neighbor(f),
                   numbers::invalid_unsigned_int);

        inspect_fiv(fiv);
        for (const auto &p : fiv.get_quadrature_points())
          deallog << p << std::endl;
        for (const auto &n : fiv.get_normal_vectors())
          deallog << n << std::endl;
        const unsigned int n_dofs = fiv.n_current_interface_dofs();
        Vector<double>     cell_vector(n_dofs);

        q_points = fiv.get_quadrature_points();
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          deallog << "qpoint " << qpoint << ": " << q_points[qpoint]
                  << std::endl;

        cell_vector = 0.0;
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          for (unsigned int i = 0; i < n_dofs; ++i)
            cell_vector(i) +=
              fiv.shape_value(true, i, qpoint) * fiv.get_JxW_values()[qpoint];
        deallog << "shape_value(true): " << cell_vector << std::endl;

        cell_vector = 0.0;
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          for (unsigned int i = 0; i < n_dofs; ++i)
            cell_vector(i) +=
              fiv.shape_value(false, i, qpoint) * fiv.get_JxW_values()[qpoint];
        deallog << "shape_value(false): " << cell_vector << std::endl;

        cell_vector = 0.0;
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          for (unsigned int i = 0; i < n_dofs; ++i)
            cell_vector(i) += fiv.jump_in_shape_values(i, qpoint) *
                              fiv.get_JxW_values()[qpoint];
        deallog << "jump_in_shape_values(): " << cell_vector << std::endl;

        cell_vector = 0.0;
        for (unsigned int qpoint = 0; qpoint < q_points.size(); ++qpoint)
          for (unsigned int i = 0; i < n_dofs; ++i)
            cell_vector(i) += fiv.average_of_shape_values(i, qpoint) *
                              fiv.get_JxW_values()[qpoint];
        deallog << "average_of_shape_values(): " << cell_vector << std::endl;
      }
}



int
main()
{
  initlog();
  test<2>(0);
  test<2>(1);
  test<3>(0);
  test<3>(1);
}