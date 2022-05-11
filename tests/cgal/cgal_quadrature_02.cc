// ---------------------------------------------------------------------
//
// Copyright (C) 2022 by the deal.II authors
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

// Intersect reference cells and compute the volume of the intersection by
// integration.

#include <deal.II/base/config.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <CGAL/IO/File_medit.h>
#include <CGAL/IO/io.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <deal.II/cgal/utilities.h>

#include "../tests.h"

// Compute volumes by splitting polyhedra into simplices.

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
using CGALPoint = CGAL::Point_3<K>;
using namespace CGALWrappers;
using Mesh_domain =
  CGAL::Polyhedral_mesh_domain_with_features_3<K,
                                               CGAL::Surface_mesh<CGALPoint>>;
using Tr = typename CGAL::
  Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type;

using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
using C3t3          = CGAL::Mesh_complex_3_in_triangulation_3<Tr,
                                                     Mesh_domain::Corner_index,
                                                     Mesh_domain::Curve_index>;

void
test()
{
  using namespace ReferenceCells;
  std::vector<std::pair<ReferenceCell, ReferenceCell>> ref_pairs = {
    {Tetrahedron, Pyramid}};

  C3t3          tria;
  constexpr int degree = 3;
  for (const auto &pair : ref_pairs)
    {
      const auto ref_cell0 = pair.first;
      const auto ref_cell1 = pair.second;
      // Triangulation<3> tria0;
      // Triangulation<3> tria1;
      // GridGenerator::reference_cell(tria0, ref_cell0);
      // GridGenerator::reference_cell(tria1, ref_cell1);
      Triangulation<3, 3> tria0;
      Triangulation<3, 3> tria1;
      GridGenerator::hyper_cube(tria0, 0.5, 1.5);
      GridGenerator::hyper_cube(tria1, 0., 1.);
      // const auto mapping0 = ref_cell0.template get_default_mapping<3>(1);
      // const auto mapping1 = ref_cell1.template get_default_mapping<3>(1);
      const auto mapping0 = MappingQ<3>(1);
      const auto mapping1 = MappingQ<3>(1);
      const auto cell0    = tria0.begin_active();
      const auto cell1    = tria1.begin_active();

      auto test_quad = compute_quadrature_on_boolean_operation(
        cell0, cell1, degree, tria, mapping0, mapping1);
      deallog << "Volume of poly with Quadrature: " << std::setprecision(12)
              << std::accumulate(test_quad.get_weights().begin(),
                                 test_quad.get_weights().end(),
                                 0.)
              << std::endl;
      tria.clear();
    }
}

int
main()
{
  initlog();
  test();
}
