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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <CGAL/IO/File_medit.h>
#include <CGAL/IO/io.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <deal.II/cgal/utilities.h>

#include "../tests.h"

// Compute volumes of intersections of polyhedra by splitting each intersection
// into simplices.

using K         = CGAL::Exact_predicates_inexact_constructions_kernel;
using CGALPoint = CGAL::Point_3<K>;
using namespace CGALWrappers;
using Mesh_domain =
  CGAL::Polyhedral_mesh_domain_with_features_3<K,
                                               CGAL::Surface_mesh<CGALPoint>>;
using Tr = typename CGAL::
  Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type;

using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;
using C3t3          = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;

void
test()
{
  using namespace ReferenceCells;
  std::vector<std::pair<ReferenceCell, ReferenceCell>> ref_pairs = {
    {Tetrahedron, Pyramid}};

  C3t3          tria;
  constexpr int degree  = 3;
  auto          bool_op = BooleanOperation::intersection;
  for (const auto &pair : ref_pairs)
    {
      const auto       ref_cell0 = pair.first;
      const auto       ref_cell1 = pair.second;
      Triangulation<3> tria0;
      Triangulation<3> tria1;
      GridGenerator::reference_cell(tria0, ref_cell0);
      GridGenerator::reference_cell(tria1, ref_cell1);
      GridTools::rotate(numbers::PI_4, 0, tria0);
      const auto mapping0 = ref_cell0.template get_default_mapping<3>(1);
      const auto mapping1 = ref_cell1.template get_default_mapping<3>(1);
      const auto cell0    = tria0.begin_active();
      const auto cell1    = tria1.begin_active();

      auto test_quad = compute_quadrature_on_boolean_operation<3, 3, 3, C3t3>(
        cell0, cell1, degree, bool_op, *mapping0, *mapping1);
      deallog << "Volume of poly with Quadrature: " << std::setprecision(12)
              << std::accumulate(test_quad.get_weights().begin(),
                                 test_quad.get_weights().end(),
                                 0.)
              << std::endl;
      // CGAL::Surface_mesh<CGALPoint> surface_1, surface_2;
      // dealii_cell_to_cgal_surface_mesh(cell0, *mapping0, surface_1);
      // dealii_cell_to_cgal_surface_mesh(cell1, *mapping1, surface_2);
      // deallog << surface_1 << std::endl;
      // deallog << surface_2 << std::endl;
    }
}

int
main()
{
  initlog();
  test();
}


// #include <CGAL/Complex_2_in_triangulation_3.h>
// #include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
// #include <CGAL/IO/facets_in_complex_2_to_triangle_mesh.h>
// #include <CGAL/IO/output_to_vtu.h>
// #include <CGAL/Mesh_complex_3_in_triangulation_3.h>
// #include <CGAL/Mesh_criteria_3.h>
// #include <CGAL/Mesh_triangulation_3.h>
// #include <CGAL/Polygon_mesh_processing/corefinement.h>
// #include <CGAL/Polygon_mesh_processing/measure.h>
// #include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
// #include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
// #include <CGAL/Surface_mesh.h>
// #include <CGAL/Surface_mesh_default_triangulation_3.h>
// #include <CGAL/Triangulation_3.h>
// #include <CGAL/make_mesh_3.h>
// #include <CGAL/make_surface_mesh.h>

// #include <fstream>
// #include <iostream>

// #include "../tests.h"
// #ifdef CGAL_CONCURRENT_MESH_3
// typedef CGAL::Parallel_tag Concurrency_tag;
// #else
// typedef CGAL::Sequential_tag Concurrency_tag;
// #endif
// typedef CGAL::Exact_predicates_exact_constructions_kernel   K;
// typedef CGAL::Surface_mesh<K::Point_3>                        Mesh;
// typedef CGAL::Polyhedral_mesh_domain_with_features_3<K, Mesh> Mesh_domain;
// typedef CGAL::
//   Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type Tr;
// typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr,
//                                                 Mesh_domain::Corner_index,
//                                                 Mesh_domain::Curve_index>
//   C3t3;



// typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// // To avoid verbose function and named parameters call
// using namespace CGAL::parameters;


// namespace PMP    = CGAL::Polygon_mesh_processing;
// namespace params = CGAL::Polygon_mesh_processing::parameters;
// int
// main(int argc, char *argv[])
// {
//   const std::string filename1 = "input_grids/tetra.off";
//   const std::string filename2 = "input_grids/pyramid.off";
//   Mesh              mesh1, mesh2;


//   std::ifstream input(filename1);
//   input >> mesh1;

//   std::ifstream input2(filename2);
//   input2 >> mesh2;
//   // if (!PMP::IO::read_polygon_mesh(filename1, mesh1) ||
//   //     !PMP::IO::read_polygon_mesh(filename2, mesh2))
//   //   {
//   //     std::cerr << "Invalid input." << std::endl;
//   //     return 1;
//   //   }
//   CGAL::Polygon_mesh_processing::triangulate_faces(mesh1);
//   CGAL::Polygon_mesh_processing::triangulate_faces(mesh2);

//   // CGAL::IO::write_polygon_mesh("tetra_original.stl",
//   //                              mesh1,
//   //                              CGAL::parameters::stream_precision(17));
//   // CGAL::IO::write_polygon_mesh("pyramid_original.stl",
//   //                              mesh2,
//   //                              CGAL::parameters::stream_precision(17));
//   Mesh out_union, out_intersection, new_intersection;
//   std::array<boost::optional<Mesh *>, 4> output;
//   output[PMP::Corefinement::UNION]        = &out_union;
//   output[PMP::Corefinement::INTERSECTION] = &out_intersection;
//   // for the example, we explicit the named parameters, this is identical to
//   // PMP::corefine_and_compute_boolean_operations(mesh1, mesh2, output)
//   std::array<bool, 4> res = PMP::corefine_and_compute_boolean_operations(
//     mesh1,
//     mesh2,
//     output,
//     params::all_default(), // mesh1 named parameters
//     params::all_default(), // mesh2 named parameters
//     std::make_tuple(
//       params::vertex_point_map(
//         get(boost::vertex_point, out_union)), // named parameters for
//         out_union
//       params::vertex_point_map(
//         get(boost::vertex_point,
//             out_intersection)), // named parameters for out_intersection
//       params::all_default(),    // named parameters for mesh1-mesh2 not used
//       params::all_default())    // named parameters for mesh2-mesh1 not used)
//   );
//   if (res[PMP::Corefinement::UNION])
//     {
//       std::cout << "Union was successfully computed\n";
//       // CGAL::IO::write_polygon_mesh("union.stl",
//       //                              out_union,
//       // CGAL::parameters::stream_precision(17));
//     }
//   else
//     std::cout << "Union could not be computed\n";
//   if (res[PMP::Corefinement::INTERSECTION])
//     {
//       std::cout << "Intersection was successfully computed\n";
//       auto test_new_intersction =
//         PMP::corefine_and_compute_intersection(mesh1, mesh2,
//         new_intersection);
//       // CGAL::IO::write_polygon_mesh("intersection.stl",
//       //                              new_intersection,
//       // CGAL::parameters::stream_precision(17));

//       // std::vector<Mesh*> poly_ptrs_vector(1, &out_intersection);
//       Mesh_domain domain(new_intersection);
//       domain.detect_features();

//       // Mesh_criteria criteria(facet_angle=25, facet_size=0.15,
//       // facet_distance=0.008,
//       //                      cell_radius_edge_ratio=3);
//       Mesh_criteria criteria;

//       // Mesh generation
//       C3t3 c3t3;
//       c3t3 =
//         CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude());

//       // Output the facets of the c3t3 to an OFF file. The facets will not be
//       // oriented.
//       // std::ofstream off_file("meshed_test.vtu");

//       std::ofstream off_file_medit("meshed_test.mesh");
//       // CGAL::IO::output_to_vtu(off_file, c3t3);
//       c3t3.output_to_medit(off_file_medit, false);

//       std::cout << "Number of cells:" << c3t3.number_of_cells() << '\n';
//       std::cout << "Number of faces:" << c3t3.number_of_facets() << '\n';
//       std::cout << "Area: " << PMP::volume(new_intersection) << '\n';
//     }

//   else
//     std::cout << "Intersection could not be computed\n";



//   return 0;
// }
