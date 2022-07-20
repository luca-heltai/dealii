// ---------------------------------------------------------------------

// Copyright (C) 2022 by the deal.II authors

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

// Compute intersection of simplices in 2D, and return a vector of arrays where
// you can build Quadrature rules. Then check that the sum of weights give the
// correct area for each region.

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/cgal/intersections.h>

#include "../tests.h"

static QGaussSimplex<2> qgauss(1);
void
test_intersection(Triangulation<3> &tria0, Triangulation<2, 3> &tria1)
{
  GridGenerator::hyper_cube(tria0, -1., 1.);
  GridGenerator::hyper_cube(tria1, -0.4, 0.2);
  GridTools::rotate(Tensor<1, 3>({1 / std::sqrt(2), 1 / std::sqrt(2), 0}),
                    numbers::PI_4,
                    tria1);
  std::ofstream out("CUBAZZO.vtk");
  GridOut().write_vtk(tria0, out);
  std::ofstream out2("QUADRATAZZO.vtk");
  GridOut().write_vtk(tria1, out2);
  double expected_measure = 0.25 * 0.25;
  auto   cell0            = tria0.begin_active();
  auto   cell1            = tria1.begin_active();
  cell0->vertex(0)        = Point<3>(-1, -1, -1);
  cell0->vertex(1)        = Point<3>(0, -1, -1);
  cell0->vertex(2)        = Point<3>(-1, 0, -1);
  cell0->vertex(3)        = Point<3>(0, 0, -1);
  cell0->vertex(4)        = Point<3>(-1, -1, 0);
  cell0->vertex(5)        = Point<3>(0, -1, 0);
  cell0->vertex(6)        = Point<3>(-1, 0, 0);
  cell0->vertex(7)        = Point<3>(0, 0, 0);



  cell1->vertex(0) = Point<3>(-0.124185, -0.385815, -0.185000);
  cell1->vertex(1) = Point<3>(0.191630, -0.331630, -0.370000);
  cell1->vertex(2) = Point<3>(-0.0700000, -0.0700000, 0.00000);
  cell1->vertex(3) = Point<3>(0.245815, -0.0158148, -0.1850001);
  const auto vec_of_arrays =
    CGALWrappers::compute_intersection_of_cells<3, 2, 3>(cell0,
                                                         cell1,
                                                         MappingQ1<3>(),
                                                         MappingQ1<2, 3>());
  for (const auto &verts : vec_of_arrays)
    {
      for (const auto &p : verts)
        deallog << p << std::endl;
    }

  const auto   quad = qgauss.mapped_quadrature(vec_of_arrays);
  const double sum =
    std::accumulate(quad.get_weights().begin(), quad.get_weights().end(), 0.);
  // assert(std::abs(sum - expected_measure) < 1e-15);
  deallog << "OK con area: " << sum << std::endl;

  {
    deallog << "DEBUGGING:" << std::endl;
    MappingQ1<3>            mapping0;
    MappingQ1<2, 3>         mapping1;
    std::array<Point<3>, 8> vertices0; // 8 vertices of the hex
    std::array<Point<3>, 4> vertices1; // 4 vertices of the quad
    std::copy_n(mapping0.get_vertices(cell0).begin(), 8, vertices0.begin());
    std::copy_n(mapping1.get_vertices(cell1).begin(), 4, vertices1.begin());

    std::array<CGALPoint3_exact, 8> pts_hex;
    std::array<CGALPoint3_exact, 4> pts_quad;
    std::transform(
      vertices0.begin(),
      vertices0.end(),
      pts_hex.begin(),
      [&](const Point<3> &p) {
        return CGALWrappers::dealii_point_to_cgal_point<CGALPoint3_exact>(p);
      });

    std::transform(
      vertices1.begin(),
      vertices1.end(),
      pts_quad.begin(),
      [&](const Point<3> &p) {
        return CGALWrappers::dealii_point_to_cgal_point<CGALPoint3_exact>(p);
      });

    // Subdivide hex into tetrahedrons
    std::vector<std::array<Point<3>, 3>> vertices;
    Triangulation3_exact                 tria;
    tria.insert(pts_hex.begin(), pts_hex.end());

    // Subdivide quad into triangles
    Delaunay tria_quad(pts_quad.begin(), pts_quad.end());
    double   somma_aree_cgal = 0.;
    for (const auto &c : tria.finite_cell_handles())
      {
        const auto &tet = tria.tetrahedron(c);
        // Check for intersection with each triangle dividing the quad:
        // for (Face_handle f : cdt.finite_face_handles())
        //   {
        for (const auto &f : tria_quad.finite_face_handles())
          {
            if ((CGAL::to_double(tria_quad.triangle(f).squared_area()) >
                 1e-12) &&
                (CGAL::do_intersect(tet, tria_quad.triangle(f))))
              {
                // somma_aree_cgal += std::sqrt(
                //   CGAL::to_double(tria_quad.triangle(f).squared_area()));


                for (unsigned int i = 0; i < 3; ++i)
                  {
                    deallog << (tria_quad.triangle(f))[i] << std::endl;
                  }
                const auto intersection =
                  CGAL::intersection(tet, tria_quad.triangle(f));

                if (const std::vector<CGALPoint3_exact> *vps =
                      boost::get<std::vector<CGALPoint3_exact>>(&*intersection))
                  {
                    deallog << "Vettore di punti: " << vps->size() << std::endl;
                    Delaunay tria_inter(vps->begin(), vps->end());
                    for (const auto &f : tria_inter.finite_face_handles())
                      {
                        auto tria_inside = tria_inter.triangle(f);
                        vertices.push_back(
                          {{CGALWrappers::cgal_point_to_dealii_point<3>(
                              (tria_inside)[0]),
                            CGALWrappers::cgal_point_to_dealii_point<3>(
                              (tria_inside)[1]),
                            CGALWrappers::cgal_point_to_dealii_point<3>(
                              (tria_inside)[2])}});
                      }
                  }

                if (const CGALTriangle3_exact *t =
                      boost::get<CGALTriangle3_exact>(&*intersection))
                  {
                    deallog << "Qua con il triangolo: " << std::endl;
                    vertices.push_back(
                      {{CGALWrappers::cgal_point_to_dealii_point<3>((*t)[0]),
                        CGALWrappers::cgal_point_to_dealii_point<3>((*t)[1]),
                        CGALWrappers::cgal_point_to_dealii_point<3>((*t)[2])}});
                    somma_aree_cgal +=
                      std::sqrt(CGAL::to_double(t->squared_area()));
                  }
              }
          }
        deallog << "AREA SOMMANDO I TRIANGOLI " << somma_aree_cgal << std::endl;
      }

    auto my_other_quad = QGaussSimplex<2>(1).mapped_quadrature(vertices);
    deallog << "Area ora: "
            << std::accumulate(my_other_quad.get_weights().begin(),
                               my_other_quad.get_weights().end(),
                               0.)
            << std::endl;
  }
}


//
int
main()
{
  initlog();
  Triangulation<3>    tria0;
  Triangulation<2, 3> tria1;

  test_intersection(tria0, tria1);
}
