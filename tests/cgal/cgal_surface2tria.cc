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

// Convert a deal.II cell to a cgal Surface_mesh

#include <deal.II/base/config.h>

#include <deal.II/base/point.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

#include <CGAL/IO/io.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <deal.II/cgal/surface_mesh.h>

#include <fstream>

#include "../tests.h"


using namespace CGALWrappers;
using CGALPoint = CGAL::Point_3<CGAL::Simple_cartesian<double>>;

template <int dim, int spacedim>
void
test()
{
  using Mesh = CGAL::Surface_mesh<CGALPoint>;
  deallog << "dim= " << dim << ",\t spacedim= " << spacedim << std::endl;
  Mesh          sm;
  std::ifstream input("cross_quad.off");
  input >> sm;

  deallog << sm.num_vertices() << std::endl;
  deallog << sm.num_faces() << std::endl;
  deallog << sm << std::endl;


  Triangulation<dim, spacedim> tria;
  convert_surface_mesh_to_dealii_tria(sm, tria);
  GridOut       go;
  std::ofstream out("test_tria.vtk");
  go.write_vtk(tria, out);
}



void
test_poly()
{
  deallog << "Poly test" << std::endl;
  using Polyhedron = CGAL::Polyhedron_3<CGAL::Simple_cartesian<double>,
                                        CGAL::Polyhedron_items_with_id_3>;
  using Point_3    = CGAL::Simple_cartesian<double>::Point_3;
  Polyhedron    P;
  std::ifstream in1("cross_quad.off");
  in1 >> P;


  Triangulation<2, 3> tria;
  convert_surface_mesh_to_dealii_tria(P, tria);
  GridOut       go;
  std::ofstream output("poly_tria.vtk");
  go.write_vtk(tria, output);

  for (auto face_it = P.facets_begin(); face_it != P.facets_end(); ++face_it)
    {
      auto circ = face_it->facet_begin();
      deallog << "Number of vertices: " << CGAL::circulator_size(circ)
              << std::endl;
      do
        {
          deallog << "Vertex id: " << circ->vertex()->id() << std::endl;
        }
      while (++circ != face_it->facet_begin());
    }
}

int
main()
{
  initlog();
  test<2, 3>();
  test_poly();
}
