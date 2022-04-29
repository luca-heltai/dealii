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
#include <deal.II/grid/tria.h>

#include <CGAL/IO/io.h>
#include <deal.II/cgal/surface_mesh.h>

#include "../tests.h"


using namespace CGALWrappers;
using CGALPoint = CGAL::Point_3<CGAL::Simple_cartesian<double>>;

template <int dim, int spacedim>
void
test()
{
  using Mesh = CGAL::Surface_mesh<CGALPoint>;
  deallog << "dim= " << dim << ",\t spacedim= " << spacedim << std::endl;
  std::vector<std::vector<unsigned int>> d2t = {{}, {2}, {3, 4}, {4, 8}};
  for (const auto nv : d2t[dim])
    {
      Triangulation<dim, spacedim> tria;
      Mesh                         mesh;
      const auto ref     = ReferenceCell::n_vertices_to_type(dim, nv);
      const auto mapping = ref.template get_default_mapping<dim, spacedim>(1);
      GridGenerator::reference_cell(tria, ref);
      tria.refine_global(1);

      deallog << "N_original_vertices: " << tria.n_used_vertices()
              << "and n_cells: " << tria.n_active_cells() << std::endl;
      for (const auto &v : tria.get_vertices())
        deallog << v << std::endl;
      if constexpr (dim == 3 && spacedim == 3)
        {
          int i = 0;
          for (const auto &cell : tria.active_cell_iterators())
            {
              const auto face_indices = cell->face_indices();
              for (const auto f : face_indices)
                {
                  if (cell->face(f)->at_boundary())
                    {
                      deallog << "Boundary face " << ++i << '\n';
                    }
                }
            }
        }
      to_cgal_mesh(tria, mesh);

      deallog << mesh << std::endl;
      Assert(mesh.is_valid(), dealii::ExcMessage("The CGAL mesh is not valid"));
    }
}

int
main()
{
  initlog();
  test<2, 2>();
  test<2, 3>();
  test<3, 3>();

  // using Mesh = CGAL::Surface_mesh<CGALPoint>;
  // Mesh                mesh;
  // Triangulation<3, 3> tria;
  // // {  GridGenerator::reference_cell(tria, ReferenceCells::Pyramid);
  // //   tria.refine_global(1);}
  // // GridGenerator::hyper_ball(tria, {0., 0., 0.}); // radius = 1
  // GridGenerator::hyper_L(tria);
  // tria.refine_global(1);
  // deallog << "Cells: " << tria.n_active_cells() << std::endl;
  // to_cgal_mesh(tria, MappingQ<3, 3>(1), mesh);
  // deallog << mesh << std::endl;
}
