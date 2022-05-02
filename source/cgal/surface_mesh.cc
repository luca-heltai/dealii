// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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



#include <deal.II/base/patterns.h>

#include <deal.II/grid/grid_tools.h>

#include <deal.II/cgal/surface_mesh.h>

#ifdef DEAL_II_WITH_CGAL
#  include <boost/hana.hpp>

#  include <CGAL/circulator.h>


DEAL_II_NAMESPACE_OPEN

namespace
{
  template <typename dealiiFace, typename Container, typename CGAL_Mesh>
  void
  add_facet(const dealiiFace &face,
            const Container & deal2cgal,
            CGAL_Mesh &       mesh,
            const bool        clockwise_ordering = true)
  {
    const unsigned                                nv = face->n_vertices();
    std::vector<typename CGAL_Mesh::Vertex_index> indices;

    switch (nv)
      {
        case 2:
          mesh.add_edge(deal2cgal.at(face->vertex_index(0)),
                        deal2cgal.at(face->vertex_index(1)));
          break;
        case 3:
          indices = {deal2cgal.at(face->vertex_index(0)),
                     deal2cgal.at(face->vertex_index(1)),
                     deal2cgal.at(face->vertex_index(2))};
          break;
        case 4:
          indices = {deal2cgal.at(face->vertex_index(0)),
                     deal2cgal.at(face->vertex_index(1)),
                     deal2cgal.at(face->vertex_index(3)),
                     deal2cgal.at(face->vertex_index(2))};
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
    auto f = mesh.null_face();
    if (clockwise_ordering)
      f = mesh.add_face(indices);
    else
      {
        std::reverse(indices.begin(), indices.end());
        f = mesh.add_face(indices);
      }
    Assert(f != mesh.null_face(),
           ExcInternalError("While trying to build a CGAL facet, "
                            "CGAL encountered a orientation problem that it "
                            "was not able to solve."));
  }

  template <typename dealiiFace, typename CGAL_Mesh>
  void
  map_vertices(
    const dealiiFace &                                        cell,
    std::map<unsigned int, typename CGAL_Mesh::Vertex_index> &deal2cgal,
    CGAL_Mesh &                                               mesh)
  {
    for (const auto i : cell->vertex_indices())
      {
        deal2cgal[cell->vertex_index(i)] = mesh.add_vertex(
          CGALWrappers::to_cgal<typename CGAL_Mesh::Point>(cell->vertex(i)));
      }
  }
} // namespace



#  ifndef DOXYGEN
// Template implementations
namespace CGALWrappers
{
  template <typename CGALPointType, int dim, int spacedim>
  void
  convert_to_cgal_surface_mesh(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &                         mesh)
  {
    Assert(dim > 1, ExcImpossibleInDim(dim));
    Assert(
      mesh.is_empty(),
      ExcMessage(
        "The CGAL::Surface_mesh object must be empty upon calling this function."));
    using Mesh           = CGAL::Surface_mesh<CGALPointType>;
    using Vertex_index   = typename Mesh::Vertex_index;
    const auto &vertices = mapping.get_vertices(cell);

    std::map<unsigned int, Vertex_index> deal2cgal;
    // Add all vertices to the mesh
    // Store CGAL ordering
    for (const auto &i : cell->vertex_indices())
      deal2cgal[cell->vertex_index(i)] =
        mesh.add_vertex(CGALWrappers::to_cgal<CGALPointType>(vertices[i]));

    // Add faces
    if (dim < 3)
      // simplices and quads
      add_facet(cell, deal2cgal, mesh);
    else
      // faces of 3d cells
      for (const auto &f : cell->face_indices())
        add_facet(cell->face(f),
                  deal2cgal,
                  mesh,
                  (f % 2 == 0 || cell->n_vertices() != 8));
  }



  template <typename CGALPointType, int dim, int spacedim>
  void
  convert_to_cgal_surface_mesh(const dealii::Triangulation<dim, spacedim> &tria,
                               CGAL::Surface_mesh<CGALPointType>          &mesh)
  {
    Assert(tria.n_cells() > 0, ExcMessage("Triangulation cannot be empty"));
    Assert(dim > 1, ExcImpossibleInDim(dim));
    using Mesh         = CGAL::Surface_mesh<CGALPointType>;
    using Vertex_index = typename Mesh::Vertex_index;

    std::map<unsigned int, Vertex_index> deal2cgal;
    if constexpr (dim == 2)
      {
        for (const auto &cell : tria.active_cell_iterators())
          {
            // to_cgal_mesh(cell, mapping, mesh);
            map_vertices(cell, deal2cgal, mesh);
            add_facet(cell, deal2cgal, mesh);
          }
      }
    else if constexpr (dim == 3 && spacedim == 3)
      {
        for (const auto &cell : tria.active_cell_iterators())
          {
            const auto &face_indices = cell->face_indices();
            for (const auto f : face_indices)
              {
                if (cell->face(f)->at_boundary())
                  {
                    map_vertices(cell->face(f), deal2cgal, mesh);
                    add_facet(cell->face(f),
                              deal2cgal,
                              mesh,
                              (f % 2 == 0 || cell->n_vertices() != 8));
                  }
              }
          }
      }
    CGAL::Polygon_mesh_processing::stitch_borders(mesh);
  }



  template <typename CGAL_MeshType, int dim, int spacedim>
  void
  convert_surface_mesh_to_dealii_tria(CGAL_MeshType                &cgal_mesh,
                                      Triangulation<dim, spacedim> &tria)
  {
    Assert(tria.n_cells() == 0,
           ExcMessage(
             "Triangulation must be empty upon calling this function."));
    Assert((dim == 2 && spacedim == 2) || (dim == 2 && spacedim == 3),
           ExcMessage(
             "This function does work only for dim==2 and spacedim==2 or 3."));


    [[maybe_unused]] auto is_surface_mesh =
      boost::hana::is_valid([](auto &&obj) -> decltype(obj.faces()) {});

    [[maybe_unused]] auto is_polyhedral =
      boost::hana::is_valid([](auto &&obj) -> decltype(obj.facets_begin()) {});

    // Collect Vertices
    [[maybe_unused]] std::size_t         n_cgal_vertices;
    std::vector<dealii::Point<spacedim>> vertices;
    std::vector<CellData<dim>>           cells;
    if constexpr (is_surface_mesh(cgal_mesh))
      {
        n_cgal_vertices = cgal_mesh.num_vertices();
        vertices.reserve(n_cgal_vertices);
        for (const auto &p : cgal_mesh.points())
          {
            vertices.emplace_back(CGALWrappers::to_dealii<spacedim>(p));
          }
      }
    else if constexpr (is_polyhedral(cgal_mesh))
      {
        n_cgal_vertices = cgal_mesh.size_of_vertices();
        vertices.reserve(n_cgal_vertices);
        for (auto it = cgal_mesh.points_begin(); it != cgal_mesh.points_end();
             ++it)
          {
            vertices.emplace_back(CGALWrappers::to_dealii<spacedim>(*it));
          }
      }


    // Different loops depending for Polyhedron or Surface_mesh types
    if constexpr (is_surface_mesh(cgal_mesh))
      {
        const unsigned int vertices_per_face =
          CGAL::vertices_around_face(
            cgal_mesh.halfedge(*(cgal_mesh.faces().begin())), cgal_mesh)
            .size();

        // Collect CellData
        for (const auto &face : cgal_mesh.faces())
          {
            CellData<dim> c(vertices_per_face);
            auto          it_vertex = c.vertices.begin();
            for (const auto v :
                 CGAL::vertices_around_face(cgal_mesh.halfedge(face),
                                            cgal_mesh))
              {
                *(it_vertex++) = v;
              }

            if (vertices_per_face == 4)
              std::swap(c.vertices[3], c.vertices[2]);

            // If vertices_per_face==3, CGAL ordering 0-1-2 is already correct
            cells.emplace_back(c);
          }

        SubCellData subcelldata;
        GridTools::delete_unused_vertices(vertices, cells, subcelldata);
        GridTools::consistently_order_cells(cells);
        tria.create_triangulation(vertices, cells, {});
      }
    else if constexpr (is_polyhedral(cgal_mesh))
      {
        const unsigned int vertices_per_face =
          cgal_mesh.facets_begin()->facet_degree(); //[TODO: fix this?]

        std::size_t i = 0;
        for (auto vertex_it = cgal_mesh.vertices_begin();
             vertex_it != cgal_mesh.vertices_end();
             ++vertex_it)
          {
            vertex_it->id() = i++;
          }
        // Loop over faces of Polyhedron, fill CellData
        for (auto face_it = cgal_mesh.facets_begin();
             face_it != cgal_mesh.facets_end();
             ++face_it)
          {
            CellData<dim> c(vertices_per_face);
            auto          it   = c.vertices.begin();
            auto          circ = face_it->facet_begin();
            do
              {
                *(it++) = circ->vertex()->id();
              }
            while (++circ != face_it->facet_begin());

            if (vertices_per_face == 4)
              std::swap(c.vertices[3], c.vertices[2]);

            cells.emplace_back(c);
          }
        SubCellData subcelldata;
        if (vertices_per_face == 4)
          {
            GridTools::delete_unused_vertices(vertices, cells, subcelldata);
            GridTools::consistently_order_cells(cells);
            tria.create_triangulation(vertices, cells, {});
          }
      }
    else
      {
        Assert(false, ExcInternalError());
      }
  }
#    include "surface_mesh.inst"
} // namespace CGALWrappers
#  endif


DEAL_II_NAMESPACE_CLOSE

#endif
