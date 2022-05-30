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

#include <deal.II/base/config.h>

#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orientation.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <deal.II/cgal/surface_mesh.h>

#ifdef DEAL_II_WITH_CGAL

DEAL_II_NAMESPACE_OPEN

namespace
{
  template <typename dealiiFace, typename CGAL_Mesh>
  void
  add_facet(
    const dealiiFace &                                              face,
    const std::map<unsigned int, typename CGAL_Mesh::Vertex_index> &deal2cgal,
    CGAL_Mesh &                                                     mesh,
    const bool clockwise_ordering = true)
  {
    const auto reference_cell_type = face->reference_cell();
    std::vector<typename CGAL_Mesh::Vertex_index> indices;

    switch (reference_cell_type)
      {
        case ReferenceCells::Line:
          mesh.add_edge(deal2cgal.at(face->vertex_index(0)),
                        deal2cgal.at(face->vertex_index(1)));
          break;
        case ReferenceCells::Triangle:
          indices = {deal2cgal.at(face->vertex_index(0)),
                     deal2cgal.at(face->vertex_index(1)),
                     deal2cgal.at(face->vertex_index(2))};
          break;
        case ReferenceCells::Quadrilateral:
          indices = {deal2cgal.at(face->vertex_index(0)),
                     deal2cgal.at(face->vertex_index(1)),
                     deal2cgal.at(face->vertex_index(3)),
                     deal2cgal.at(face->vertex_index(2))};
          break;
        default:
          Assert(false, ExcInternalError());
          break;
      }
    if (clockwise_ordering == true)
      std::reverse(indices.begin(), indices.end());

    [[maybe_unused]] const auto new_face = mesh.add_face(indices);
    Assert(new_face != mesh.null_face(),
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
          CGALWrappers::dealii_point_to_cgal_point<typename CGAL_Mesh::Point>(
            cell->vertex(i)));
      }
  }
} // namespace



#  ifndef DOXYGEN
// Template implementations
namespace CGALWrappers
{
  template <typename CGALPointType, int dim, int spacedim>
  void
  dealii_cell_to_cgal_surface_mesh(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &                         mesh)
  {
    Assert(dim > 1, ExcImpossibleInDim(dim));
    using Mesh           = CGAL::Surface_mesh<CGALPointType>;
    const auto &vertices = mapping.get_vertices(cell);
    std::map<unsigned int, typename Mesh::Vertex_index> deal2cgal;

    // Needed for PolygonSoup
    using K  = CGAL::Exact_predicates_inexact_constructions_kernel;
    using FT = K::FT;

    // Add all vertices to the mesh
    // Store CGAL ordering
    // Add faces
    if (dim < 3)
      {
        // simplices and quads are allowable faces for CGAL
        for (const auto &i : cell->vertex_indices())
          {
            deal2cgal[cell->vertex_index(i)] = mesh.add_vertex(
              CGALWrappers::dealii_point_to_cgal_point<CGALPointType>(
                vertices[i]));
          }
        add_facet(cell, deal2cgal, mesh);
      }
    else
      {
        std::vector<std::array<FT, 3>> points(vertices.size());

        for (unsigned int i = 0; i < vertices.size(); ++i)
          {
            points[i] = CGAL::make_array<FT>(vertices[i][0],
                                             vertices[i][1],
                                             vertices[i][2]);
          }

        typedef std::array<FT, 3> Custom_point;

        struct Array_traits
        {
          struct Equal_3
          {
            bool
            operator()(const Custom_point &p, const Custom_point &q) const
            {
              return (p == q);
            }
          };
          struct Less_xyz_3
          {
            bool
            operator()(const Custom_point &p, const Custom_point &q) const
            {
              return std::lexicographical_compare(p.begin(),
                                                  p.end(),
                                                  q.begin(),
                                                  q.end());
            }
          };
          Equal_3
          equal_3_object() const
          {
            return Equal_3();
          }
          Less_xyz_3
          less_xyz_3_object() const
          {
            return Less_xyz_3();
          }
        };

        std::vector<std::vector<unsigned int>> polygons;
        for (const auto &f : cell->face_indices())
          {
            const auto reference_cell_type = cell->face(f)->reference_cell();
            std::vector<unsigned int> indices;
            switch (reference_cell_type)
              {
                case ReferenceCells::Triangle:
                  indices = {cell->face(f)->vertex_index(0),
                             cell->face(f)->vertex_index(1),
                             cell->face(f)->vertex_index(2)};
                  // std::transform(indices.begin(),
                  //                indices.end(),
                  //                std::back_inserter(indices),
                  //                [&indices](unsigned int i) -> unsigned int {
                  //                  return std::modulus<unsigned int>()(i, 4);
                  //                });
                  break;
                case ReferenceCells::Quadrilateral:
                  indices = {cell->face(f)->vertex_index(0),
                             cell->face(f)->vertex_index(1),
                             cell->face(f)->vertex_index(3),
                             cell->face(f)->vertex_index(2)};


                  // for (unsigned int i = 0; i < indices.size(); ++i)
                  //   {
                  //     indices[i] = std::modulus<unsigned int>()(i, 8);
                  //   }
                  break;
                default:
                  Assert(false, ExcInternalError());
                  break;
              }
            polygons.push_back(indices);
            // std::cout << "Qui pure per faccia=" << f << std::endl;
            // for (const auto v : indices)
            //   {
            //     std::cout << v << std::endl;
            //   }
            indices.clear();
          }
        // for (const auto &p : points)
        //   std::cout << p << std::endl;

        // in 3d, we build a surface mesh containing all the faces of the 3d
        // cell. Simplices, Tetrahedrons, and Pyramids have their faces numbered
        // in the same way as CGAL does (all faces are numbered clockwise).
        // Hexahedrons, instead, have their faces numbered lexicographically,
        // and one cannot deduce the direction of the normals by just looking at
        // the vertices. In order for CGAL to be able to produce the right
        // orientation, we need to revers the order of the vertices for faces
        // with even index.
        // std::vector<CGAL::Surface_mesh<CGALPointType>> meshes(
        //   cell->reference_cell().n_faces());
        // unsigned int counter_faces = 0;
        // for (const auto &f : cell->face_indices())
        //   {
        //     add_facet(cell->face(f),
        //               deal2cgal,
        //               mesh,
        //               cell->reference_cell() != ReferenceCells::Hexahedron ||
        //                 (f % 2 == 0));
        //     // ++counter_faces;

        CGAL::Polygon_mesh_processing::repair_polygon_soup(
          points, polygons, CGAL::parameters::geom_traits(Array_traits()));
        std::cout << "Ha riparato" << std::endl;
        [[maybe_unused]] bool orienta =
          CGAL::Polygon_mesh_processing::orient_polygon_soup(points, polygons);

        std::cout << "Ha orientato" << std::endl;
        CGAL::Polygon_mesh_processing::polygon_soup_to_polygon_mesh(points,
                                                                    polygons,
                                                                    mesh);
        CGAL::Polygon_mesh_processing::reverse_face_orientations(mesh);
        Assert(orienta, ExcMessage("Polygon soup didn't work!"));
      }
  }



  template <typename CGALPointType, int dim, int spacedim>
  void
  dealii_tria_to_cgal_surface_mesh(
    const dealii::Triangulation<dim, spacedim> &tria,
    CGAL::Surface_mesh<CGALPointType> &         mesh)
  {
    Assert(tria.n_cells() > 0,
           ExcMessage(
             "Triangulation cannot be empty upon calling this function."));
    Assert(mesh.is_empty(),
           ExcMessage(
             "The surface mesh must be empty upon calling this function."));

    Assert(dim > 1, ExcImpossibleInDim(dim));
    using Mesh         = CGAL::Surface_mesh<CGALPointType>;
    using Vertex_index = typename Mesh::Vertex_index;

    std::map<unsigned int, Vertex_index> deal2cgal;
    if constexpr (dim == 2)
      {
        for (const auto &cell : tria.active_cell_iterators())
          {
            map_vertices(cell, deal2cgal, mesh);
            add_facet(cell, deal2cgal, mesh);
          }
      }
    else if constexpr (dim == 3 && spacedim == 3)
      {
        for (const auto &cell : tria.active_cell_iterators())
          {
            for (const auto &f : cell->face_indices())

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
    else
      {
        Assert(false, ExcImpossibleInDimSpacedim(dim, spacedim));
      }
  } // explicit instantiations
#    include "surface_mesh.inst"

} // namespace CGALWrappers
#  endif


DEAL_II_NAMESPACE_CLOSE

#endif
