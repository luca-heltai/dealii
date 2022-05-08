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

#include <deal.II/fe/mapping_q.h> //[TODO: fix mapping stuff]

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
    Assert(
      mesh.is_empty(),
      ExcMessage(
        "The CGAL::Surface_mesh object must be empty upon calling this function."));
    using Mesh           = CGAL::Surface_mesh<CGALPointType>;
    const auto &vertices = mapping.get_vertices(cell);
    std::map<unsigned int, typename Mesh::Vertex_index> deal2cgal;

    // Add all vertices to the mesh
    // Store CGAL ordering
    for (const auto &i : cell->vertex_indices())
      deal2cgal[cell->vertex_index(i)] = mesh.add_vertex(
        CGALWrappers::dealii_point_to_cgal_point<CGALPointType>(vertices[i]));

    // Add faces
    if (dim < 3)
      // simplices and quads are allowable faces for CGAL
      add_facet(cell, deal2cgal, mesh);
    else
      // in 3d, we build a surface mesh containing all the faces of the 3d cell.
      // Simplices, Tetrahedrons, and Pyramids have their faces numbered in the
      // same way as CGAL does (all faces are numbered clockwise). Hexahedrons,
      // instead, have their faces numbered lexicographically, and one cannot
      // deduce the direction of the normals by just looking at the vertices.
      // In order for CGAL to be able to produce the right orientation, we need
      // to revers the order of the vertices for faces with even index.
      for (const auto &f : cell->face_indices())
        add_facet(cell->face(f),
                  deal2cgal,
                  mesh,
                  cell->reference_cell() != ReferenceCells::Hexahedron ||
                    (f % 2 == 0));
  }



  template <int dim1, int dim2, int spacedim>
      Quadrature<spacedim>
  compute_quadrature_rule_over_boolean_operation(
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const typename Triangulation<dim2, spacedim>::cell_iterator &cell2,
    const Mapping<dim1,spacedim>& mapping1,
    const Mapping<dim2,spacedim>& mapping2,
    const BooleanOperation &boolean_operation)
  {
    Assert(dim != 1 || spacedim != 1,
           ExcMessage(
             "This function does not work with 1-dimensional objects."));
    Assert(
      outsm.is_empty(),
      ExcMessage(
        "The output surface mesh needs to be empty upon calling this function."));
    namespace PMP = CGAL::Polygon_mesh_processing;
    CGAL::Surface_mesh<CGALPointType> sm1, sm2, outsm;
    dealii_cell_to_cgal_surface_mesh(cell1,
mapping1    ,                                 sm1); //[TODO: remove Mapping from here...]
    dealii_cell_to_cgal_surface_mesh(cell2, mapping2, sm2);
    PMP::triangulate_faces(sm1);
    PMP::triangulate_faces(sm2);

    [[maybe_unused]] bool res = false;
    // wrap into utility
    switch (boolean_operation)
      {
        case BooleanOperation::UNION:
          res = PMP::corefine_and_compute_union(sm1, sm2, outsm);
          break;
        case BooleanOperation::INTERSECTION:
          res = PMP::corefine_and_compute_intersection(sm1, sm2, outsm);
          break;
        case BooleanOperation::DIFFERENCE:
          res = PMP::corefine_and_compute_difference(sm1, sm2, outsm);
          break;
        case BooleanOperation::NONE:
          PMP::corefine(sm1, sm2);
          (void)outsm;
          res = true;
          break;
        default:
          Assert(
            res,
            ExcMessage(
              "The boolean operation you provided doesn't make sense. Please check it."));
          break;
      }
    Assert(res,
           ExcMessage("The boolean operation was not succesfully computed."));
  }



  template <typename CGALPointType, typename CGALTriangulation>
  Quadrature<CGALPointType::Ambient_dimension::value>
  collect_quadratures_inside_surface(
    const CGAL::Surface_mesh<CGALPointType> &sm,
    const unsigned int                       degree,
    CGALTriangulation &                      tr)
  {
    constexpr unsigned int spacedim = 3;
    Assert(spacedim != 1,
           ExcNotImplemented("1D quadratures are not yet supported."));
    Assert(
      (!sm.is_empty() && tr.dimension() == -1),
      ExcMessage(
        "The input mesh must be non-empty and the triangulation must be empty. Check the call to this function."));
//sm to coarse
    CGAL::Surface_mesh<CGALPointType> dummy;
    CGAL::convex_hull_3(sm.points().begin(), sm.points().end(), dummy);
    tr.insert(dummy.points().begin(), dummy.points().end());

    QGaussSimplex<spacedim>              quad(degree);
    std::vector<dealii::Point<spacedim>> pts;
    std::vector<double>                  wts;
    for (const auto &f : tr.finite_cell_handles())
      {
        std::array<dealii::Point<spacedim>, spacedim + 1> vertices; // tets
        for (unsigned int i = 0; i < (spacedim + 1); ++i)
          {
            vertices[i] =
              cgal_point_to_dealii_point<spacedim>(f->vertex(i)->point());
          }

        auto local_quad = quad.compute_affine_transformation(vertices);
        std::transform(local_quad.get_points().begin(),
                       local_quad.get_points().end(),
                       std::back_inserter(pts),
                       [&pts](const auto &p) { return p; });
        std::transform(local_quad.get_weights().begin(),
                       local_quad.get_weights().end(),
                       std::back_inserter(wts),
                       [&wts](const double w) { return w; });
      }
    return Quadrature<spacedim>(pts, wts);
  }
  // explicit instantiations
#    include "surface_mesh.inst"


} // namespace CGALWrappers
#  endif


DEAL_II_NAMESPACE_CLOSE

#endif
