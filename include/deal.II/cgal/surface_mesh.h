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

#ifndef dealii_cgal_surface_mesh_h
#define dealii_cgal_surface_mesh_h

#include <deal.II/base/config.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/cgal/utilities.h>

#ifdef DEAL_II_WITH_CGAL
#  include <CGAL/Polygon_mesh_processing/corefinement.h>
#  include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#  include <CGAL/Surface_mesh.h>
#  include <CGAL/Triangulation_3.h>
#  include <CGAL/convex_hull_3.h>


DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  /**
   * The following functions are built on top of
   * [CGAL::Surface_mesh](https://doc.cgal.org/latest/Surface_mesh/index.html)
   * and the `CGAL::PolygonMeshProcessing` namespace. With these objects we can
   * perform boolean operations on deal.II types such as difference, union and
   * intersection and many more. They allow to generate quadrature rules to
   * integrate integrate functions over arbitrarily shaped polygons using
   * `collect_quadratures_inside_surface()`.
   *
   * The shaded region in the following picture is the intersection between a
   * cube and a tetrahedron, converted to a deal.II triangulation
   *
   * @image html intersection_cube_with_tetrahedron.png
   *
   */
  enum class BooleanOperation
  {
    NONE         = 1 << 0,
    UNION        = 1 << 2,
    INTERSECTION = 1 << 3,
    DIFFERENCE   = 1 << 4,
  };

  /**
   * Build a CGAL::Surface_mesh from a deal.II cell.
   *
   * The class Surface_mesh implements a halfedge data structure and can be used
   * to represent polyhedral surfaces. It is an edge-centered data structure
   * capable of maintaining incidence information of vertices, edges, and faces.
   * Each edge is represented by two halfedges with opposite orientation. The
   * orientation of a face is chosen so that the halfedges around a face are
   * oriented counterclockwise.
   *
   * More information on this class is available at
   * https://doc.cgal.org/latest/Surface_mesh/index.html
   *
   * The function will throw an exception in dimension one. In dimension two, it
   * generates a surface mesh of the quadrilateral cell or of the triangle cell,
   * while in dimension three it will generate the surface mesh of the cell,
   * i.e., a polyhedral mesh containing the faces of the input cell.
   *
   * The generated mesh is useful when performing geometric operations using
   * CGAL::Polygon_mesh_processing, i.e., to compute boolean operations on
   * cells, splitting, cutting, slicing, etc.
   *
   * For examples on how to use the resulting  CGAL::Surface_mesh see
   * https://doc.cgal.org/latest/Polygon_mesh_processing/
   *
   * @param[in] cell The input deal.II cell iterator
   * @param[in] mapping The mapping used to map the vertices of the cell
   * @param[out] mesh The output CGAL::Surface_mesh
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  dealii_cell_to_cgal_surface_mesh(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &                                 mesh);

  /**
   * Performs corefinement and boolean operations on two deal.II cells and put
   * the result in a CGAL::Surface_mesh object. If the cells are disjoint and
   * you try to perform a boolean operation which makes no sense, like their
   * intersection, an exception is thrown.
   *
   * Extensive examples can be found in the CGAL documentation at
   * https://doc.cgal.org/latest/Polygon_mesh_processing/index.html#coref_coref_subsec
   *
   * @param[in] cell1 First input cell
   * @param[in] cell2 Second input cell
   * @param[in] boolean_operation BooleanOperation::INTERSECTION and
   * BooleanOperation::UNION and BooleanOperation::DIFFERENCE perform
   * intersection, union and difference of cells, respectively. If not
   * specified, this parameter is defaulted to BooleanOperation::NONE, and only
   * a corefinement is performed.
   * @param[out] outsm Output surface mesh
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  boolean_operation(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell1,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell2,
    CGAL::Surface_mesh<CGALPointType> &                         outsm,
    const BooleanOperation &boolean_operation = BooleanOperation::NONE);

  /**
   * Build a Quadrature<spacedim> formula to integrate over the volumetric
   * region described by a CGAL::Surface_mesh, by filling the interior with
   * simplices and collecting points and weights to integrate over it.
   *
   * @param[in] sm The Surface_mesh over which quadrature formulas are collected
   * @param[in] degree The degree of the quadrature formula
   * @param[out] tr A CGAL triangulation storing the convex_hull of the volume
   * bounded by the Surface_mesh
   * @return Quadrature<spacedim> Collection of Quadratures. The sum of the weights equals to the volume of the polygonal region bounded by sm
   */
  template <typename CGALPointType, typename CGALTriangulation>
  Quadrature<CGALPointType::Ambient_dimension::value>
  collect_quadratures_inside_surface(
    const CGAL::Surface_mesh<CGALPointType> &sm,
    const unsigned int                       degree,
    CGALTriangulation &                      tr);
} // namespace CGALWrappers



DEAL_II_NAMESPACE_CLOSE

#endif
#endif
