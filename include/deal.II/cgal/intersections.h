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

#ifndef dealii_cgal_intersections_h
#define dealii_cgal_intersections_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <deal.II/cgal/point_conversion.h>

#ifdef DEAL_II_WITH_CGAL
#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/fe/mapping.h>

#  include <deal.II/grid/tria.h>

#  include <CGAL/Boolean_set_operations_2.h>
#  include <CGAL/Cartesian.h>
#  include <CGAL/Circular_kernel_intersections.h>
#  include <CGAL/Constrained_Delaunay_triangulation_2.h>
#  include <CGAL/Delaunay_mesh_face_base_2.h>
#  include <CGAL/Delaunay_mesh_size_criteria_2.h>
#  include <CGAL/Delaunay_mesher_2.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel_with_sqrt.h>
#  include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#  include <CGAL/Kernel_traits.h>
#  include <CGAL/Polygon_2.h>
#  include <CGAL/Polygon_with_holes_2.h>
#  include <CGAL/Segment_3.h>
#  include <CGAL/Simple_cartesian.h>
#  include <CGAL/Tetrahedron_3.h>
#  include <CGAL/Triangle_2.h>
#  include <CGAL/Triangle_3.h>
#  include <CGAL/Triangulation_2.h>
#  include <CGAL/Triangulation_3.h>
#  include <CGAL/Triangulation_face_base_with_id_2.h>
#  include <CGAL/Triangulation_face_base_with_info_2.h>

//#  include <CGAL/tetrahedral_remeshing.h> REQUIRES CGAL_VERSION>=5.1.5

#  include <fstream>
#  include <type_traits>


// using K           = CGAL::Exact_predicates_inexact_constructions_kernel;
using K           = CGAL::Exact_predicates_exact_constructions_kernel_with_sqrt;
using CGALPolygon = CGAL::Polygon_2<K>;
using Polygon_with_holes_2 = CGAL::Polygon_with_holes_2<K>;
using CGALTriangle2        = K::Triangle_2;
using CGALTriangle3        = K::Triangle_3;
using CGALPoint2           = K::Point_2;
using CGALPoint3           = K::Point_3;
using CGALSegment2         = K::Segment_2;
using CGALSegment3         = K::Segment_3;
using CGALTetra            = K::Tetrahedron_3;
using Triangulation2       = CGAL::Triangulation_2<K>;
using Triangulation3       = CGAL::Triangulation_3<K>;

struct FaceInfo2
{
  FaceInfo2()
  {}
  int nesting_level;
  bool
  in_domain()
  {
    return nesting_level % 2 == 1;
  }
};

using Vb            = CGAL::Triangulation_vertex_base_2<K>;
using Fbb           = CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K>;
using CFb           = CGAL::Constrained_triangulation_face_base_2<K, Fbb>;
using Fb            = CGAL::Delaunay_mesh_face_base_2<K, CFb>;
using Tds           = CGAL::Triangulation_data_structure_2<Vb, Fb>;
using Itag          = CGAL::Exact_predicates_tag;
using CDT           = CGAL::Constrained_Delaunay_triangulation_2<K, Tds, Itag>;
using Criteria      = CGAL::Delaunay_mesh_size_criteria_2<CDT>;
using Vertex_handle = CDT::Vertex_handle;
using Face_handle   = CDT::Face_handle;

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  namespace internal
  {
    // Collection of utilities that compute intersection between simplices
    // identified by array of points. The return type is the one of
    // CGAL::intersection(), i.e. a boost::optional<boost::variant<>>.
    // Intersection between 2D and 3D objects and 1D/3D objects are available
    // only with CGAL versions greater or equal than 5.1.5, hence the
    // corresponding functions are guarded by #ifdef directives. All the
    // signatures follow the convection that the first entity has an intrinsic
    // dimension higher than the second one.

    // triangle,triangle
    boost::optional<boost::variant<CGALPoint2,
                                   CGALSegment2,
                                   CGALTriangle2,
                                   std::vector<CGALPoint2>>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex);

    // line,triangle
    boost::optional<boost::variant<CGALPoint2, CGALSegment2>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 2> &second_simplex);

    // quad,quad
    decltype(auto)
    compute_intersection(const std::array<Point<2>, 4> &first_simplex,
                         const std::array<Point<2>, 4> &second_simplex);
    // quad, line
    decltype(auto)
    compute_intersection(const std::array<Point<2>, 4> &first_simplex,
                         const std::array<Point<2>, 2> &second_simplex);

#  if defined(CGAL_GEQ_515)
    // tetra, line
    boost::optional<boost::variant<CGALPoint3, CGALSegment3>>
    compute_intersection(const std::array<Point<3>, 4> &first_simplex,
                         const std::array<Point<3>, 2> &second_simplex);

    // tetra, triangle
    boost::optional<boost::variant<CGALPoint3,
                                   CGALSegment3,
                                   CGALTriangle3,
                                   std::vector<CGALPoint3>>>
    compute_intersection(const std::array<Point<3>, 4> &first_simplex,
                         const std::array<Point<3>, 3> &second_simplex);
#  endif
  } // namespace internal

  /**
   * Given two deal.II cells, compute the intersection and return a vector of
   * simplices, each one identified by an array of deal.II Points. Each array
   * identify a simplex, and all the simplices together are a subdivision of the
   * intersection. If cells are non-affine, a geometrical error will be
   * necessarily introduced.
   *
   *
   * @param cell0 Iterator to the first cell
   * @param cell1 Iterator to the second cell
   * @param mapping0 Mapping for the first cell
   * @param mapping1 Mapping for the second cell
   * @param tol
   * @return std::vector<std::array<Point<spacedim>, N>>
   */
  template <int dim0, int dim1, int spacedim>
  std::vector<std::array<Point<spacedim>, dim1 + 1>>
  compute_intersection_of_cells(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const Mapping<dim0, spacedim>                               &mapping0,
    const Mapping<dim1, spacedim>                               &mapping1,
    const double                                                 tol = 1e-9);

} // namespace CGALWrappers


DEAL_II_NAMESPACE_CLOSE

#endif
#endif
