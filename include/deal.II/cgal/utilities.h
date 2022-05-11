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

#ifndef dealii_cgal_utilities_h
#define dealii_cgal_utilities_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CGAL

#  include <deal.II/base/quadrature_lib.h>

#  include <deal.II/grid/tria.h>

#  include <CGAL/Cartesian.h>
#  include <CGAL/Complex_2_in_triangulation_3.h>
#  include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#  include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#  include <CGAL/Kernel_traits.h>
#  include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#  include <CGAL/Mesh_criteria_3.h>
#  include <CGAL/Mesh_triangulation_3.h>
#  include <CGAL/Polygon_mesh_processing/corefinement.h>
#  include <CGAL/Polygon_mesh_processing/measure.h>
#  include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#  include <CGAL/Polyhedral_mesh_domain_with_features_3.h>
#  include <CGAL/Simple_cartesian.h>
#  include <CGAL/Surface_mesh.h>
#  include <CGAL/Triangulation_3.h>
#  include <CGAL/make_mesh_3.h>
#  include <CGAL/make_surface_mesh.h>
#  include <deal.II/cgal/surface_mesh.h>

#  include <type_traits>



DEAL_II_NAMESPACE_OPEN
/**
 * Interface to the Computational Geometry Algorithm Library (CGAL).
 *
 * CGAL is a software project that provides easy access to efficient and
 * reliable geometric algorithms. The library offers data structures and
 * algorithms like triangulations, Voronoi diagrams, Boolean operations on
 * polygons and polyhedra, point set processing, arrangements of curves, surface
 * and volume mesh generation, geometry processing,  alpha shapes, convex hull
 * algorithms, shape reconstruction, AABB and KD trees...
 *
 * You can learn more about the CGAL library at https://www.cgal.org/
 */
namespace CGALWrappers
{
#  ifdef CGAL_CONCURRENT_MESH_3
  using Concurrency_tag = CGAL::Parallel_tag;
#  else
  using Concurrency_tag = CGAL::Sequential_tag;
#  endif

  enum class BooleanOperation
  {
    corefine     = 1 << 0, ///< Corefine the two surfaces
    difference12 = 1 << 1, ///< Compute the boolean difference of the first
                           ///< input argument minus the second
    difference21 = 1 << 2, ///< Compute the boolean difference of the second
                           ///< input argument minus the first
    intersection = 1 << 3, ///< Compute the intersection of the input arguments
    merge        = 1 << 4, ///< Compute the union of the input arguments
  };

  /**
   * Convert from deal.II Point to any compatible CGAL point.
   *
   * @tparam CGALPointType Any of the CGAL point types
   * @tparam dim Dimension of the point
   * @param [in] p An input deal.II Point<dim>
   * @return CGALPointType A CGAL point
   */
  template <typename CGALPointType, int dim>
  inline CGALPointType
  dealii_point_to_cgal_point(const dealii::Point<dim> &p);

  /**
   * Convert from various CGAL point types to deal.II Point.
   *
   * @tparam dim Dimension of the point
   * @tparam CGALPointType Any of the CGAL point types
   * @param p An input CGAL point type
   * @return dealii::Point<dim> The corresponding deal.II point.
   */
  template <int dim, typename CGALPointType>
  inline dealii::Point<dim>
  cgal_point_to_dealii_point(const CGALPointType &p);

  /**
   * Given a closed CGAL::Surface_mesh, this function fills the
   * internal region bounded by the surface with tets. This should be
   * used to get the coordinates of the (few) tets inside, in order to construct
   * Quadrature rules over each tetrahedron.
   *
   * @param [in] surface_mesh The (closed) surface mesh bounding the volume that has to be filled.
   * @param [out] triangulation The output triangulation filled with tetrahedra.
   */
  template <typename C3t3>
  void
  cgal_surface_mesh_to_cgal_coarse_triangulation(
    CGAL::Surface_mesh<typename C3t3::Point::Point> &surface_mesh,
    C3t3 &                                           triangulation);

  /**
   * Given two triangulated surface meshes, the corefinement operation consists
   * in refining both meshes so that their intersection polylines are a subset
   * of edges in both refined meshes. The corefinement of two triangulated
   * surface meshes can naturally be used for computing Boolean operations on
   * volumes. The last parameter drives the selection of the boolean operation
   * that one wants to perform, and can be `BooleanOperation::corefine`,
   * `BooleanOperation::difference12`, `BooleanOperation::difference21`,
   * `BooleanOperation::intersection`, `BooleanOperation::merge` if one wants to
   * compute the corefinement only, differences, intersection or union of the
   * two meshes, respectively. In case of corefinement only, the operation will
   * be performed directly on the original meshes `surf_1` and `surf_2`. The
   * shaded region in the following picture shows the result of the corefinement
   * and intersection between a cube and the green polyhedra
   *
   * @image html corefine_and_compute_intersection.png
   *
   * See the CGAL documentation for an extended discussion and several examples:
   * https://doc.cgal.org/latest/Polygon_mesh_processing/index.html#title14
   *
   * @tparam CGALPointType
   * @param[in] surf_1 The first surface mesh.
   * @param[in] surf_2 The second surface mesh.
   * @param[out] outsurf The surface mesh with storing the result of the boolean
   * operation. Notice that in case of corefinement only, the corefined meshes
   * will be the first ones.
   * @param bool_op One of BooleanOperation::corefine, BooleanOperation::difference12, BooleanOperation::difference21,
   * BooleanOperation::intersection, BooleanOperation::merge.
   */
  template <typename CGALPointType>
  void
  compute_boolean_operation(CGAL::Surface_mesh<CGALPointType> &surf_1,
                            CGAL::Surface_mesh<CGALPointType> &surf_2,
                            CGAL::Surface_mesh<CGALPointType> &outsurf,
                            const BooleanOperation &           bool_op);

  /**
   * Given a CGAL Triangulation describing a polygonal region, create
   * a Quadrature rule to integrate over the polygon by looping trough all the
   * vertices and exploiting QGaussSimplex.
   *
   * @param[in] tria The CGAL triangulation object describing the polyhedral
   * region.
   * @param[in] degree Desired degree of the Quadrature rule on each element of
   * the polyhedral.
   * @return [out] A global Quadrature rule on the polyhedron.
   */
  template <typename Tr>
  dealii::Quadrature<Tr::Point::Ambient_dimension::value>
  compute_quadrature(const Tr &tria, const unsigned int degree);

  /**
   *
   *
   * @param cell0
   * @param cell1
   * @param mapping0
   * @param mapping1
   * @param degree
   * @return Quadrature<spacedim>
   */
  template <int dim0, int dim1, int spacedim, typename Tr>
  dealii::Quadrature<spacedim>
  compute_quadrature_on_boolean_operation(
    const typename dealii::Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename dealii::Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                                   degree,
    Tr                                                                   tria,
    const Mapping<dim0, spacedim> &mapping0 =
      (dealii::ReferenceCells::get_hypercube<dim0>()
         .template get_default_linear_mapping<dim0, spacedim>()),
    const Mapping<dim1, spacedim> &mapping1 =
      (dealii::ReferenceCells::get_hypercube<dim1>()
         .template get_default_linear_mapping<dim1, spacedim>()));


} // namespace CGALWrappers

#  ifndef DOXYGEN
// Template implementations
namespace CGALWrappers
{
  template <typename CGALPointType, int dim>
  inline CGALPointType
  dealii_point_to_cgal_point(const dealii::Point<dim> &p)
  {
    constexpr int cdim = CGALPointType::Ambient_dimension::value;
    static_assert(dim <= cdim, "Only dim <= cdim supported");
    if constexpr (cdim == 1)
      return CGALPointType(p[0]);
    else if constexpr (cdim == 2)
      return CGALPointType(p[0], dim > 1 ? p[1] : 0);
    else if constexpr (cdim == 3)
      return CGALPointType(p[0], dim > 1 ? p[1] : 0, dim > 2 ? p[2] : 0);
    else
      Assert(false, dealii::ExcNotImplemented());
    return CGALPointType();
  }



  template <int dim, typename CGALPointType>
  inline dealii::Point<dim>
  cgal_point_to_dealii_point(const CGALPointType &p)
  {
    constexpr int cdim = CGALPointType::Ambient_dimension::value;
    if constexpr (dim == 1)
      return dealii::Point<dim>(CGAL::to_double(p.x()));
    else if constexpr (dim == 2)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                cdim > 1 ? CGAL::to_double(p.y()) : 0);
    else if constexpr (dim == 3)
      return dealii::Point<dim>(CGAL::to_double(p.x()),
                                cdim > 1 ? CGAL::to_double(p.y()) : 0,
                                cdim > 2 ? CGAL::to_double(p.z()) : 0);
    else
      Assert(false, dealii::ExcNotImplemented());
  }



  template <typename C3t3>
  void
  cgal_surface_mesh_to_cgal_coarse_triangulation(
    CGAL::Surface_mesh<typename C3t3::Point::Point> &surface_mesh,
    C3t3 &                                           triangulation)
  {
    using CGALPointType = typename C3t3::Point::Point;
    Assert(CGAL::is_closed(surface_mesh),
           ExcMessage("The surface mesh must be closed."));
    std::cout << "Assert passato!" << std::endl;

    using K           = typename CGAL::Kernel_traits<CGALPointType>::Kernel;
    using Mesh_domain = CGAL::Polyhedral_mesh_domain_with_features_3<
      K,
      CGAL::Surface_mesh<CGALPointType>>;
    using Tr = typename CGAL::
      Mesh_triangulation_3<Mesh_domain, CGAL::Default, Concurrency_tag>::type;
    using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;

    if (!CGAL::is_triangle_mesh(surface_mesh))
      CGAL::Polygon_mesh_processing::triangulate_faces(surface_mesh);

    Mesh_domain domain(surface_mesh);
    // domain.detect_features();
    std::cout << "Qui sì" << std::endl;
    Mesh_criteria criteria(CGAL::parameters::facet_size             = 0,
                           CGAL::parameters::facet_distance         = 0,
                           CGAL::parameters::cell_radius_edge_ratio = 2,
                           CGAL::parameters::cell_size              = 0);
    // Mesh generation
    triangulation = CGAL::make_mesh_3<C3t3>(domain,
                                            criteria,
                                            CGAL::parameters::no_perturb(),
                                            CGAL::parameters::no_exude());
    std::cout << "Qui sì(magari...)!" << std::endl;
  }



  template <typename CGALPointType>
  void
  compute_boolean_operation(CGAL::Surface_mesh<CGALPointType> &surf_1,
                            CGAL::Surface_mesh<CGALPointType> &surf_2,
                            CGAL::Surface_mesh<CGALPointType> &outsurf,
                            const BooleanOperation &           bool_op)
  {
    Assert(
      outsurf.is_empty() && CGAL::is_closed(surf_1) && CGAL::is_closed(surf_2),
      ExcMessage(
        "The output surface_mesh must be empty upon calling this function"));
    bool res      = false;
    namespace PMP = CGAL::Polygon_mesh_processing;
    switch (bool_op)
      {
        case BooleanOperation::merge:
          res = PMP::corefine_and_compute_union(surf_1, surf_2, outsurf);
          break;
        case BooleanOperation::intersection:
          res = PMP::corefine_and_compute_intersection(surf_1, surf_2, outsurf);
          break;
        case BooleanOperation::difference12:
          res = PMP::corefine_and_compute_difference(surf_1, surf_2, outsurf);
          break;
        case BooleanOperation::difference21:
          res = PMP::corefine_and_compute_difference(surf_2, surf_1, outsurf);
          break;
        case BooleanOperation::corefine:
          PMP::corefine(
            surf_1,
            surf_2); // both surfaces are corefined, forget about outsurf
          (void)outsurf;
          res = true;
          break;
        default:
          outsurf.clear();
          break;
      }
    Assert(res,
           ExcMessage("The boolean operation was not succesfully computed."));
  }


  template <typename Tr>
  dealii::Quadrature<Tr::Point::Ambient_dimension::value>
  compute_quadrature(const Tr &tria, const unsigned int degree)
  {
    Assert(tria.is_valid(), ExcMessage("The triangulation is not valid."));
    Assert(Tr::Point::Ambient_dimension::value == 3, ExcNotImplemented());
    Assert(degree > 0,
           ExcMessage("The degree of the Quadrature formula is not positive."));
    std::cout << "Asserts passati!" << std::endl;

    constexpr int           spacedim = Tr::Point::Ambient_dimension::value;
    QGaussSimplex<spacedim> quad(degree);
    std::vector<dealii::Point<spacedim>>              pts;
    std::vector<double>                               wts;
    std::array<dealii::Point<spacedim>, spacedim + 1> vertices; // tets
    for (auto it = tria.cells_in_complex_begin();
         it != tria.cells_in_complex_end();
         ++it)
      {
        for (unsigned int i = 0; i < (spacedim + 1); ++i)
          {
            vertices[i] =
              cgal_point_to_dealii_point<spacedim>(it->vertex(i)->point());
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



  template <int dim0, int dim1, int spacedim, typename Tr>
  dealii::Quadrature<spacedim>
  compute_quadrature_on_boolean_operation(
    const typename dealii::Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename dealii::Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const unsigned int                                                   degree,
    Tr                                                                   tria,
    const Mapping<dim0, spacedim> &mapping0,
    const Mapping<dim1, spacedim> &mapping1)
  {
    Assert(dim1 <= dim0,
           ExcMessage("This function can only work if dim1<=dim0"));
    CGAL::Surface_mesh<typename Tr::Point::Point> surface_1, surface_2,
      out_surface;
    std::cout << "Inizio d2c " << std::endl;
    dealii_cell_to_cgal_surface_mesh(cell0, mapping0, surface_1);
    dealii_cell_to_cgal_surface_mesh(cell1, mapping1, surface_2);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_1);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_2);
    std::cout << "Fine d2c " << std::endl;
    Assert(CGAL::is_triangle_mesh(surface_1),
           ExcMessage("Not a TRIANGLE MESH"));
    Assert(CGAL::is_triangle_mesh(surface_2),
           ExcMessage("Not a TRIANGLE MESH"));
    compute_boolean_operation(
      surface_1,
      surface_2,
      out_surface,
      BooleanOperation::intersection); //[TODO: not_only_intersection]
    std::cout
      << "Qui sì e volume: "
      << CGAL::to_double(CGAL::Polygon_mesh_processing::volume(out_surface))
      << CGAL::to_double(CGAL::Polygon_mesh_processing::volume(surface_1))
      << CGAL::to_double(CGAL::Polygon_mesh_processing::volume(surface_2))
      << std::endl;
    cgal_surface_mesh_to_cgal_coarse_triangulation(out_surface, tria);
    std::cout << "Coarse TRIA computed. " << std::endl;
    auto result = compute_quadrature(tria, degree);
    return result;
  }
} // namespace CGALWrappers
#  endif

DEAL_II_NAMESPACE_CLOSE

#endif
#endif
