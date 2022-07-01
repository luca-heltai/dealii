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

#include <deal.II/cgal/intersections.h>

#ifdef DEAL_II_WITH_CGAL

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  namespace internal
  {
    boost::optional<boost::variant<CGALPoint2,
                                   CGALSegment2,
                                   CGALTriangle2,
                                   std::vector<CGALPoint2>>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex)
    {
      CGALPoint2 p1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[0]);
      CGALPoint2 q1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[1]);
      CGALPoint2 r1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[2]);
      CGALTriangle2 triangle1{p1, q1, r1};

      CGALPoint2 p2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[0]);
      CGALPoint2 q2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[1]);
      CGALPoint2 r2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[2]);
      CGALTriangle2 triangle2{p2, q2, r2};

      return CGAL::intersection(triangle1, triangle2);
    }



    boost::optional<boost::variant<CGALPoint2, CGALSegment2>>
    compute_intersection(const std::array<Point<2>, 2> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex)
    {
      CGALPoint2 p1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[0]);
      CGALPoint2 q1 = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[1]);
      CGALSegment2 segm{p1, q1};

      CGALPoint2 p2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[0]);
      CGALPoint2 q2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[1]);
      CGALPoint2 r2 = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[2]);
      CGALTriangle2 triangle{p2, q2, r2};

      return CGAL::intersection(segm, triangle);
    }



    // rectangle-rectangle
    decltype(auto)
    compute_intersection(const std::array<Point<2>, 4> &first_simplex,
                         const std::array<Point<2>, 4> &second_simplex)
    {
      std::array<CGALPoint2, 4> pts1;
      pts1[0] = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[0]);
      pts1[1] = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[1]);
      pts1[2] = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[2]);
      pts1[3] = dealii_point_to_cgal_point<CGALPoint2>(first_simplex[3]);
      const CGALPolygon first{pts1.begin(), pts1.end()};

      std::array<CGALPoint2, 4> pts2;
      pts2[0] = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[0]);
      pts2[1] = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[1]);
      pts2[2] = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[2]);
      pts2[3] = dealii_point_to_cgal_point<CGALPoint2>(second_simplex[3]);
      const CGALPolygon second{pts2.begin(), pts2.end()};

      std::vector<Polygon_with_holes_2> poly_list;

      CGAL::intersection(first, second, std::back_inserter(poly_list));
      return poly_list;
    }

#  if defined(CGAL_GEQ_515)


    // line, tetra
    boost::optional<boost::variant<CGALPoint3, CGALSegment3>>
    compute_intersection(const std::array<Point<3>, 2> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex)
    {
      CGALPoint3 p1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[0]);
      CGALPoint3 q1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[1]);
      CGALSegment3 segm{p1, q1};

      CGALPoint3 p2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[0]);
      CGALPoint3 q2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[1]);
      CGALPoint3 r2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[2]);
      CGALPoint3 s2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[3]);
      CGALTetra  tetra{p2, q2, r2, s2};

      return CGAL::intersection(segm, tetra);
    }



    // triangle, tetra
    boost::optional<boost::variant<CGALPoint3,
                                   CGALSegment3,
                                   CGALTriangle3,
                                   std::vector<CGALPoint3>>>
    compute_intersection(const std::array<Point<3>, 3> &first_simplex,
                         const std::array<Point<3>, 4> &second_simplex)
    {
      CGALPoint3 p1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[0]);
      CGALPoint3 q1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[1]);
      CGALPoint3 r1 = dealii_point_to_cgal_point<CGALPoint3>(first_simplex[2]);
      CGALTriangle3 triangle{p1, q1, r1};

      CGALPoint3 p2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[0]);
      CGALPoint3 q2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[1]);
      CGALPoint3 r2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[2]);
      CGALPoint3 s2 = dealii_point_to_cgal_point<CGALPoint3>(second_simplex[3]);
      CGALTetra  tetra{p2, q2, r2, s2};

      return CGAL::intersection(triangle, tetra);
    }
#  endif
  } // namespace internal



  template <int dim0, int dim1, int spacedim, int N>
  std::vector<std::array<Point<spacedim>, N>>
  compute_intersection_of_cells(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const Mapping<dim0, spacedim> &                              mapping0,
    const Mapping<dim1, spacedim> &                              mapping1,
    const double                                                 tol)
  {
    (void)cell0;
    (void)cell1;
    (void)mapping0;
    (void)mapping1;
    Assert(false, ExcNotImplemented());
  }



  template <>
  std::vector<std::array<Point<2>, 3>>
  compute_intersection_of_cells<2, 2, 2, 3>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<2, 2>::cell_iterator &cell1,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<2, 2> &                              mapping1,
    const double                                       tol)
  {
    const auto              vertices_cell0 = mapping0.get_vertices(cell0);
    const auto              vertices_cell1 = mapping1.get_vertices(cell1);
    std::array<Point<2>, 4> vertices0, vertices1;
    std::copy(vertices_cell0.begin(), vertices_cell0.end(), vertices0.begin());
    std::copy(vertices_cell1.begin(), vertices_cell1.end(), vertices1.begin());

    std::swap(vertices0[2], vertices0[3]);
    std::swap(vertices1[2], vertices1[3]);
    const auto intersection_test =
      internal::compute_intersection(vertices0, vertices1);

    if (!intersection_test.empty())
      {
        const auto &poly = intersection_test[0].outer_boundary();

        const unsigned int size_poly = poly.size();
        if (size_poly == 3)
          {
            // the intersection is a triangle itself, so return directly the
            // vertices.
            return {{{cgal_point_to_dealii_point<2>(poly.vertex(0)),
                      cgal_point_to_dealii_point<2>(poly.vertex(1)),
                      cgal_point_to_dealii_point<2>(poly.vertex(2))}}};
          }
        else if (size_poly >= 4)
          {
            // the intersection is a polygon, you need to triangulate it.
            std::vector<std::array<Point<2>, 3>> collection;

            CDT cdt;
            cdt.insert_constraint(poly.vertices_begin(),
                                  poly.vertices_end(),
                                  true);

            internal::mark_domains(cdt);
            std::array<Point<2>, 3> vertices;

            for (Face_handle f : cdt.finite_face_handles())
              {
                if (f->info().in_domain() &&
                    CGAL::to_double(cdt.triangle(f).area()) > tol)
                  {
                    for (unsigned int i = 0; i < 3; ++i)
                      {
                        vertices[i] = cgal_point_to_dealii_point<2>(
                          cdt.triangle(f).vertex(i));
                      }

                    collection.push_back(vertices);
                  }
              }
            return collection;
          }
        else
          {
            std::cout << "We have " << size_poly << " vertices" << std::endl;
            Assert(false, ExcMessage("The polygon is degenerate."));
            return {};
          }
      }
    else
      {
        Assert(false, ExcMessage("Cells do not intersect."));
        return {};
      }
  }



  template <>
  std::vector<std::array<Point<2>, 2>>
  compute_intersection_of_cells<2, 1, 2, 2>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<1, 2>::cell_iterator &cell1,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<1, 2> &                              mapping1,
    const double                                       tol)
  {
    const auto              vertices_cell0 = mapping0.get_vertices(cell0);
    const auto              vertices_cell1 = mapping1.get_vertices(cell1);
    std::array<Point<2>, 4> vertices0;
    std::array<Point<2>, 2> vertices1;
    std::copy(vertices_cell0.begin(), vertices_cell0.end(), vertices0.begin());
    std::copy(vertices_cell1.begin(), vertices_cell1.end(), vertices1.begin());

    std::swap(vertices0[2], vertices0[3]);

    std::array<CGALPoint2, 4> pts;
    pts[0] = dealii_point_to_cgal_point<CGALPoint2>(vertices0[0]);
    pts[1] = dealii_point_to_cgal_point<CGALPoint2>(vertices0[1]);
    pts[2] = dealii_point_to_cgal_point<CGALPoint2>(vertices0[2]);
    pts[3] = dealii_point_to_cgal_point<CGALPoint2>(vertices0[3]);

    CGALPolygon poly(pts.begin(), pts.end());

    CGALSegment2 segm(dealii_point_to_cgal_point<CGALPoint2>(vertices1[0]),
                      dealii_point_to_cgal_point<CGALPoint2>(vertices1[1]));
    CDT          cdt;
    cdt.insert_constraint(poly.vertices_begin(), poly.vertices_end(), true);
    std::vector<std::array<Point<2>, 2>> vertices;
    internal::mark_domains(cdt);

    for (Face_handle f : cdt.finite_face_handles())
      {
        if (f->info().in_domain() &&
            CGAL::to_double(cdt.triangle(f).area()) > tol)
          {
            const auto intersection = CGAL::intersection(segm, cdt.triangle(f));
            if (const CGALSegment2 *s =
                  boost::get<CGALSegment2>(&*intersection))
              vertices.push_back({{cgal_point_to_dealii_point<2>((*s)[0]),
                                   cgal_point_to_dealii_point<2>((*s)[1])}});
          }
      }
    return vertices;
  }


} // namespace CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
