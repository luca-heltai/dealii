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

#include <algorithm>


#ifdef DEAL_II_WITH_CGAL

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  namespace internal
  {
    static std::map<unsigned int, unsigned int> deal_to_cgal_hex{
      {0, 0},
      {1, 1},
      {2, 3},
      {3, 2},
      {4, 6},
      {5, 4},
      {6, 5},
      {7, 7},
    };

    boost::optional<boost::variant<CGALPoint2,
                                   CGALSegment2,
                                   CGALTriangle2,
                                   std::vector<CGALPoint2>>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 3> &second_simplex)
    {
      std::array<CGALPoint2, 3> pts0, pts1;
      std::transform(first_simplex.begin(),
                     first_simplex.end(),
                     pts0.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });

      std::transform(second_simplex.begin(),
                     second_simplex.end(),
                     pts1.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });

      CGALTriangle2 triangle1{pts0[0], pts0[1], pts0[2]};
      CGALTriangle2 triangle2{pts1[0], pts1[1], pts1[2]};
      return CGAL::intersection(triangle1, triangle2);
    }



    boost::optional<boost::variant<CGALPoint2, CGALSegment2>>
    compute_intersection(const std::array<Point<2>, 3> &first_simplex,
                         const std::array<Point<2>, 2> &second_simplex)
    {
      std::array<CGALPoint2, 3> pts0;
      std::array<CGALPoint2, 2> pts1;
      std::transform(first_simplex.begin(),
                     first_simplex.end(),
                     pts0.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });

      std::transform(second_simplex.begin(),
                     second_simplex.end(),
                     pts1.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });

      CGALTriangle2 triangle{pts0[0], pts0[1], pts0[2]};
      CGALSegment2  segm{pts1[0], pts1[1]};
      return CGAL::intersection(segm, triangle);
    }



    // rectangle-rectangle
    decltype(auto)
    compute_intersection(const std::array<Point<2>, 4> &first_simplex,
                         const std::array<Point<2>, 4> &second_simplex)
    {
      std::array<CGALPoint2, 4> pts0, pts1;
      std::transform(first_simplex.begin(),
                     first_simplex.end(),
                     pts0.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });
      std::transform(second_simplex.begin(),
                     second_simplex.end(),
                     pts1.begin(),
                     [&](const Point<2> &p) {
                       return dealii_point_to_cgal_point<CGALPoint2>(p);
                     });
      const CGALPolygon first{pts0.begin(), pts0.end()};
      const CGALPolygon second{pts1.begin(), pts1.end()};

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
      std::array<CGALPoint3, 4> pts0;
      std::array<CGALPoint3, 2> pts1;
      std::transform(first_simplex.begin(),
                     first_simplex.end(),
                     pts0.begin(),
                     [&](const Point<3> &p) {
                       return dealii_point_to_cgal_point<CGALPoint3>(p);
                     });


      std::transform(second_simplex.begin(),
                     second_simplex.end(),
                     pts1.begin(),
                     [&](const Point<3> &p) {
                       return dealii_point_to_cgal_point<CGALPoint3>(p);
                     });

      CGALTetra    tetra{pts0[0], pts0[1], pts0[2], pts0[3]};
      CGALSegment3 segm{pts1[0], pts1[1]};
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
      std::array<CGALPoint3, 4> pts0;
      std::array<CGALPoint3, 3> pts1;
      std::transform(first_simplex.begin(),
                     first_simplex.end(),
                     pts0.begin(),
                     [&](const Point<3> &p) {
                       return dealii_point_to_cgal_point<CGALPoint3>(p);
                     });

      std::transform(second_simplex.begin(),
                     second_simplex.end(),
                     pts1.begin(),
                     [&](const Point<3> &p) {
                       return dealii_point_to_cgal_point<CGALPoint3>(p);
                     });
      CGALTetra     tetra{pts0[0], pts0[1], pts0[2], pts0[3]};
      CGALTriangle3 triangle{pts1[0], pts1[1], pts1[2]};
      return CGAL::intersection(triangle, tetra);
    }
#  endif
  } // namespace internal



  template <int dim0, int dim1, int spacedim>
  std::vector<std::array<Point<spacedim>, dim1 + 1>>
  compute_intersection_of_cells(
    const typename Triangulation<dim0, spacedim>::cell_iterator &cell0,
    const typename Triangulation<dim1, spacedim>::cell_iterator &cell1,
    const Mapping<dim0, spacedim> &                              mapping0,
    const Mapping<dim1, spacedim> &                              mapping1,
    const double                                                 tol)
  {
    Assert((dim1 <= dim0) && (dim0 <= spacedim),
           ExcMessage("Invalid combination of dimension templates."));
    (void)cell0;
    (void)cell1;
    (void)mapping0;
    (void)mapping1;
    Assert(false, ExcNotImplemented());
  }


  // Specialization for quads
  template <>
  std::vector<std::array<Point<2>, 3>>
  compute_intersection_of_cells<2, 2, 2>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<2, 2>::cell_iterator &cell1,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<2, 2> &                              mapping1,
    const double                                       tol)
  {
    std::array<Point<2>, 4> vertices0, vertices1;
    std::copy_n(mapping0.get_vertices(cell0).begin(), 4, vertices0.begin());
    std::copy_n(mapping1.get_vertices(cell1).begin(), 4, vertices1.begin());

    std::swap(vertices0[2], vertices0[3]);
    std::swap(vertices1[2], vertices1[3]);
    const auto intersection_test =
      internal::compute_intersection(vertices0, vertices1);

    if (!intersection_test.empty())
      {
        const auto &       poly      = intersection_test[0].outer_boundary();
        const unsigned int size_poly = poly.size();
        if (size_poly == 3)
          {
            // intersection is a triangle itself, so directly return its
            // vertices.
            return {{{cgal_point_to_dealii_point<2>(poly.vertex(0)),
                      cgal_point_to_dealii_point<2>(poly.vertex(1)),
                      cgal_point_to_dealii_point<2>(poly.vertex(2))}}};
          }
        else if (size_poly >= 4)
          {
            // intersection is a polygon, need to triangulate it.
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



  // Specialization for quad \cap line
  template <>
  std::vector<std::array<Point<2>, 2>>
  compute_intersection_of_cells<2, 1, 2>(
    const typename Triangulation<2, 2>::cell_iterator &cell0,
    const typename Triangulation<1, 2>::cell_iterator &cell1,
    const Mapping<2, 2> &                              mapping0,
    const Mapping<1, 2> &                              mapping1,
    const double                                       tol)
  {
    std::array<Point<2>, 4> vertices0;
    std::array<Point<2>, 2> vertices1;
    std::copy_n(mapping0.get_vertices(cell0).begin(), 4, vertices0.begin());
    std::copy_n(mapping1.get_vertices(cell1).begin(), 2, vertices1.begin());

    std::swap(vertices0[2], vertices0[3]);

    std::array<CGALPoint2, 4> pts;
    std::transform(vertices0.begin(),
                   vertices0.end(),
                   pts.begin(),
                   [&](const Point<2> &p) {
                     return dealii_point_to_cgal_point<CGALPoint2>(p);
                   });

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
            CGAL::to_double(cdt.triangle(f).area()) > tol &&
            CGAL::do_intersect(segm, cdt.triangle(f)))
          {
            const auto intersection = CGAL::intersection(segm, cdt.triangle(f));
            if (const CGALSegment2 *s =
                  boost::get<CGALSegment2>(&*intersection))
              {
                vertices.push_back({{cgal_point_to_dealii_point<2>((*s)[0]),
                                     cgal_point_to_dealii_point<2>((*s)[1])}});
              }
          }
      }
    return vertices;
  }



#  if defined(CGAL_GEQ_515)
  // specialization for hex \cap line
  template <>
  std::vector<std::array<Point<3>, 2>>
  compute_intersection_of_cells<3, 1, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<1, 3>::cell_iterator &cell1,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<1, 3> &                              mapping1,
    const double                                       tol)
  {
    std::array<Point<3>, 8> vertices0; // 8 vertices of the hex
    std::array<Point<3>, 2> vertices1; // 2 endpoints of a segment
    std::copy_n(mapping0.get_vertices(cell0).begin(), 8, vertices0.begin());
    std::copy_n(mapping1.get_vertices(cell1).begin(), 2, vertices1.begin());

    std::array<CGALPoint3, 8> pts;
    std::transform(vertices0.begin(),
                   vertices0.end(),
                   pts.begin(),
                   [&](const Point<3> &p) {
                     return dealii_point_to_cgal_point<CGALPoint3>(p);
                   });

    std::transform(vertices0.begin(),
                   vertices0.end(),
                   pts.begin(),
                   dealii_point_to_cgal_point);

    CGALSegment3 segm(dealii_point_to_cgal_point<CGALPoint3>(vertices1[0]),
                      dealii_point_to_cgal_point<CGALPoint3>(vertices1[1]));

    // Subdivide the hex into tetrahedrons, and intersect each one of them with
    // the line
    std::vector<std::array<Point<3>, 2>> vertices;
    Triangulation3                       tria;
    tria.insert(pts.begin(), pts.end());
    for (const auto &c : tria.finite_cell_handles())
      {
        const auto &tet          = tria.tetrahedron(c);
        const auto  intersection = CGAL::intersection(segm, tet);
        if (const CGALSegment3 *s = boost::get<CGALSegment3>(&*intersection))
          {
            vertices.push_back({{cgal_point_to_dealii_point<3>((*s)[0]),
                                 cgal_point_to_dealii_point<3>((*s)[1])}});
          }
      }

    return vertices;
  }



  template <>
  std::vector<std::array<Point<3>, 3>>
  compute_intersection_of_cells<3, 2, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<2, 3>::cell_iterator &cell1,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<2, 3> &                              mapping1,
    const double                                       tol)
  {
    (void)cell0;
    (void)cell1;
    (void)mapping0;
    (void)mapping1;
    (void)tol;
    return {};
    Assert(false, ExcNotImplemented("2D/3D interesection not yet implemented"));
  }

#  else

  template <>
  std::vector<std::array<Point<3>, 2>>
  compute_intersection_of_cells<3, 1, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<1, 3>::cell_iterator &cell1,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<1, 3> &                              mapping1,
    const double                                       tol)
  {
    (void)cell0;
    (void)cell1;
    (void)mapping0;
    (void)mapping1;
    (void)tol;
    return {};
    Assert(false,
           ExcMessage(
             "This requires a version of CGAL greater or equal to 5.1.5."));
  }



  template <>
  std::vector<std::array<Point<3>, 3>>
  compute_intersection_of_cells<3, 2, 3>(
    const typename Triangulation<3, 3>::cell_iterator &cell0,
    const typename Triangulation<2, 3>::cell_iterator &cell1,
    const Mapping<3, 3> &                              mapping0,
    const Mapping<2, 3> &                              mapping1,
    const double                                       tol)
  {
    (void)cell0;
    (void)cell1;
    (void)mapping0;
    (void)mapping1;
    (void)tol;
    return {};
    Assert(false,
           ExcMessage(
             "This requires a version of CGAL greater or equal to 5.1.5."));
  }
#  endif


} // namespace CGALWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
