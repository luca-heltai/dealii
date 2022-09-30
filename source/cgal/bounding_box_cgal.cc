// ---------------------------------------------------------------------
//
// Copyright (C) 1999 - 2022 by the deal.II authors
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

#include <deal.II/cgal/bounding_box_cgal.h>

DEAL_II_NAMESPACE_OPEN


template <typename Kernel>
class Custom_traits_BBox : public Kernel
{
public:
  Custom_traits_BBox()
  {}
  /// The field number type
  using FT = typename Kernel::FT;

  /// The affine transformation type
  using Aff_transformation_3 = typename CGAL::Aff_transformation_3<Kernel>;

  // Matrix type
  using Matrix = FullMatrix<FT>;

  // Vector type
  // using Vector = Vector<FT>;

public:
  static Matrix
  get_Q(const Matrix &m)
  {
    QR<Vector<FT>> qr;
    Vector<FT>     v0(3);
    v0[0] = m(0, 0);
    v0[1] = m(1, 0);
    v0[2] = m(2, 0);
    Vector<FT> v1(3);
    v1[0] = m(0, 1);
    v1[1] = m(1, 1);
    v1[2] = m(2, 1);
    Vector<FT> v2(3);
    v2[0]                      = m(0, 2);
    v2[1]                      = m(1, 2);
    v2[2]                      = m(2, 2);
    [[maybe_unused]] bool   c0 = qr.append_column(v0);
    [[maybe_unused]] bool   c1 = qr.append_column(v1);
    [[maybe_unused]] bool   c2 = qr.append_column(v2);
    std::vector<Vector<FT>> Q(3);
    Matrix                  Q_matrix(3, 3);
    for (unsigned int j = 0; j < 3; ++j)
      {
        Vector<FT> x(3);
        x    = 0;
        x[j] = 1.;
        Q[j].reinit(3);
        qr.multiply_with_Q(Q[j], x);
      }

    for (unsigned int i = 0; i < 3; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            Q_matrix(i, j) = Q[i][j];
          }
      }
    return Q_matrix;
  }
};



template <int spacedim, typename Number>
OptimalBoundingBox<spacedim, Number>::OptimalBoundingBox(
  const std::vector<Point<spacedim, Number>> &points)
{
#if DEAL_II_CGAL_VERSION_GTE(5, 1, 5)
  Assert(points.size() > Utilities::pow(2, spacedim),
         ExcMessage("Invalid number of points."));
  Assert(spacedim == 3, ExcNotImplemented("Not implemented in 1D and 2D."));
  using K          = CGAL::Exact_predicates_inexact_constructions_kernel;
  using CGALPoint3 = K::Point_3;

  std::vector<CGALPoint3>                             cgal_pts(points.size());
  std::array<CGALPoint3, Utilities::pow(2, spacedim)> cgal_out_pts;
  std::transform(points.begin(),
                 points.end(),
                 cgal_pts.begin(),
                 [&](const Point<spacedim, Number> &p) {
                   return CGALWrappers::dealii_point_to_cgal_point<CGALPoint3>(
                     p);
                 });
  Custom_traits_BBox<K> custom_traits;
  CGAL::oriented_bounding_box(cgal_pts,
                              cgal_out_pts,
                              CGAL::parameters::geom_traits(custom_traits));
  std::transform(cgal_out_pts.begin(),
                 cgal_out_pts.end(),
                 extreme_pts.begin(),
                 [&](const CGALPoint3 &p) {
                   return CGALWrappers::cgal_point_to_dealii_point<spacedim>(p);
                 });


#else
  Assert(false, ExcNeedsCGAL());
#endif
}



template class OptimalBoundingBox<3, double>;
// template class OptimalBoundingBox<2, double>;
// template class OptimalBoundingBox<1, double>;

DEAL_II_NAMESPACE_CLOSE