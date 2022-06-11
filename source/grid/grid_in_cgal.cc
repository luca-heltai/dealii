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

#include <deal.II/base/ndarray.h>

#include <deal.II/grid/grid_in.h>

#ifdef DEAL_II_WITH_CGAL
// Functions needed by the CGAL mesh generation utilities are inside
#  include <deal.II/cgal/triangulation.h>
#endif


DEAL_II_NAMESPACE_OPEN

template <>
void
GridIn<3, 3>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data)
{
#ifdef DEAL_II_WITH_CGAL
  // AssertThrow(dim == 3 && spacedim == 3,
  //             ExcMessage(
  //               "This function can be used for 3D mesh-generation only."));
  // Domain
  using K             = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Mesh_domain   = CGAL::Labeled_mesh_domain_3<K>;
  using Tr            = CGAL::Mesh_triangulation_3<Mesh_domain>::type;
  using C3t3          = CGAL::Mesh_complex_3_in_triangulation_3<Tr>;
  using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;

  CGAL::Image_3 image;
  AssertThrow(image.read(filename.c_str()), ExcMessage("Cannot read file."));

  Mesh_domain domain =
    Mesh_domain::create_labeled_image_mesh_domain(image, 2.9f, 0.f);
  // Mesh criteria
  Mesh_criteria criteria(CGAL::parameters::facet_angle    = data.facet_angle,
                         CGAL::parameters::facet_size     = data.facet_size,
                         CGAL::parameters::facet_distance = data.facet_distance,
                         CGAL::parameters::cell_radius_edge_ratio =
                           data.cell_radius_edge_ratio,
                         CGAL::parameters::cell_size = data.cell_size);
  C3t3          c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria);

  CGALWrappers::cgal_triangulation_to_dealii_triangulation(c3t3, *tria);
#else
  (void)filename;
  (void)data;
  AssertThrow(false, ExcMessage("This function needs CGAL to be installed."));
#endif
}



template <int dim, int spacedim>
void
GridIn<dim, spacedim>::read_inr(const std::string &                    filename,
                                const CGALWrappers::AdditionalData<3> &data)
{
  (void)filename;
  (void)data;
  AssertThrow(false,
              ExcNotImplemented(
                "This function is not supposed to work in 1D or 2D."));
}

template void
GridIn<1, 1>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data);

template void
GridIn<1, 2>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data);

template void
GridIn<1, 3>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data);

template void
GridIn<2, 2>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data);

template void
GridIn<2, 3>::read_inr(const std::string &                    filename,
                       const CGALWrappers::AdditionalData<3> &data);

DEAL_II_NAMESPACE_CLOSE
