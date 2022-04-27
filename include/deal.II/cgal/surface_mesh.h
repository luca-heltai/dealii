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

#ifndef dealii_cgal_surface_mesh_h
#define dealii_cgal_surface_mesh_h

#include <deal.II/base/config.h>

#include <deal.II/fe/mapping.h>

#include <deal.II/grid/tria.h>

#include <deal.II/cgal/utilities.h>

#ifdef DEAL_II_WITH_CGAL
#  include <CGAL/Surface_mesh.h>

DEAL_II_NAMESPACE_OPEN

namespace CGALWrappers
{
  /**
   * Build a CGAL::Surface_mesh from a deal.II cell. [TODO: spiega volume per
   * intersezioni]
   *
   * @param[in] cell The input deal.II cell iterator
   * @param[in] mapping The mapping used to map the vertices of the cell
   * @param[out] mesh The output CGAL::Surface_mesh
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(
    const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
    const dealii::Mapping<dim, spacedim> &                              mapping,
    CGAL::Surface_mesh<CGALPointType> &                                 mesh);

  /**
   * Build a CGAL::Surface_mesh from a deal.II cell. [TODO: spiega volume per
   * intersezioni]
   *
   * @param[in] cell The input deal.II cell iterator
   * @param[in] mapping The mapping used to map the vertices of the cell
   * @param[out] mesh The output CGAL::Surface_mesh
   */
  template <typename CGALPointType, int dim, int spacedim>
  void
  to_cgal_mesh(const dealii::Triangulation<dim, spacedim> &cell,
               CGAL::Surface_mesh<CGALPointType>          &mesh);
} // namespace CGALWrappers



DEAL_II_NAMESPACE_CLOSE

#endif
#endif
