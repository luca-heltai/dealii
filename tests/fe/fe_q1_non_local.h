// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2018 by the deal.II authors
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

#ifndef dealii_fe_enriched_global_h
#define dealii_fe_enriched_global_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>

#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>

#include <map>
#include <numeric>
#include <utility>
#include <vector>

DEAL_II_NAMESPACE_OPEN


template <int dim, int spacedim = dim>
class FE_Q1_NonLocal : public FiniteElement<dim, spacedim>
{
public:
  FE_Q1_NonLocal(const Triangulation<dim, spacedim> &tria);

  virtual std::unique_ptr<FiniteElement<dim, spacedim>>
  clone() const override;

  virtual UpdateFlags
  requires_update_flags(const UpdateFlags update_flags) const override;

  virtual std::string
  get_name() const override;

  virtual types::global_dof_index
  n_non_local_dofs() const override;

  virtual IndexSet
  get_non_local_dofs_on_cell(const unsigned int) const override;

protected:
  const Triangulation<dim, spacedim> &tria;

  virtual std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_data(
    const UpdateFlags             flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim> &       quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_face_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim - 1> &   quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual std::unique_ptr<
    typename FiniteElement<dim, spacedim>::InternalDataBase>
  get_subface_data(
    const UpdateFlags             update_flags,
    const Mapping<dim, spacedim> &mapping,
    const Quadrature<dim - 1> &   quadrature,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void
  fill_fe_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const CellSimilarity::Similarity                            cell_similarity,
    const Quadrature<dim> &                                     quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void
  fill_fe_face_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;

  virtual void
  fill_fe_subface_values(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const unsigned int                                          face_no,
    const unsigned int                                          sub_no,
    const Quadrature<dim - 1> &                                 quadrature,
    const Mapping<dim, spacedim> &                              mapping,
    const typename Mapping<dim, spacedim>::InternalDataBase &mapping_internal,
    const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                       spacedim>
      &                                                            mapping_data,
    const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
    dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                       spacedim>
      &output_data) const override;
};

//////========================= template functions =======================
namespace internal
{
  template <int dim>
  std::vector<unsigned int>
  get_dpo();

  template <>
  std::vector<unsigned int>
  get_dpo<1>()
  {
    return {{0, 0, 2}};
  };

  template <>
  std::vector<unsigned int>
  get_dpo<2>()
  {
    return {{0, 0, 0, 4}};
  };

  template <>
  std::vector<unsigned int>
  get_dpo<3>()
  {
    return {{0, 0, 0, 0, 8}};
  };

} // namespace internal

template <int dim, int spacedim>
FE_Q1_NonLocal<dim, spacedim>::FE_Q1_NonLocal(
  const Triangulation<dim, spacedim> &tria)
  : FiniteElement<dim, spacedim>(
      FiniteElementData<dim>(internal::get_dpo<dim>(), 1, 1),
      {{false}},
      {{ComponentMask(1, true)}})
  , tria(tria)
{}



template <int dim, int spacedim>
types::global_dof_index
FE_Q1_NonLocal<dim, spacedim>::n_non_local_dofs() const
{
  return tria.n_vertices();
}



template <int dim, int spacedim>
IndexSet
FE_Q1_NonLocal<dim, spacedim>::get_non_local_dofs_on_cell(
  const unsigned int needed_index) const
{
  IndexSet     res(tria.n_vertices());
  unsigned int index = 0;
  for (auto it = tria.begin_active(); it != tria.end(); ++it, ++index)
    if (index == needed_index)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          res.add_index(it->vertex_index(v));

        return res;
      }

  Assert(false, ExcInternalError());
  return res;
}



template <int dim, int spacedim>
std::unique_ptr<FiniteElement<dim, spacedim>>
FE_Q1_NonLocal<dim, spacedim>::clone() const
{
  return std::unique_ptr<FE_Q1_NonLocal<dim, spacedim>>(
    new FE_Q1_NonLocal<dim, spacedim>(tria));
}


template <int dim, int spacedim>
UpdateFlags
FE_Q1_NonLocal<dim, spacedim>::requires_update_flags(
  const UpdateFlags flags) const
{
  return flags;
}



template <int dim, int spacedim>
std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
FE_Q1_NonLocal<dim, spacedim>::get_face_data(
  const UpdateFlags             update_flags,
  const Mapping<dim, spacedim> &mapping,
  const Quadrature<dim - 1> &   quadrature,
  internal::FEValuesImplementation::FiniteElementRelatedData<dim, spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
  return std_cxx14::make_unique<
    typename FiniteElement<dim, spacedim>::InternalDataBase>();
}


template <int dim, int spacedim>
std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
FE_Q1_NonLocal<dim, spacedim>::get_subface_data(
  const UpdateFlags             update_flags,
  const Mapping<dim, spacedim> &mapping,
  const Quadrature<dim - 1> &   quadrature,
  dealii::internal::FEValuesImplementation::FiniteElementRelatedData<dim,
                                                                     spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
  return std_cxx14::make_unique<
    typename FiniteElement<dim, spacedim>::InternalDataBase>();
}


template <int dim, int spacedim>
std::unique_ptr<typename FiniteElement<dim, spacedim>::InternalDataBase>
FE_Q1_NonLocal<dim, spacedim>::get_data(
  const UpdateFlags             flags,
  const Mapping<dim, spacedim> &mapping,
  const Quadrature<dim> &       quadrature,
  internal::FEValuesImplementation::FiniteElementRelatedData<dim, spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
  return std_cxx14::make_unique<
    typename FiniteElement<dim, spacedim>::InternalDataBase>();
}



template <int dim, int spacedim>
std::string
FE_Q1_NonLocal<dim, spacedim>::get_name() const
{
  std::ostringstream namebuf;
  namebuf << "FE_Q1_NonLocal<" << Utilities::dim_string(dim, spacedim) << ">";
  return namebuf.str();
}



template <int dim, int spacedim>
void
FE_Q1_NonLocal<dim, spacedim>::fill_fe_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const CellSimilarity::Similarity                            cell_similarity,
  const Quadrature<dim> &                                     quadrature,
  const Mapping<dim, spacedim> &                              mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &   mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
    &                                                            mapping_data,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  internal::FEValuesImplementation::FiniteElementRelatedData<dim, spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
}


template <int dim, int spacedim>
void
FE_Q1_NonLocal<dim, spacedim>::fill_fe_face_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int                                          face_no,
  const Quadrature<dim - 1> &                                 quadrature,
  const Mapping<dim, spacedim> &                              mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &   mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
    &                                                            mapping_data,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  internal::FEValuesImplementation::FiniteElementRelatedData<dim, spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
}


template <int dim, int spacedim>
void
FE_Q1_NonLocal<dim, spacedim>::fill_fe_subface_values(
  const typename Triangulation<dim, spacedim>::cell_iterator &cell,
  const unsigned int                                          face_no,
  const unsigned int                                          sub_no,
  const Quadrature<dim - 1> &                                 quadrature,
  const Mapping<dim, spacedim> &                              mapping,
  const typename Mapping<dim, spacedim>::InternalDataBase &   mapping_internal,
  const dealii::internal::FEValuesImplementation::MappingRelatedData<dim,
                                                                     spacedim>
    &                                                            mapping_data,
  const typename FiniteElement<dim, spacedim>::InternalDataBase &fe_internal,
  internal::FEValuesImplementation::FiniteElementRelatedData<dim, spacedim>
    &output_data) const
{
  Assert(false, ExcNotImplemented());
}

DEAL_II_NAMESPACE_CLOSE

#endif // dealii_fe_enriched_h
