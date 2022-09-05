// ---------------------------------------------------------------------
//
// Copyright (C) 2018 - 2022 by the deal.II authors
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

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

namespace Step85
{
  using namespace dealii;

  template <int dim>
  class LaplaceSolver
  {
  public:
    LaplaceSolver();

    void run();

  private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve();

    void output_results() const;

    double compute_L2_error_from_inside() const;

    double compute_L2_error_from_outside() const;

    bool face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

    bool face_has_ghost_penalty_outside(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const;

    void distribute_penalty_terms(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const unsigned int                                    f,
      const double                                          ghost_parameter,
      const double                                          cell_side_length,
      const unsigned                                        reminder);

    AffineConstraints<double> constraints;

    const unsigned int fe_degree;

    // const Functions::ConstantFunction<dim> boundary_condition;

    Triangulation<dim> triangulation;

    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double>  level_set;

    DoFHandler<dim> dof_handler;

    FESystem<dim>         fe_in;
    FESystem<dim>         fe_surf;
    FESystem<dim>         fe_out;
    hp::FECollection<dim> fe_collection;
    Vector<double>        solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> stiffness_matrix;
    Vector<double>       rhs;

    mutable TimerOutput timer;
  };

  template <int dim>
  LaplaceSolver<dim>::LaplaceSolver()
    : fe_degree(1)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , fe_in(FE_Q<dim>(fe_degree), 1, FE_Nothing<dim>(), 1)
    , fe_surf(FE_Q<dim>(fe_degree), 1, FE_Q<dim>(fe_degree), 1)
    , fe_out(FE_Nothing<dim>(), 1, FE_Q<dim>(fe_degree), 1)
    , mesh_classifier(level_set_dof_handler, level_set)
    , timer(std::cout, TimerOutput::summary, TimerOutput::cpu_times)
  {
    fe_collection.push_back(fe_in);
    fe_collection.push_back(fe_surf);
    fe_collection.push_back(fe_out);
  }

  template <int dim>
  void LaplaceSolver<dim>::make_grid()
  {
    std::cout << "Creating background mesh" << std::endl;

    GridGenerator::hyper_cube(triangulation, -1., 1.);
    triangulation.refine_global(3);
  }

  template <int dim>
  void LaplaceSolver<dim>::setup_discrete_level_set()
  {
    std::cout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());
    const double                                 R = .45;
    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere({}, R);
    VectorTools::interpolate(level_set_dof_handler,
                             signed_distance_sphere,
                             level_set);
  }

  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim> & point,
                 const unsigned int component = 0) const override;
  };

  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> & point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    // const double R = .45;
    // const double r = point.norm();
    // return (r <= R) ? point[0] : ((R * R) / (r * r)) * point[0];

    return std::sin(2. * numbers::PI * point[0]) *
           std::sin(2. * numbers::PI * point[1]);
    // 1. - 2. / dim * (point.norm_square() - 1.);
  }

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim)
    {}

    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim> &p,
                              Vector<double> &  value) const override;
  };

  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & p,
                                    const unsigned int component) const
  {
    Assert(component < this->n_components,
           ExcIndexRange(component, 0, this->n_components));

    if (component == dim - 1)
      switch (dim)
        {
          case 2:
            return AnalyticalSolution<dim>().value(p);
          default:
            Assert(false, ExcNotImplemented());
        }

    return 0;
  }

  template <int dim>
  void BoundaryValues<dim>::vector_value(const Point<dim> &p,
                                         Vector<double> &  values) const
  {
    for (unsigned int c = 0; c < this->n_components; ++c)
      values(c) = BoundaryValues<dim>::value(p, c);
  }

  template <int dim>
  class RhsFunction : public Function<dim>
  {
  public:
    RhsFunction()
      : Function<dim>()
    {}
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };

  template <int dim>
  double RhsFunction<dim>::value(const Point<dim> & p,
                                 const unsigned int component) const
  {
    (void)component;
    return 8. * numbers::PI * numbers::PI * std::sin(2. * numbers::PI * p[0]) *
           std::sin(2. * numbers::PI * p[1]);
    // return 0.;
  }

  enum ActiveFEIndex
  {
    sol_in           = 0, // inside
    sol_intersection = 1, // intersection
    sol_out          = 2  // outside
  };

  template <int dim>
  void LaplaceSolver<dim>::distribute_dofs()
  {
    std::cout << "Distributing degrees of freedom" << std::endl;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

        if (cell_location == NonMatching::LocationToLevelSet::inside)
          {
            // Q1, Nothing
            cell->set_active_fe_index(ActiveFEIndex::sol_in);
          }
        if (cell_location == NonMatching::LocationToLevelSet::outside)
          {
            // Nothing, Q1
            cell->set_active_fe_index(ActiveFEIndex::sol_out);
          }
        if (cell_location == NonMatching::LocationToLevelSet::intersected)
          {
            // Q1,Q1
            cell->set_active_fe_index(ActiveFEIndex::sol_intersection);
          }
      }
    dof_handler.distribute_dofs(fe_collection);

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const FEValuesExtractors::Vector exterior(0);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             constraints,
                                             fe_collection.component_mask(
                                               exterior));
    // VectorTools::interpolate_boundary_values(
    //   dof_handler,
    //   0,
    //   /*Functions::ZeroFunction<dim>(dim),*/
    //   Functions::ConstantFunction<dim>(0., dim),
    //   constraints,
    //   fe_collection.component_mask(exterior));

    constraints.close();
  }

  template <int dim>
  void LaplaceSolver<dim>::initialize_matrices()
  {
    std::cout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto &       cell,
                                            const unsigned int face_index) {
      if (this->face_has_ghost_penalty(cell, face_index) ||
          this->face_has_ghost_penalty_outside(cell, face_index))
        {
          return true;
        }
      else
        {
          return false;
        }
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int           n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    cell_coupling[0][1] = DoFTools::always;
    cell_coupling[1][1] = DoFTools::always;
    cell_coupling[1][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;
    face_coupling[0][1] = DoFTools::always;
    face_coupling[1][0] = DoFTools::always;
    face_coupling[1][1] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool                      keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }

  /**
   * Distribute interface penalty terms. Since this is going to be done when
   * we're *outside* and *inside*, we add a member function that takes care of
   * this, to avoid code duplication. We split the jump term
   *
   * [u][v] = u0v0 - u0v1 - u1v0 + u1v1
   *
   * so we get 4 local matrices, each one of size n_local_dofs x n_local_dofs
   * and we load them one by one into the stiffness matrix.
   */
  template <int dim>
  void LaplaceSolver<dim>::distribute_penalty_terms(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    f,
    const double                                          ghost_parameter,
    const double                                          cell_side_length,
    const unsigned int                                    reminder)
  {
    const QGauss<dim - 1>    face_quadrature(fe_degree + 1);
    hp::QCollection<dim - 1> q_collection(face_quadrature);
    hp::FEFaceValues<dim>    hp_fe_face0(fe_collection,
                                      q_collection,
                                      update_gradients | update_JxW_values |
                                        update_normal_vectors);
    hp::FEFaceValues<dim>    hp_fe_face1(fe_collection,
                                      q_collection,
                                      update_gradients | update_JxW_values |
                                        update_normal_vectors);
    hp_fe_face0.reinit(cell, f);
    // initialize for neighboring cell and the same face f, seen by
    // the neighboring cell
    hp_fe_face1.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));
    const FEFaceValues<dim> &fe_face0 = hp_fe_face0.get_present_fe_values();
    const FEFaceValues<dim> &fe_face1 = hp_fe_face1.get_present_fe_values();

    // std::cout << "DOfs0 per cell: " << fe_face0.dofs_per_cell << std::endl;
    // std::cout << "DOfs1per cell: " << fe_face1.dofs_per_cell << std::endl;
    // if (fe_face0.dofs_per_cell == 4)
    //   {
    //     std::cout << "4 DoFS cell0" << std::endl;
    //   }
    // if (fe_face1.dofs_per_cell == 4)
    //   {
    //     std::cout << "4 DoFS cell1" << std::endl;
    //   }
    Assert(!(fe_face0.dofs_per_cell == 4 && fe_face1.dofs_per_cell == 4),
           ExcMessage("They cannot have both 4 DoFs."));

    std::vector<types::global_dof_index> local_dofs0_indices(
      fe_face0.dofs_per_cell);
    std::vector<types::global_dof_index> local_dofs1_indices(
      fe_face1.dofs_per_cell);
    cell->get_dof_indices(local_dofs0_indices);
    cell->neighbor(f)->get_dof_indices(local_dofs1_indices);

    FullMatrix<double> A_00(fe_face0.dofs_per_cell, fe_face0.dofs_per_cell);
    FullMatrix<double> A_01(fe_face0.dofs_per_cell, fe_face1.dofs_per_cell);
    FullMatrix<double> A_10(fe_face1.dofs_per_cell, fe_face0.dofs_per_cell);
    FullMatrix<double> A_11(fe_face1.dofs_per_cell, fe_face1.dofs_per_cell);
    // A00
    for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
      {
        const Tensor<1, dim> normal = fe_face0.normal_vector(q);
        Assert(std::abs(fe_face0.JxW(q) - fe_face1.JxW(q)) < 1e-15,
               ExcMessage("Not consistent JxW")); // sanity check.
        const double interface_JxW = fe_face0.JxW(q);
        {
          for (unsigned int i = 0; i < fe_face0.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < fe_face0.dofs_per_cell; ++j)
                {
                  if (fe_face0.dofs_per_cell == 8)
                    {
                      if (i % 2 == reminder && j % 2 == reminder)
                        {
                          A_00(i, j) +=
                            .5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else
                    {
                      A_00(i, j) += .5 * ghost_parameter * cell_side_length *
                                    normal * fe_face0.shape_grad(i, q) *
                                    normal * fe_face0.shape_grad(j, q) *
                                    interface_JxW;
                    }
                }
            }
        }
      }
    // A01
    for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
      {
        const Tensor<1, dim> normal        = fe_face0.normal_vector(q);
        const double         interface_JxW = fe_face0.JxW(q);
        {
          for (unsigned int i = 0; i < fe_face0.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < fe_face1.dofs_per_cell; ++j)
                {
                  if (fe_face1.dofs_per_cell == 8 &&
                      fe_face0.dofs_per_cell == 4)
                    {
                      if (j % 2 == reminder)
                        {
                          A_01(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 4 &&
                           fe_face0.dofs_per_cell == 8)
                    {
                      if (i % 2 == reminder)
                        {
                          A_01(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 8 &&
                           fe_face0.dofs_per_cell == 8)
                    {
                      if (i % 2 == reminder && j % 2 == reminder)
                        {
                          A_01(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                }
            }
        }
      }
    // A10
    for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
      {
        const Tensor<1, dim> normal        = fe_face1.normal_vector(q);
        const double         interface_JxW = fe_face0.JxW(q);
        {
          for (unsigned int i = 0; i < fe_face1.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < fe_face0.dofs_per_cell; ++j)
                {
                  if (fe_face1.dofs_per_cell == 8 &&
                      fe_face0.dofs_per_cell == 4)
                    {
                      if (i % 2 == reminder)
                        {
                          A_10(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 4 &&
                           fe_face0.dofs_per_cell == 8)
                    {
                      if (j % 2 == reminder)
                        {
                          A_10(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 8 &&
                           fe_face0.dofs_per_cell == 8)
                    {
                      if (i % 2 == reminder && j % 2 == reminder)
                        {
                          A_10(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                }
            }
        }
      }
    // A11
    for (unsigned int q = 0; q < fe_face1.n_quadrature_points; ++q)
      {
        const Tensor<1, dim> normal        = fe_face1.normal_vector(q);
        const double         interface_JxW = fe_face0.JxW(q);
        {
          for (unsigned int i = 0; i < fe_face1.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < fe_face1.dofs_per_cell; ++j)
                {
                  if (fe_face1.dofs_per_cell == 8)
                    {
                      if (i % 2 == reminder && j % 2 == reminder)
                        {
                          A_11(i, j) +=
                            .5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else
                    {
                      A_11(i, j) += .5 * ghost_parameter * cell_side_length *
                                    normal * fe_face1.shape_grad(i, q) *
                                    normal * fe_face1.shape_grad(j, q) *
                                    interface_JxW;
                    }
                }
            }
        }
      }

    stiffness_matrix.add(local_dofs0_indices, A_00);
    stiffness_matrix.add(local_dofs0_indices, local_dofs1_indices, A_01);
    stiffness_matrix.add(local_dofs1_indices, local_dofs0_indices, A_10);
    stiffness_matrix.add(local_dofs1_indices, A_11);
  }

  template <int dim>
  bool LaplaceSolver<dim>::face_has_ghost_penalty(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  }

  template <int dim>
  bool LaplaceSolver<dim>::face_has_ghost_penalty_outside(
    const typename Triangulation<dim>::active_cell_iterator &cell,
    const unsigned int                                       face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
      mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
      mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::inside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::inside)
      return true;

    return false;
  }

  template <int dim>
  void LaplaceSolver<dim>::assemble_system()
  {
    std::cout << "Assembling" << std::endl;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    // std::cout << "nDoFs per intersected cell: "
    //           << fe_collection[1].dofs_per_cell << std::endl;
    const unsigned int n_dofs_per_intersected_cell =
      fe_collection[1].dofs_per_cell;
    FullMatrix<double> local_stiffness_surf(n_dofs_per_intersected_cell,
                                            n_dofs_per_intersected_cell);
    Vector<double>     local_rhs_surf(n_dofs_per_intersected_cell);
    std::vector<types::global_dof_index> local_dof_indices_surf(
      n_dofs_per_intersected_cell);

    const double ghost_parameter   = 0.5;
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree; // 1e5

    const QGauss<dim - 1> face_quadrature(fe_degree + 1);
    const QGauss<1>       quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.outside = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    hp::QCollection<dim> q_collection{QGauss<dim>(2 * fe_degree + 1)};
    hp::FEValues<dim>    hp_fe_values(fe_collection,
                                   q_collection,
                                   update_values | update_gradients |
                                     update_JxW_values |
                                     update_quadrature_points);

    RhsFunction<dim> rhs_function;

    // Here I loop only on the inside
    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_in))
      {
        local_stiffness = 0.;
        local_rhs       = 0.;

        const double cell_side_length = cell->minimum_vertex_distance();
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        // std::cout << inside_fe_values->shape_grad(0, 1) << std::endl;

        if (inside_fe_values)
          for (const unsigned int q :
               inside_fe_values->quadrature_point_indices())
            {
              const Point<dim> &point = inside_fe_values->quadrature_point(q);
              for (const unsigned int i : inside_fe_values->dof_indices())
                {
                  for (const unsigned int j : inside_fe_values->dof_indices())
                    {
                      local_stiffness(i, j) +=
                        inside_fe_values->shape_grad(i, q) *
                        inside_fe_values->shape_grad(j, q) *
                        inside_fe_values->JxW(q);
                    }
                  local_rhs(i) += rhs_function.value(point) *
                                  inside_fe_values->shape_value(i, q) *
                                  inside_fe_values->JxW(q);
                }
            }

        cell->get_dof_indices(local_dof_indices);

        stiffness_matrix.add(local_dof_indices, local_stiffness);
        rhs.add(local_dof_indices, local_rhs);

        for (unsigned int f : cell->face_indices())
          {
            if (face_has_ghost_penalty(cell, f))
              {
                distribute_penalty_terms(cell,
                                         f,
                                         ghost_parameter,
                                         cell_side_length,
                                         0); // reminder == 0
              }
          }
      }

    // Solve on outside
    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_out))
      {
        local_stiffness = 0.;
        local_rhs       = 0.;

        const double cell_side_length = cell->minimum_vertex_distance();
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
          non_matching_fe_values.get_outside_fe_values();

        if (outside_fe_values)
          {
            for (const unsigned int q :
                 outside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  outside_fe_values->quadrature_point(q);
                for (const unsigned int i : outside_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         outside_fe_values->dof_indices())
                      {
                        local_stiffness(i, j) +=
                          outside_fe_values->shape_grad(i, q) *
                          outside_fe_values->shape_grad(j, q) *
                          outside_fe_values->JxW(q);
                      }
                    local_rhs(i) += rhs_function.value(point) *
                                    outside_fe_values->shape_value(i, q) *
                                    outside_fe_values->JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
          local_stiffness, local_rhs, local_dof_indices, stiffness_matrix, rhs);

        // for (unsigned int f : cell->face_indices())
        //   {
        //     if (face_has_ghost_penalty_outside(cell, f))
        //       {
        //         distribute_penalty_terms(cell,
        //                                  f,
        //                                  ghost_parameter,
        //                                  cell_side_length,
        //                                  1); // reminder == 1
        //       }
        //   }
      }

    // Solve on intersection
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::ActiveFEIndexEqualTo(
                                ActiveFEIndex::sol_intersection))
      {
        local_stiffness_surf = 0.;
        local_rhs_surf       = 0.;

        const double cell_side_length = cell->minimum_vertex_distance();
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        hp_fe_values.reinit(cell);
        const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

        if (inside_fe_values)
          {
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                const Point<dim> &point = fe_values.quadrature_point(q);
                for (const unsigned int i : fe_values.dof_indices())
                  {
                    for (const unsigned int j : fe_values.dof_indices())
                      {
                        if (i % 2 == 0 && j % 2 == 0)
                          {
                            local_stiffness_surf(i, j) +=
                              fe_values.shape_grad(i, q) *
                              fe_values.shape_grad(j, q) * fe_values.JxW(q);
                          }
                      }

                    if (i % 2 == 0)
                      {
                        local_rhs_surf(i) += rhs_function.value(point) *
                                             fe_values.shape_value(i, q) *
                                             fe_values.JxW(q);
                      }
                  }
              }
          }

        const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
          non_matching_fe_values.get_outside_fe_values();

        if (outside_fe_values)
          {
            for (const unsigned int q : fe_values.quadrature_point_indices())
              {
                const Point<dim> &point = fe_values.quadrature_point(q);
                for (const unsigned int i : fe_values.dof_indices())
                  {
                    for (const unsigned int j : fe_values.dof_indices())
                      {
                        if (i % 2 == 1 && j % 2 == 1)
                          {
                            local_stiffness_surf(i, j) +=
                              fe_values.shape_grad(i, q) *
                              fe_values.shape_grad(j, q) * fe_values.JxW(q);
                          }
                      }

                    if (i % 2 == 1)
                      {
                        local_rhs_surf(i) += rhs_function.value(point) *
                                             fe_values.shape_value(i, q) *
                                             fe_values.JxW(q);
                      }
                  }
              }
          }
        // non_matching_fe_values.reinit(cell);

        // const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
        //   non_matching_fe_values.get_inside_fe_values();

        // if (inside_fe_values)
        //   {
        //     for (const unsigned int q :
        //          inside_fe_values->quadrature_point_indices())
        //       {
        //         const Point<dim> &point =
        //         inside_fe_values->quadrature_point(q); for (const unsigned
        //         int i : inside_fe_values->dof_indices())
        //           {
        //             for (const unsigned int j :
        //             inside_fe_values->dof_indices())
        //               {
        //                 if (i % 2 == 0 && j % 2 == 0)
        //                   {
        //                     local_stiffness_surf(i, j) +=
        //                       inside_fe_values->shape_grad(i, q) *
        //                       inside_fe_values->shape_grad(j, q) *
        //                       inside_fe_values->JxW(q);
        //                   }
        //               }

        //             if (i % 2 == 0)
        //               {
        //                 local_rhs_surf(i) +=
        //                   rhs_function.value(point) *
        //                   inside_fe_values->shape_value(i, q) *
        //                   inside_fe_values->JxW(q);
        //               }
        //           }
        //       }
        //   }

        // for (unsigned int f : cell->face_indices())
        //   {
        //     if (face_has_ghost_penalty(cell, f))
        //       {
        //         distribute_penalty_terms(cell,
        //                                  f,
        //                                  ghost_parameter,
        //                                  cell_side_length,
        //                                  0); // reminder == 0
        //       }
        //     if (face_has_ghost_penalty_outside(cell, f))
        //       {
        //         distribute_penalty_terms(cell,
        //                                  f,
        //                                  ghost_parameter,
        //                                  cell_side_length,
        //                                  1); // reminder == 1
        //       }
        //   }

        // const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
        //   non_matching_fe_values.get_outside_fe_values();

        // if (outside_fe_values)
        //   {
        //     for (const unsigned int q :
        //          outside_fe_values->quadrature_point_indices())
        //       {
        //         const Point<dim> &point =
        //           outside_fe_values->quadrature_point(q);
        //         for (const unsigned int i : outside_fe_values->dof_indices())
        //           {
        //             for (const unsigned int j :
        //                  outside_fe_values->dof_indices())
        //               {
        //                 if (i % 2 == 1 && j % 2 == 1)
        //                   {
        //                     local_stiffness_surf(i, j) +=
        //                       outside_fe_values->shape_grad(i, q) *
        //                       outside_fe_values->shape_grad(j, q) *
        //                       outside_fe_values->JxW(q);
        //                   }
        //               }
        //             if (i % 2 == 1)
        //               {
        //                 local_rhs_surf(i) +=
        //                   rhs_function.value(point) *
        //                   outside_fe_values->shape_value(i, q) *
        //                   outside_fe_values->JxW(q);
        //               }
        //           }
        //       }
        //   }

        const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
        if (surface_fe_values)
          {
            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const Tensor<1, dim> &normal =
                  surface_fe_values->normal_vector(q);
                for (const unsigned int i : surface_fe_values->dof_indices())
                  {
                    for (const unsigned int j :
                         surface_fe_values->dof_indices())
                      {
                        if (i % 2 == 0 && j % 2 == 0)
                          {
                            Assert(fe_collection[1]
                                         .system_to_component_index(i)
                                         .first == 0 &&
                                     fe_collection[1]
                                         .system_to_component_index(j)
                                         .first == 0,
                                   ExcMessage("Component mismatch!"));
                            local_stiffness_surf(i, j) +=
                              (-normal * surface_fe_values->shape_grad(i, q) *
                                 surface_fe_values->shape_value(j, q) +
                               -normal * surface_fe_values->shape_grad(j, q) *
                                 surface_fe_values->shape_value(i, q) +
                               nitsche_parameter / cell_side_length *
                                 surface_fe_values->shape_value(i, q) *
                                 surface_fe_values->shape_value(j, q)) *
                              surface_fe_values->JxW(q);
                          }
                        else if (i % 2 == 1 && j % 2 == 1)
                          {
                            Assert(fe_collection[1]
                                         .system_to_component_index(i)
                                         .first == 1 &&
                                     fe_collection[1]
                                         .system_to_component_index(j)
                                         .first == 1,
                                   ExcMessage("Component mismatch!"));
                            local_stiffness_surf(i, j) +=
                              (normal * surface_fe_values->shape_grad(i, q) *
                                 surface_fe_values->shape_value(j, q) +
                               normal * surface_fe_values->shape_grad(j, q) *
                                 surface_fe_values->shape_value(i, q) +
                               nitsche_parameter / cell_side_length *
                                 surface_fe_values->shape_value(i, q) *
                                 surface_fe_values->shape_value(j, q)) *
                              surface_fe_values->JxW(q);
                          }
                      }

                    if (i % 2 == 0)
                      {
                        local_rhs_surf(i) +=
                          AnalyticalSolution<dim>().value(point) *
                          (nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) -
                           normal * surface_fe_values->shape_grad(i, q)) *
                          surface_fe_values->JxW(q);
                      }
                    else
                      {
                        local_rhs_surf(i) +=
                          AnalyticalSolution<dim>().value(point) *
                          (nitsche_parameter / cell_side_length *
                             surface_fe_values->shape_value(i, q) +
                           normal * surface_fe_values->shape_grad(i, q)) *
                          surface_fe_values->JxW(q);
                      }
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices_surf);

        stiffness_matrix.add(local_dof_indices_surf, local_stiffness_surf);
        rhs.add(local_dof_indices_surf, local_rhs_surf);
      }
  }

  template <int dim>
  void LaplaceSolver<dim>::solve()
  {
    std::cout << "Solving system" << std::endl;

    const unsigned int max_iterations = 1e5; //;solution.size();

    SolverControl solver_control(max_iterations);
    SolverCG<>    solver(solver_control);
    solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
    std::cout << "Solved in " << solver_control.last_step() << " iterations."
              << std::endl;
    constraints.distribute(solution);
  }

  template <int dim>
  void LaplaceSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output_inside_inter("cutFEM_inside_intersected_hybrid.vtu");
    data_out.write_vtu(output_inside_inter);

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) ==
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output_outside("cutFEM_outside_hybrid.vtu");
    data_out.write_vtu(output_outside);
  }

  template <int dim>
  double LaplaceSolver<dim>::compute_L2_error_from_inside() const
  {
    std::cout << "Computing L2 error from inside" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
      update_values | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    const AnalyticalSolution<dim> analytical_solution;
    double                        error_L2_squared = 0;
    FEValuesExtractors::Scalar    interior(0);


    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_in))
      {
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            (*fe_values)[interior].get_function_values(solution,
                                                       solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
      }

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::ActiveFEIndexEqualTo(
                                ActiveFEIndex::sol_intersection))
      {
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

        if (inside_fe_values)
          {
            std::vector<double> solution_values(
              inside_fe_values->n_quadrature_points);
            (*inside_fe_values)[interior].get_function_values(solution,
                                                              solution_values);

            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * inside_fe_values->JxW(q);
              }
          }


        const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
        if (surface_fe_values)
          {
            std::vector<double> solution_values(
              surface_fe_values->n_quadrature_points);
            (*surface_fe_values)[interior].get_function_values(solution,
                                                               solution_values);

            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const double error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * surface_fe_values->JxW(q);
              }
          }
      }

    return std::sqrt(error_L2_squared);
  }



  template <int dim>
  double LaplaceSolver<dim>::compute_L2_error_from_outside() const
  {
    std::cout << "Computing L2 error from outside" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;

    region_update_flags.outside =
      update_values | update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    const AnalyticalSolution<dim> analytical_solution;
    double                        error_L2_squared = 0.;
    FEValuesExtractors::Scalar    exterior(1);

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
           IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_out))
      {
        non_matching_fe_values.reinit(cell);

        const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_outside_fe_values();

        if (fe_values)
          {
            std::vector<double> solution_values(fe_values->n_quadrature_points);
            (*fe_values)[exterior].get_function_values(solution,
                                                       solution_values);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * fe_values->JxW(q);
              }
          }
      }

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::ActiveFEIndexEqualTo(
                                ActiveFEIndex::sol_intersection))
      {
        non_matching_fe_values.reinit(cell);
        const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
          non_matching_fe_values.get_outside_fe_values();

        if (outside_fe_values)
          {
            std::vector<double> solution_values(
              outside_fe_values->n_quadrature_points);
            (*outside_fe_values)[exterior].get_function_values(solution,
                                                               solution_values);

            for (const unsigned int q :
                 outside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  outside_fe_values->quadrature_point(q);
                const double error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * outside_fe_values->JxW(q);
              }
          }


        const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();
        if (surface_fe_values)
          {
            std::vector<double> solution_values(
              surface_fe_values->n_quadrature_points);
            (*surface_fe_values)[exterior].get_function_values(solution,
                                                               solution_values);

            for (const unsigned int q :
                 surface_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  surface_fe_values->quadrature_point(q);
                const double error_at_point =
                  solution_values[q] - analytical_solution.value(point);
                error_L2_squared +=
                  std::pow(error_at_point, 2) * surface_fe_values->JxW(q);
              }
          }
      }

    return std::sqrt(error_L2_squared);
  }

  template <int dim>
  void LaplaceSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 6;

    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        TimerOutput::Scope timer_section(timer,
                                         "Total time cycle " +
                                           std::to_string(cycle));
        std::cout << "Refinement cycle " << cycle << std::endl;
        triangulation.refine_global(1);
        setup_discrete_level_set();
        std::cout << "Classifying cells" << std::endl;
        mesh_classifier.reclassify();
        distribute_dofs();
        std::cout << "Solving with " << dof_handler.n_dofs() << "DoFs"
                  << std::endl;
        initialize_matrices();
        assemble_system();
        solve();
        // if (cycle == 3)
        output_results();
        // const double error_L2 = compute_L2_error_from_inside();
        const double error_L2 = compute_L2_error_from_outside();
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("Mesh size", cell_side_length);
        convergence_table.add_value("L2-Error", error_L2);

        convergence_table.evaluate_convergence_rates(
          "L2-Error", ConvergenceTable::reduction_rate_log2);
        convergence_table.set_scientific("L2-Error", true);

        std::cout << std::endl;
        convergence_table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

} // namespace Step85

int main()
{
  const int dim = 2;

  Step85::LaplaceSolver<dim> laplace_solver;
  laplace_solver.run();
}
