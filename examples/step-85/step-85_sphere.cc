// // ---------------------------------------------------------------------
// //
// // Copyright (C) 2018 - 2022 by the deal.II authors
// //
// // This file is part of the deal.II library.
// //
// // The deal.II library is free software; you can use it, redistribute
// // it, and/or modify it under the terms of the GNU Lesser General
// // Public License as published by the Free Software Foundation; either
// // version 2.1 of the License, or (at your option) any later version.
// // The full text of the license can be found in the file LICENSE.md at
// // the top level directory of deal.II.
// //
// // ---------------------------------------------------------------------

// #include <deal.II/base/function.h>

// #include <deal.II/base/convergence_table.h>
// #include <deal.II/base/point.h>
// #include <deal.II/base/quadrature.h>
// #include <deal.II/base/quadrature_lib.h>
// #include <deal.II/base/tensor.h>

// #include <deal.II/dofs/dof_tools.h>

// #include <deal.II/fe/fe_interface_values.h>
// #include <deal.II/fe/fe_nothing.h>
// #include <deal.II/fe/fe_q.h>
// #include <deal.II/fe/fe_system.h>
// #include <deal.II/fe/fe_update_flags.h>
// #include <deal.II/fe/fe_values.h>
// #include <deal.II/hp/q_collection.h>

// #include <deal.II/grid/filtered_iterator.h>
// #include <deal.II/grid/grid_generator.h>
// #include <deal.II/grid/tria.h>

// #include <deal.II/hp/fe_collection.h>
// #include <deal.II/hp/q_collection.h>

// #include <deal.II/lac/affine_constraints.h>
// #include <deal.II/lac/dynamic_sparsity_pattern.h>
// #include <deal.II/lac/full_matrix.h>
// #include <deal.II/lac/precondition.h>
// #include <deal.II/lac/solver_cg.h>
// #include <deal.II/lac/solver_control.h>
// #include <deal.II/lac/sparse_matrix.h>
// #include <deal.II/lac/sparsity_pattern.h>
// #include <deal.II/lac/vector.h>

// #include <deal.II/numerics/data_out.h>
// #include <deal.II/numerics/vector_tools.h>

// #include <fstream>
// #include <vector>

// #include <deal.II/base/function_signed_distance.h>

// #include <deal.II/non_matching/fe_immersed_values.h>
// #include <deal.II/non_matching/fe_values.h>
// #include <deal.II/non_matching/mesh_classifier.h>

// namespace Step85
// {
//   using namespace dealii;
//   class ImplicitFunction : public Function<3>
//   {
//   public:
//     virtual double value(const Point<3> &   p,
//                          const unsigned int component = 0) const override
//     {
//       // const double xc = 0.2 / std::sqrt(20.);
//       // const double yc = 0.2 / std::sqrt(20.);
//       // const double w  = 12.;
//       const double R = .3;
//       const double r = .1;
//       const double x = p[0];
//       const double y = p[1];
//       const double z = p[2];
//       const double oscillating_term =
//         (1. - 2. * (y * y) / (x * x + y * y)) *
//         (1. - 16. * (x * x * y * y) / ((x * x + y * y) * (x * x + y * y)));
//       // return std::sqrt(p[0]*p[0] + p[1]*p[1]) - r*(p[0]*p[0]*p[0]
//       // - 3.*p[0]*p[1]*p[1])*std::pow(p[0]*p[0] + p[1]*p[1],-3./2.) - R;
//       // return std::sqrt(x * x + y * y) - r * oscillating_term - R;

//       return std::sqrt(x * x + y * y + z * z) - R;
//     }
//   };
//   template <int dim>
//   class LaplaceSolver
//   {
//   public:
//     LaplaceSolver();

//     void run();

//   private:
//     void make_grid();

//     void setup_discrete_level_set();

//     void distribute_dofs();

//     void initialize_matrices();

//     void assemble_system();

//     void solve();

//     void output_results() const;

//     double compute_L2_error_from_inside() const;

//     double compute_L2_error_from_outside() const;

//     double compute_H1_error_from_outside() const;

//     bool face_has_ghost_penalty(
//       const typename Triangulation<dim>::active_cell_iterator &cell,
//       const unsigned int face_index) const;

//     bool face_has_ghost_penalty_outside(
//       const typename Triangulation<dim>::active_cell_iterator &cell,
//       const unsigned int face_index) const;

//     void distribute_penalty_terms(
//       const typename DoFHandler<dim>::active_cell_iterator &cell,
//       const unsigned int                                    f,
//       const double                                          ghost_parameter,
//       const double                                          cell_side_length,
//       const unsigned                                        reminder);

//     AffineConstraints<double> constraints;

//     const unsigned int fe_degree;

//     // const Functions::ConstantFunction<dim> boundary_condition;

//     Triangulation<dim> triangulation;

//     const FE_Q<dim> fe_level_set;
//     DoFHandler<dim> level_set_dof_handler;
//     Vector<double>  level_set;

//     DoFHandler<dim> dof_handler;

//     FESystem<dim>         fe_in;
//     FESystem<dim>         fe_surf;
//     FESystem<dim>         fe_out;
//     hp::FECollection<dim> fe_collection;
//     Vector<double>        solution;

//     NonMatching::MeshClassifier<dim> mesh_classifier;

//     SparsityPattern      sparsity_pattern;
//     SparseMatrix<double> stiffness_matrix;
//     Vector<double>       rhs;
//   };

//   template <int dim>
//   LaplaceSolver<dim>::LaplaceSolver()
//     : fe_degree(1)
//     , fe_level_set(fe_degree)
//     , level_set_dof_handler(triangulation)
//     , dof_handler(triangulation)
//     , fe_in(FE_Q<dim>(fe_degree), 1, FE_Nothing<dim>(), 1)
//     , fe_surf(FE_Q<dim>(fe_degree), 1, FE_Q<dim>(fe_degree), 1)
//     , fe_out(FE_Nothing<dim>(), 1, FE_Q<dim>(fe_degree), 1)
//     , mesh_classifier(level_set_dof_handler, level_set)
//   {
//     fe_collection.push_back(fe_in);
//     fe_collection.push_back(fe_surf);
//     fe_collection.push_back(fe_out);
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::make_grid()
//   {
//     std::cout << "Creating background mesh" << std::endl;

//     GridGenerator::hyper_cube(triangulation, -1., 1.);
//     triangulation.refine_global(1);
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::setup_discrete_level_set()
//   {
//     std::cout << "Setting up discrete level set function" << std::endl;

//     level_set_dof_handler.distribute_dofs(fe_level_set);
//     level_set.reinit(level_set_dof_handler.n_dofs());
//     const double                                 R = .45;
//     const Functions::SignedDistance::Sphere<dim> signed_distance_sphere({},
//     R); ImplicitFunction                             implicit_function;
//     VectorTools::interpolate(level_set_dof_handler,
//                              signed_distance_sphere,
//                              level_set);
//   }

//   template <int dim>
//   class AnalyticalSolution : public Function<dim>
//   {
//   public:
//     double value(const Point<dim> & point,
//                  const unsigned int component = 0) const override;

//     Tensor<1, dim> gradient(const Point<dim> & point,
//                             const unsigned int component = 0) const override;
//   };

//   template <int dim>
//   double AnalyticalSolution<dim>::value(const Point<dim> & point,
//                                         const unsigned int component) const
//   {
//     AssertIndexRange(component, this->n_components);
//     (void)component;

//     switch (dim)
//       {
//         case 2:
//           return std::sin(2. * numbers::PI * point[0]) *
//                  std::sin(2. * numbers::PI * point[1]);
//         case 3:
//           return std::sin(2. * numbers::PI * point[0]) *
//                  std::sin(2. * numbers::PI * point[1]) *
//                  std::sin(2. * numbers::PI * point[2]);
//         default:
//           Assert(false, ExcNotImplemented());
//       }

//     // 1. - 2. / dim * (point.norm_square() - 1.);
//   }

//   template <int dim>
//   Tensor<1, dim>
//   AnalyticalSolution<dim>::gradient(const Point<dim> & point,
//                                     const unsigned int component) const
//   {
//     AssertIndexRange(component, this->n_components);
//     (void)component;


//     Tensor<1, dim> grad;
//     switch (dim)
//       {
//         case 2:
//           grad[0] = 2. * M_PI * std::cos(2. * M_PI * point[0]) *
//                     std::sin(2. * M_PI * point[1]);
//           grad[1] = 2. * M_PI * std::cos(2. * M_PI * point[1]) *
//                     std::sin(2. * M_PI * point[0]);
//           return grad;
//         case 3:
//           grad[0] = 2. * M_PI * std::cos(2. * M_PI * point[0]) *
//                     std::sin(2. * M_PI * point[1]) *
//                     std::sin(2. * M_PI * point[2]);
//           grad[1] = 2. * M_PI * std::sin(2. * M_PI * point[0]) *
//                     std::cos(2. * M_PI * point[1]) *
//                     std::sin(2. * M_PI * point[2]);
//           grad[2] = 2. * M_PI * std::sin(2. * M_PI * point[0]) *
//                     std::sin(2. * M_PI * point[1]) *
//                     std::cos(2. * M_PI * point[2]);
//           return grad;
//         default:
//           Assert(false, ExcNotImplemented());
//       }
//   }



//   template <int dim>
//   class BoundaryValues : public Function<dim>
//   {
//   public:
//     BoundaryValues()
//       : Function<dim>(dim - 1)
//     {}

//     virtual double value(const Point<dim> & p,
//                          const unsigned int component = 0) const override;

//     virtual void vector_value(const Point<dim> &p,
//                               Vector<double> &  value) const override;
//   };

//   template <int dim>
//   double BoundaryValues<dim>::value(const Point<dim> & p,
//                                     const unsigned int component) const
//   {
//     Assert(component < this->n_components,
//            ExcIndexRange(component, 0, this->n_components));

//     if (component == dim - 1)
//       switch (dim)
//         {
//           case 2:
//             return AnalyticalSolution<dim>().value(p);
//           case 3:
//             return AnalyticalSolution<dim>().value(p);
//           default:
//             Assert(false, ExcNotImplemented());
//         }

//     return 0;
//   }

//   template <int dim>
//   void BoundaryValues<dim>::vector_value(const Point<dim> &p,
//                                          Vector<double> &  values) const
//   {
//     for (unsigned int c = 0; c < this->n_components; ++c)
//       values(c) = BoundaryValues<dim>::value(p, c);
//   }

//   template <int dim>
//   class RhsFunction : public Function<dim>
//   {
//   public:
//     RhsFunction()
//       : Function<dim>()
//     {}
//     virtual double value(const Point<dim> & p,
//                          const unsigned int component = 0) const override;
//   };

//   template <int dim>
//   double RhsFunction<dim>::value(const Point<dim> & p,
//                                  const unsigned int component) const
//   {
//     (void)component;
//     switch (dim)
//       {
//         case 2:
//           return 8. * numbers::PI * numbers::PI *
//                  std::sin(2. * numbers::PI * p[0]) *
//                  std::sin(2. * numbers::PI * p[1]);
//         case 3:
//           return 12. * numbers::PI * numbers::PI *
//                  std::sin(2. * numbers::PI * p[0]) *
//                  std::sin(2. * numbers::PI * p[1]) *
//                  std::sin(2. * numbers::PI * p[2]);
//         default:
//           Assert(false, ExcMessage("Nonsense"));
//       };
//   }

//   enum ActiveFEIndex
//   {
//     sol_in           = 0, // inside
//     sol_intersection = 1, // intersection
//     sol_out          = 2  // outside
//   };

//   template <int dim>
//   void LaplaceSolver<dim>::distribute_dofs()
//   {
//     std::cout << "Distributing degrees of freedom" << std::endl;

//     for (const auto &cell : dof_handler.active_cell_iterators())
//       {
//         const NonMatching::LocationToLevelSet cell_location =
//           mesh_classifier.location_to_level_set(cell);

//         if (cell_location == NonMatching::LocationToLevelSet::inside)
//           {
//             // Q1, Nothing
//             cell->set_active_fe_index(ActiveFEIndex::sol_in);
//           }
//         if (cell_location == NonMatching::LocationToLevelSet::outside)
//           {
//             // Nothing, Q1
//             cell->set_active_fe_index(ActiveFEIndex::sol_out);
//           }
//         if (cell_location == NonMatching::LocationToLevelSet::intersected)
//           {
//             // Q1,Q1
//             cell->set_active_fe_index(ActiveFEIndex::sol_intersection);
//           }
//       }
//     dof_handler.distribute_dofs(fe_collection);

//     constraints.clear();
//     DoFTools::make_hanging_node_constraints(dof_handler, constraints);

//     const FEValuesExtractors::Vector exterior(0);
//     std::cout << fe_collection.n_components() << std::endl;
//     std::vector<bool> bools{true, true};
//     VectorTools::interpolate_boundary_values(
//       dof_handler,
//       0,
//       /*Functions::ZeroFunction<dim>(dim - 1),*/
//       BoundaryValues<dim>(),
//       constraints,
//       fe_collection[0].component_mask(bools));
//     // VectorTools::interpolate_boundary_values(
//     //   dof_handler,
//     //   0,
//     //   /*Functions::ZeroFunction<dim>(dim),*/
//     //   Functions::ConstantFunction<dim>(0., dim),
//     //   constraints,
//     //   fe_collection.component_mask(exterior));

//     constraints.close();
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::initialize_matrices()
//   {
//     std::cout << "Initializing matrices" << std::endl;

//     const auto face_has_flux_coupling = [&](const auto &       cell,
//                                             const unsigned int face_index) {
//       if (this->face_has_ghost_penalty(cell, face_index) ||
//           this->face_has_ghost_penalty_outside(cell, face_index))
//         {
//           return true;
//         }
//       else
//         {
//           return false;
//         }
//     };

//     DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

//     const unsigned int           n_components = fe_collection.n_components();
//     Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
//     Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
//     cell_coupling[0][0] = DoFTools::always;
//     cell_coupling[0][1] = DoFTools::always;
//     cell_coupling[1][1] = DoFTools::always;
//     cell_coupling[1][0] = DoFTools::always;
//     face_coupling[0][0] = DoFTools::always;
//     face_coupling[0][1] = DoFTools::always;
//     face_coupling[1][0] = DoFTools::always;
//     face_coupling[1][1] = DoFTools::always;

//     const AffineConstraints<double> constraints;
//     const bool                      keep_constrained_dofs = true;

//     DoFTools::make_flux_sparsity_pattern(dof_handler,
//                                          dsp,
//                                          constraints,
//                                          keep_constrained_dofs,
//                                          cell_coupling,
//                                          face_coupling,
//                                          numbers::invalid_subdomain_id,
//                                          face_has_flux_coupling);
//     sparsity_pattern.copy_from(dsp);

//     stiffness_matrix.reinit(sparsity_pattern);
//     solution.reinit(dof_handler.n_dofs());
//     rhs.reinit(dof_handler.n_dofs());
//   }

//   /**
//    * Distribute interface penalty terms. Since this is going to be done when
//    * we're *outside* and *inside*, we add a member function that takes care
//    of
//    * this, to avoid code duplication. We split the jump term
//    *
//    * [u][v] = u0v0 - u0v1 - u1v0 + u1v1
//    *
//    * so we get 4 local matrices, each one of size n_local_dofs x n_local_dofs
//    * and we load them one by one into the stiffness matrix.
//    */
//   template <int dim>
//   void LaplaceSolver<dim>::distribute_penalty_terms(
//     const typename DoFHandler<dim>::active_cell_iterator &cell,
//     const unsigned int                                    f,
//     const double                                          ghost_parameter,
//     const double                                          cell_side_length,
//     const unsigned int                                    reminder)
//   {
//     const QGauss<dim - 1>    face_quadrature(fe_degree + 1);
//     hp::QCollection<dim - 1> q_collection(face_quadrature);
//     hp::FEFaceValues<dim>    hp_fe_face0(fe_collection,
//                                       q_collection,
//                                       update_gradients | update_JxW_values |
//                                         update_normal_vectors);
//     hp::FEFaceValues<dim>    hp_fe_face1(fe_collection,
//                                       q_collection,
//                                       update_gradients | update_JxW_values |
//                                         update_normal_vectors);
//     hp_fe_face0.reinit(cell, f);
//     // initialize for neighboring cell and the same face f, seen by
//     // the neighboring cell
//     hp_fe_face1.reinit(cell->neighbor(f), cell->neighbor_of_neighbor(f));
//     const FEFaceValues<dim> &fe_face0 = hp_fe_face0.get_present_fe_values();
//     const FEFaceValues<dim> &fe_face1 = hp_fe_face1.get_present_fe_values();

//     // std::cout << "DOfs0 per cell: " << fe_face0.dofs_per_cell <<
//     std::endl;
//     // std::cout << "DOfs1per cell: " << fe_face1.dofs_per_cell << std::endl;
//     // if (fe_face0.dofs_per_cell == 4)
//     //   {
//     //     std::cout << "4 DoFS cell0" << std::endl;
//     //   }
//     // if (fe_face1.dofs_per_cell == 4)
//     //   {
//     //     std::cout << "4 DoFS cell1" << std::endl;
//     //   }
//     Assert(!(fe_face0.dofs_per_cell == 8 && fe_face1.dofs_per_cell == 8),
//            ExcMessage("They cannot have both 8 DoFs."));

//     std::vector<types::global_dof_index> local_dofs0_indices(
//       fe_face0.dofs_per_cell);
//     std::vector<types::global_dof_index> local_dofs1_indices(
//       fe_face1.dofs_per_cell);
//     cell->get_dof_indices(local_dofs0_indices);
//     cell->neighbor(f)->get_dof_indices(local_dofs1_indices);

//     FullMatrix<double> A_00(fe_face0.dofs_per_cell, fe_face0.dofs_per_cell);
//     FullMatrix<double> A_01(fe_face0.dofs_per_cell, fe_face1.dofs_per_cell);
//     FullMatrix<double> A_10(fe_face1.dofs_per_cell, fe_face0.dofs_per_cell);
//     FullMatrix<double> A_11(fe_face1.dofs_per_cell, fe_face1.dofs_per_cell);
//     // A00
//     for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
//       {
//         const Tensor<1, dim> normal = fe_face0.normal_vector(q);
//         Assert(std::abs(fe_face0.JxW(q) - fe_face1.JxW(q)) < 1e-15,
//                ExcMessage("Not consistent JxW")); // sanity check.
//         const double interface_JxW = fe_face0.JxW(q);
//         {
//           for (unsigned int i = 0; i < fe_face0.dofs_per_cell; ++i)
//             {
//               for (unsigned int j = 0; j < fe_face0.dofs_per_cell; ++j)
//                 {
//                   if (fe_face0.dofs_per_cell == 8 * 2)
//                     {
//                       if (i % 2 == reminder && j % 2 == reminder)
//                         {
//                           A_00(i, j) +=
//                             .5 * ghost_parameter * cell_side_length * normal
//                             * fe_face0.shape_grad(i, q) * normal *
//                             fe_face0.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else
//                     {
//                       A_00(i, j) += .5 * ghost_parameter * cell_side_length *
//                                     normal * fe_face0.shape_grad(i, q) *
//                                     normal * fe_face0.shape_grad(j, q) *
//                                     interface_JxW;
//                     }
//                 }
//             }
//         }
//       }
//     // A01
//     for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
//       {
//         const Tensor<1, dim> normal        = fe_face0.normal_vector(q);
//         const double         interface_JxW = fe_face0.JxW(q);
//         {
//           for (unsigned int i = 0; i < fe_face0.dofs_per_cell; ++i)
//             {
//               for (unsigned int j = 0; j < fe_face1.dofs_per_cell; ++j)
//                 {
//                   // std::cout
//                   //   << "fe_face0.dofs_per_cell = " <<
//                   fe_face0.dofs_per_cell
//                   //   << std::endl;

//                   // std::cout
//                   //   << "fe_face1.dofs_per_cell = " <<
//                   fe_face1.dofs_per_cell
//                   //   << std::endl;

//                   if (fe_face1.dofs_per_cell == 8 * 2 &&
//                       fe_face0.dofs_per_cell == 4 * 2)
//                     {
//                       if (j % 2 == reminder)
//                         {
//                           A_01(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face0.shape_grad(i, q) * normal *
//                             fe_face1.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else if (fe_face1.dofs_per_cell == 4 * 2 &&
//                            fe_face0.dofs_per_cell == 8 * 2)
//                     {
//                       if (i % 2 == reminder)
//                         {
//                           A_01(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face0.shape_grad(i, q) * normal *
//                             fe_face1.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else if (fe_face1.dofs_per_cell == 8 * 2 &&
//                            fe_face0.dofs_per_cell == 8 * 2)
//                     {
//                       if (i % 2 == reminder && j % 2 == reminder)
//                         {
//                           A_01(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face0.shape_grad(i, q) * normal *
//                             fe_face1.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                 }
//             }
//         }
//       }
//     // A10
//     for (unsigned int q = 0; q < fe_face0.n_quadrature_points; ++q)
//       {
//         const Tensor<1, dim> normal        = fe_face1.normal_vector(q);
//         const double         interface_JxW = fe_face0.JxW(q);
//         {
//           for (unsigned int i = 0; i < fe_face1.dofs_per_cell; ++i)
//             {
//               for (unsigned int j = 0; j < fe_face0.dofs_per_cell; ++j)
//                 {
//                   if (fe_face1.dofs_per_cell == 8 * 2 &&
//                       fe_face0.dofs_per_cell == 4 * 2)
//                     {
//                       if (i % 2 == reminder)
//                         {
//                           A_10(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face1.shape_grad(i, q) * normal *
//                             fe_face0.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else if (fe_face1.dofs_per_cell == 4 * 2 &&
//                            fe_face0.dofs_per_cell == 8 * 2)
//                     {
//                       if (j % 2 == reminder)
//                         {
//                           A_10(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face1.shape_grad(i, q) * normal *
//                             fe_face0.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else if (fe_face1.dofs_per_cell == 8 * 2 &&
//                            fe_face0.dofs_per_cell == 8 * 2)
//                     {
//                       if (i % 2 == reminder && j % 2 == reminder)
//                         {
//                           A_10(i, j) +=
//                             -.5 * ghost_parameter * cell_side_length * normal
//                             * fe_face1.shape_grad(i, q) * normal *
//                             fe_face0.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                 }
//             }
//         }
//       }
//     // A11
//     for (unsigned int q = 0; q < fe_face1.n_quadrature_points; ++q)
//       {
//         const Tensor<1, dim> normal        = fe_face1.normal_vector(q);
//         const double         interface_JxW = fe_face0.JxW(q);
//         {
//           for (unsigned int i = 0; i < fe_face1.dofs_per_cell; ++i)
//             {
//               for (unsigned int j = 0; j < fe_face1.dofs_per_cell; ++j)
//                 {
//                   if (fe_face1.dofs_per_cell == 8 * 2)
//                     {
//                       if (i % 2 == reminder && j % 2 == reminder)
//                         {
//                           A_11(i, j) +=
//                             .5 * ghost_parameter * cell_side_length * normal
//                             * fe_face1.shape_grad(i, q) * normal *
//                             fe_face1.shape_grad(j, q) * interface_JxW;
//                         }
//                     }
//                   else
//                     {
//                       A_11(i, j) += .5 * ghost_parameter * cell_side_length *
//                                     normal * fe_face1.shape_grad(i, q) *
//                                     normal * fe_face1.shape_grad(j, q) *
//                                     interface_JxW;
//                     }
//                 }
//             }
//         }
//       }

//     stiffness_matrix.add(local_dofs0_indices, A_00);
//     stiffness_matrix.add(local_dofs0_indices, local_dofs1_indices, A_01);
//     stiffness_matrix.add(local_dofs1_indices, local_dofs0_indices, A_10);
//     stiffness_matrix.add(local_dofs1_indices, A_11);
//   }

//   template <int dim>
//   bool LaplaceSolver<dim>::face_has_ghost_penalty(
//     const typename Triangulation<dim>::active_cell_iterator &cell,
//     const unsigned int                                       face_index)
//     const
//   {
//     if (cell->at_boundary(face_index))
//       return false;

//     const NonMatching::LocationToLevelSet cell_location =
//       mesh_classifier.location_to_level_set(cell);

//     const NonMatching::LocationToLevelSet neighbor_location =
//       mesh_classifier.location_to_level_set(cell->neighbor(face_index));

//     if (cell_location == NonMatching::LocationToLevelSet::intersected &&
//         neighbor_location != NonMatching::LocationToLevelSet::outside)
//       return true;

//     if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
//         cell_location != NonMatching::LocationToLevelSet::outside)
//       return true;

//     return false;
//   }

//   template <int dim>
//   bool LaplaceSolver<dim>::face_has_ghost_penalty_outside(
//     const typename Triangulation<dim>::active_cell_iterator &cell,
//     const unsigned int                                       face_index)
//     const
//   {
//     if (cell->at_boundary(face_index))
//       return false;

//     const NonMatching::LocationToLevelSet cell_location =
//       mesh_classifier.location_to_level_set(cell);

//     const NonMatching::LocationToLevelSet neighbor_location =
//       mesh_classifier.location_to_level_set(cell->neighbor(face_index));

//     if (cell_location == NonMatching::LocationToLevelSet::intersected &&
//         neighbor_location != NonMatching::LocationToLevelSet::inside)
//       return true;

//     if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
//         cell_location != NonMatching::LocationToLevelSet::inside)
//       return true;

//     return false;
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::assemble_system()
//   {
//     std::cout << "Assembling" << std::endl;

//     const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
//     FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
//     Vector<double>     local_rhs(n_dofs_per_cell);
//     std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

//     // std::cout << "nDoFs per intersected cell: "
//     //           << fe_collection[1].dofs_per_cell << std::endl;
//     const unsigned int n_dofs_per_intersected_cell =
//       fe_collection[1].dofs_per_cell;
//     FullMatrix<double> local_stiffness_surf(n_dofs_per_intersected_cell,
//                                             n_dofs_per_intersected_cell);
//     Vector<double>     local_rhs_surf(n_dofs_per_intersected_cell);
//     std::vector<types::global_dof_index> local_dof_indices_surf(
//       n_dofs_per_intersected_cell);

//     const double ghost_parameter   = 0.5;
//     const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

//     const QGauss<dim - 1> face_quadrature(fe_degree + 1);
//     const QGauss<1>       quadrature_1D(fe_degree + 1);

//     NonMatching::RegionUpdateFlags region_update_flags;
//     region_update_flags.inside = update_values | update_gradients |
//                                  update_JxW_values | update_quadrature_points;
//     region_update_flags.outside = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points;
//     region_update_flags.surface = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points |
//                                   update_normal_vectors;

//     NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
//                                                       quadrature_1D,
//                                                       region_update_flags,
//                                                       mesh_classifier,
//                                                       level_set_dof_handler,
//                                                       level_set);

//     RhsFunction<dim> rhs_function;

//     // Here I loop only on the inside
//     for (const auto &cell :
//          dof_handler.active_cell_iterators() |
//            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_in))
//       {
//         local_stiffness = 0.;
//         local_rhs       = 0.;

//         const double cell_side_length = cell->minimum_vertex_distance();
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
//           non_matching_fe_values.get_inside_fe_values();

//         // std::cout << inside_fe_values->shape_grad(0, 1) << std::endl;

//         if (inside_fe_values)
//           for (const unsigned int q :
//                inside_fe_values->quadrature_point_indices())
//             {
//               const Point<dim> &point =
//               inside_fe_values->quadrature_point(q); for (const unsigned int
//               i : inside_fe_values->dof_indices())
//                 {
//                   for (const unsigned int j :
//                   inside_fe_values->dof_indices())
//                     {
//                       local_stiffness(i, j) +=
//                         inside_fe_values->shape_grad(i, q) *
//                         inside_fe_values->shape_grad(j, q) *
//                         inside_fe_values->JxW(q);
//                     }
//                   local_rhs(i) += rhs_function.value(point) *
//                                   inside_fe_values->shape_value(i, q) *
//                                   inside_fe_values->JxW(q);
//                 }
//             }

//         cell->get_dof_indices(local_dof_indices);

//         stiffness_matrix.add(local_dof_indices, local_stiffness);
//         rhs.add(local_dof_indices, local_rhs);

//         for (unsigned int f : cell->face_indices())
//           {
//             if (face_has_ghost_penalty(cell, f))
//               {
//                 distribute_penalty_terms(cell,
//                                          f,
//                                          ghost_parameter,
//                                          cell_side_length,
//                                          0); // reminder == 0
//               }
//           }
//       }

//     // Solve on outside
//     for (const auto &cell :
//          dof_handler.active_cell_iterators() |
//            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_out))
//       {
//         local_stiffness = 0.;
//         local_rhs       = 0.;

//         const double cell_side_length = cell->minimum_vertex_distance();
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (outside_fe_values)
//           {
//             for (const unsigned int q :
//                  outside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   outside_fe_values->quadrature_point(q);
//                 for (const unsigned int i : outside_fe_values->dof_indices())
//                   {
//                     for (const unsigned int j :
//                          outside_fe_values->dof_indices())
//                       {
//                         local_stiffness(i, j) +=
//                           outside_fe_values->shape_grad(i, q) *
//                           outside_fe_values->shape_grad(j, q) *
//                           outside_fe_values->JxW(q);
//                       }
//                     local_rhs(i) += rhs_function.value(point) *
//                                     outside_fe_values->shape_value(i, q) *
//                                     outside_fe_values->JxW(q);
//                   }
//               }
//           }

//         cell->get_dof_indices(local_dof_indices);
//         constraints.distribute_local_to_global(
//           local_stiffness, local_rhs, local_dof_indices, stiffness_matrix,
//           rhs);

//         for (unsigned int f : cell->face_indices())
//           {
//             if (face_has_ghost_penalty_outside(cell, f))
//               {
//                 distribute_penalty_terms(cell,
//                                          f,
//                                          ghost_parameter,
//                                          cell_side_length,
//                                          1); // reminder == 1
//               }
//           }
//       }

//     // Solve on intersection
//     for (const auto &cell : dof_handler.active_cell_iterators() |
//                               IteratorFilters::ActiveFEIndexEqualTo(
//                                 ActiveFEIndex::sol_intersection))
//       {
//         local_stiffness_surf = 0.;
//         local_rhs_surf       = 0.;

//         const double cell_side_length = cell->minimum_vertex_distance();
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
//           non_matching_fe_values.get_inside_fe_values();

//         if (inside_fe_values)
//           {
//             for (const unsigned int q :
//                  inside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                 inside_fe_values->quadrature_point(q); for (const unsigned
//                 int i : inside_fe_values->dof_indices())
//                   {
//                     for (const unsigned int j :
//                     inside_fe_values->dof_indices())
//                       {
//                         if (i % 2 == 0 && j % 2 == 0)
//                           {
//                             local_stiffness_surf(i, j) +=
//                               inside_fe_values->shape_grad(i, q) *
//                               inside_fe_values->shape_grad(j, q) *
//                               inside_fe_values->JxW(q);
//                           }
//                       }

//                     if (i % 2 == 0)
//                       {
//                         local_rhs_surf(i) +=
//                           rhs_function.value(point) *
//                           inside_fe_values->shape_value(i, q) *
//                           inside_fe_values->JxW(q);
//                       }
//                   }
//               }
//           }

//         for (unsigned int f : cell->face_indices())
//           {
//             if (face_has_ghost_penalty(cell, f))
//               {
//                 distribute_penalty_terms(cell,
//                                          f,
//                                          ghost_parameter,
//                                          cell_side_length,
//                                          0); // reminder == 0
//               }
//             if (face_has_ghost_penalty_outside(cell, f))
//               {
//                 distribute_penalty_terms(cell,
//                                          f,
//                                          ghost_parameter,
//                                          cell_side_length,
//                                          1); // reminder == 1
//               }
//           }

//         const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (outside_fe_values)
//           {
//             for (const unsigned int q :
//                  outside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   outside_fe_values->quadrature_point(q);
//                 for (const unsigned int i : outside_fe_values->dof_indices())
//                   {
//                     for (const unsigned int j :
//                          outside_fe_values->dof_indices())
//                       {
//                         if (i % 2 == 1 && j % 2 == 1)
//                           {
//                             local_stiffness_surf(i, j) +=
//                               outside_fe_values->shape_grad(i, q) *
//                               outside_fe_values->shape_grad(j, q) *
//                               outside_fe_values->JxW(q);
//                           }
//                       }
//                     if (i % 2 == 1)
//                       {
//                         local_rhs_surf(i) +=
//                           rhs_function.value(point) *
//                           outside_fe_values->shape_value(i, q) *
//                           outside_fe_values->JxW(q);
//                       }
//                   }
//               }
//           }

//         const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
//           &surface_fe_values =
//           non_matching_fe_values.get_surface_fe_values();
//         if (surface_fe_values)
//           {
//             for (const unsigned int q :
//                  surface_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   surface_fe_values->quadrature_point(q);
//                 const Tensor<1, dim> &normal =
//                   surface_fe_values->normal_vector(q);
//                 for (const unsigned int i : surface_fe_values->dof_indices())
//                   {
//                     for (const unsigned int j :
//                          surface_fe_values->dof_indices())
//                       {
//                         if (i % 2 == 0 && j % 2 == 0)
//                           {
//                             Assert(fe_collection[1]
//                                          .system_to_component_index(i)
//                                          .first == 0 &&
//                                      fe_collection[1]
//                                          .system_to_component_index(j)
//                                          .first == 0,
//                                    ExcMessage("Component mismatch!"));
//                             local_stiffness_surf(i, j) +=
//                               (-normal * surface_fe_values->shape_grad(i, q)
//                               *
//                                  surface_fe_values->shape_value(j, q) +
//                                -normal * surface_fe_values->shape_grad(j, q)
//                                *
//                                  surface_fe_values->shape_value(i, q) +
//                                nitsche_parameter / cell_side_length *
//                                  surface_fe_values->shape_value(i, q) *
//                                  surface_fe_values->shape_value(j, q)) *
//                               surface_fe_values->JxW(q);
//                           }
//                         else if (i % 2 == 1 && j % 2 == 1)
//                           {
//                             Assert(fe_collection[1]
//                                          .system_to_component_index(i)
//                                          .first == 1 &&
//                                      fe_collection[1]
//                                          .system_to_component_index(j)
//                                          .first == 1,
//                                    ExcMessage("Component mismatch!"));
//                             local_stiffness_surf(i, j) +=
//                               (normal * surface_fe_values->shape_grad(i, q) *
//                                  surface_fe_values->shape_value(j, q) +
//                                normal * surface_fe_values->shape_grad(j, q) *
//                                  surface_fe_values->shape_value(i, q) +
//                                nitsche_parameter / cell_side_length *
//                                  surface_fe_values->shape_value(i, q) *
//                                  surface_fe_values->shape_value(j, q)) *
//                               surface_fe_values->JxW(q);
//                           }
//                       }

//                     if (i % 2 == 0)
//                       {
//                         // std::cout
//                         //   << "Test: "
//                         //   << surface_fe_values->shape_value_component(i,
//                         q,
//                         //   1)
//                         //   << std::endl;
//                         local_rhs_surf(i) +=
//                           AnalyticalSolution<dim>().value(point) *
//                           (nitsche_parameter / cell_side_length *
//                              surface_fe_values->shape_value(i, q) -
//                            normal * surface_fe_values->shape_grad(i, q)) *
//                           surface_fe_values->JxW(q);
//                       }
//                     else
//                       {
//                         local_rhs_surf(i) +=
//                           AnalyticalSolution<dim>().value(point) *
//                           (nitsche_parameter / cell_side_length *
//                              surface_fe_values->shape_value(i, q) +
//                            normal * surface_fe_values->shape_grad(i, q)) *
//                           surface_fe_values->JxW(q);
//                       }
//                   }
//               }
//           }

//         cell->get_dof_indices(local_dof_indices_surf);

//         stiffness_matrix.add(local_dof_indices_surf, local_stiffness_surf);
//         rhs.add(local_dof_indices_surf, local_rhs_surf);
//       }
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::solve()
//   {
//     std::cout << "Solving system" << std::endl;

//     const unsigned int max_iterations = solution.size();
//     SolverControl      solver_control(max_iterations);
//     SolverCG<>         solver(solver_control);
//     solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
//     std::cout << "Solved in " << solver_control.last_step() << " iterations."
//               << std::endl;
//     constraints.distribute(solution);
//   }

//   template <int dim>
//   void LaplaceSolver<dim>::output_results() const
//   {
//     std::cout << "Writing vtu file" << std::endl;

//     DataOut<dim> data_out;
//     data_out.add_data_vector(dof_handler, solution, "solution_sphere_3D");
//     data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

//     data_out.set_cell_selection(
//       [this](const typename Triangulation<dim>::cell_iterator &cell) {
//         return cell->is_active() &&
//                mesh_classifier.location_to_level_set(cell) !=
//                  NonMatching::LocationToLevelSet::outside;
//       });

//     data_out.build_patches();
//     std::ofstream output_inside_inter("cutFEM_inside_intersected.vtu");
//     data_out.write_vtu(output_inside_inter);

//     data_out.set_cell_selection(
//       [this](const typename Triangulation<dim>::cell_iterator &cell) {
//         return cell->is_active() &&
//                mesh_classifier.location_to_level_set(cell) ==
//                  NonMatching::LocationToLevelSet::outside;
//       });

//     data_out.build_patches();
//     std::ofstream output_outside("cutFEM_outside.vtu");
//     data_out.write_vtu(output_outside);
//   }

//   template <int dim>
//   double LaplaceSolver<dim>::compute_L2_error_from_inside() const
//   {
//     std::cout << "Computing L2 error from inside" << std::endl;

//     const QGauss<1> quadrature_1D(fe_degree + 1);

//     NonMatching::RegionUpdateFlags region_update_flags;
//     region_update_flags.inside =
//       update_values | update_JxW_values | update_quadrature_points;
//     region_update_flags.surface = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points |
//                                   update_normal_vectors;

//     NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
//                                                       quadrature_1D,
//                                                       region_update_flags,
//                                                       mesh_classifier,
//                                                       level_set_dof_handler,
//                                                       level_set);

//     const AnalyticalSolution<dim> analytical_solution;
//     double                        error_L2_squared = 0;
//     FEValuesExtractors::Scalar    interior(0);

//     for (const auto &cell :
//          dof_handler.active_cell_iterators() |
//            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_in))
//       {
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &fe_values =
//           non_matching_fe_values.get_inside_fe_values();

//         if (fe_values)
//           {
//             std::vector<double>
//             solution_values(fe_values->n_quadrature_points);
//             (*fe_values)[interior].get_function_values(solution,
//                                                        solution_values);

//             for (const unsigned int q :
//             fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point = fe_values->quadrature_point(q);
//                 const double      error_at_point =
//                   solution_values[q] - analytical_solution.value(point);
//                 error_L2_squared +=
//                   std::pow(error_at_point, 2) * fe_values->JxW(q);
//               }
//           }
//       }

//     for (const auto &cell : dof_handler.active_cell_iterators() |
//                               IteratorFilters::ActiveFEIndexEqualTo(
//                                 ActiveFEIndex::sol_intersection))
//       {
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
//           non_matching_fe_values.get_inside_fe_values();

//         if (inside_fe_values)
//           {
//             std::vector<double> solution_values(
//               inside_fe_values->n_quadrature_points);
//             (*inside_fe_values)[interior].get_function_values(solution,
//                                                               solution_values);

//             for (const unsigned int q :
//                  inside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                 inside_fe_values->quadrature_point(q); const double
//                 error_at_point =
//                   solution_values[q] - analytical_solution.value(point);
//                 error_L2_squared +=
//                   std::pow(error_at_point, 2) * inside_fe_values->JxW(q);
//               }
//           }

//         const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
//           &surface_fe_values =
//           non_matching_fe_values.get_surface_fe_values();
//         if (surface_fe_values)
//           {
//             std::vector<double> solution_values(
//               surface_fe_values->n_quadrature_points);
//             (*surface_fe_values)[interior].get_function_values(solution,
//                                                                solution_values);

//             for (const unsigned int q :
//                  surface_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   surface_fe_values->quadrature_point(q);
//                 const double error_at_point =
//                   solution_values[q] - analytical_solution.value(point);
//                 error_L2_squared +=
//                   std::pow(error_at_point, 2) * surface_fe_values->JxW(q);
//               }
//           }
//       }

//     return std::sqrt(error_L2_squared);
//   }

//   template <int dim>
//   double LaplaceSolver<dim>::compute_L2_error_from_outside() const
//   {
//     std::cout << "Computing L2 error from outside" << std::endl;

//     const QGauss<1> quadrature_1D(fe_degree + 1);

//     NonMatching::RegionUpdateFlags region_update_flags;

//     region_update_flags.outside =
//       update_values | update_JxW_values | update_quadrature_points;
//     region_update_flags.surface = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points |
//                                   update_normal_vectors;

//     NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
//                                                       quadrature_1D,
//                                                       region_update_flags,
//                                                       mesh_classifier,
//                                                       level_set_dof_handler,
//                                                       level_set);

//     const AnalyticalSolution<dim> analytical_solution;
//     double                        error_L2_squared = 0.;
//     FEValuesExtractors::Scalar    exterior(1);

//     for (const auto &cell :
//          dof_handler.active_cell_iterators() |
//            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_out))
//       {
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (fe_values)
//           {
//             std::vector<double>
//             solution_values(fe_values->n_quadrature_points);
//             (*fe_values)[exterior].get_function_values(solution,
//                                                        solution_values);

//             for (const unsigned int q :
//             fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point = fe_values->quadrature_point(q);
//                 const double      error_at_point =
//                   solution_values[q] - analytical_solution.value(point);
//                 error_L2_squared +=
//                   std::pow(error_at_point, 2) * fe_values->JxW(q);
//               }
//           }
//       }

//     for (const auto &cell : dof_handler.active_cell_iterators() |
//                               IteratorFilters::ActiveFEIndexEqualTo(
//                                 ActiveFEIndex::sol_intersection))
//       {
//         non_matching_fe_values.reinit(cell);
//         const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (outside_fe_values)
//           {
//             std::vector<double> solution_values(
//               outside_fe_values->n_quadrature_points);
//             (*outside_fe_values)[exterior].get_function_values(solution,
//                                                                solution_values);

//             for (const unsigned int q :
//                  outside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   outside_fe_values->quadrature_point(q);
//                 const double error_at_point =
//                   solution_values[q] - analytical_solution.value(point);
//                 error_L2_squared +=
//                   std::pow(error_at_point, 2) * outside_fe_values->JxW(q);
//               }
//           }
//         /*
//                 const
//            std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
//                   &surface_fe_values =
//            non_matching_fe_values.get_surface_fe_values(); if
//            (surface_fe_values)
//                   {
//                     std::vector<double> solution_values(
//                       surface_fe_values->n_quadrature_points);
//                     (*surface_fe_values)[exterior].get_function_values(solution,
//                                                                        solution_values);

//                     for (const unsigned int q :
//                          surface_fe_values->quadrature_point_indices())
//                       {
//                         const Point<dim> &point =
//                           surface_fe_values->quadrature_point(q);
//                         const double error_at_point =
//                           solution_values[q] -
//                           analytical_solution.value(point);
//                         error_L2_squared +=
//                           std::pow(error_at_point, 2) *
//            surface_fe_values->JxW(q);
//                       }
//                   }*/
//       }

//     return std::sqrt(error_L2_squared);
//   }


//   template <int dim>
//   double LaplaceSolver<dim>::compute_H1_error_from_outside() const
//   {
//     std::cout << "Computing H1 error from outside" << std::endl;

//     const QGauss<1> quadrature_1D(fe_degree + 1);

//     NonMatching::RegionUpdateFlags region_update_flags;

//     region_update_flags.outside = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points;
//     region_update_flags.surface = update_values | update_gradients |
//                                   update_JxW_values | update_quadrature_points |
//                                   update_normal_vectors;

//     NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
//                                                       quadrature_1D,
//                                                       region_update_flags,
//                                                       mesh_classifier,
//                                                       level_set_dof_handler,
//                                                       level_set);

//     const AnalyticalSolution<dim> analytical_solution;
//     double                        error_H1_squared = 0.;
//     FEValuesExtractors::Scalar    exterior(1);

//     for (const auto &cell :
//          dof_handler.active_cell_iterators() |
//            IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::sol_out))
//       {
//         non_matching_fe_values.reinit(cell);

//         const std_cxx17::optional<FEValues<dim>> &fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (fe_values)
//           {
//             std::vector<Tensor<1, dim>> solution_gradients(
//               fe_values->n_quadrature_points);
//             (*fe_values)[exterior].get_function_gradients(solution,
//                                                           solution_gradients);

//             for (const unsigned int q :
//             fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point = fe_values->quadrature_point(q);
//                 const double      error_at_point =
//                   (analytical_solution.gradient(point) -
//                   solution_gradients[q])
//                     .norm_square();
//                 error_H1_squared += error_at_point * fe_values->JxW(q);
//               }
//           }
//       }

//     for (const auto &cell : dof_handler.active_cell_iterators() |
//                               IteratorFilters::ActiveFEIndexEqualTo(
//                                 ActiveFEIndex::sol_intersection))
//       {
//         non_matching_fe_values.reinit(cell);
//         const std_cxx17::optional<FEValues<dim>> &outside_fe_values =
//           non_matching_fe_values.get_outside_fe_values();

//         if (outside_fe_values)
//           {
//             std::vector<Tensor<1, dim>> solution_gradients(
//               outside_fe_values->n_quadrature_points);
//             (*outside_fe_values)[exterior].get_function_gradients(
//               solution, solution_gradients);

//             for (const unsigned int q :
//                  outside_fe_values->quadrature_point_indices())
//               {
//                 const Point<dim> &point =
//                   outside_fe_values->quadrature_point(q);
//                 const double error_at_point =
//                   (analytical_solution.gradient(point) -
//                   solution_gradients[q])
//                     .norm_square();
//                 error_H1_squared += error_at_point *
//                 outside_fe_values->JxW(q);
//               }
//           }

//         /*
//                 const
//            std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
//                   &surface_fe_values =
//            non_matching_fe_values.get_surface_fe_values(); if
//            (surface_fe_values)
//                   {
//                     std::vector<double> solution_values(
//                       surface_fe_values->n_quadrature_points);
//                     (*surface_fe_values)[exterior].get_function_values(solution,
//                                                                        solution_values);

//                     std::vector<Tensor<1, dim>> solution_gradients(
//                       surface_fe_values->n_quadrature_points);
//                     (*surface_fe_values)[exterior].get_function_gradients(
//                       solution, solution_gradients);

//                     for (const unsigned int q :
//                          surface_fe_values->quadrature_point_indices())
//                       {
//                         const Point<dim> &point =
//                           surface_fe_values->quadrature_point(q);
//                         const double error_at_point =
//                           (analytical_solution.gradient(point) -
//            solution_gradients[q]) .norm_square(); error_H1_squared +=
//            error_at_point * surface_fe_values->JxW(q);
//                       }
//                   }*/
//       }
//     const double sqrtL2error = compute_L2_error_from_outside();
//     return std::sqrt(error_H1_squared + sqrtL2error);
//   }


//   template <int dim>
//   void LaplaceSolver<dim>::run()
//   {
//     ConvergenceTable   convergence_table;
//     const unsigned int n_refinements = 5;

//     make_grid();
//     for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
//       {
//         std::cout << "Refinement cycle " << cycle << std::endl;
//         triangulation.refine_global(1);
//         setup_discrete_level_set();
//         std::cout << "Classifying cells" << std::endl;
//         mesh_classifier.reclassify();
//         distribute_dofs();
//         initialize_matrices();
//         assemble_system();
//         solve();
//         // if (cycle == 3)
//         output_results();
//         // const double error_L2 = compute_L2_error_from_inside();
//         const double error_L2 = compute_L2_error_from_outside();
//         const double error_H1 = compute_H1_error_from_outside();
//         const double cell_side_length =
//           triangulation.begin_active()->minimum_vertex_distance();

//         convergence_table.add_value("Cycle", cycle);
//         convergence_table.add_value("Mesh size", cell_side_length);
//         convergence_table.add_value("dofs", dof_handler.n_dofs());
//         convergence_table.add_value("L2-Error", error_L2);
//         convergence_table.add_value("H1-Error", error_H1);

//         convergence_table.set_scientific("L2-Error", true);
//         convergence_table.set_scientific("H1-Error", true);
//         convergence_table.evaluate_convergence_rates(
//           "L2-Error", ConvergenceTable::reduction_rate_log2);
//         convergence_table.evaluate_convergence_rates(
//           "H1-Error", ConvergenceTable::reduction_rate_log2);


//         std::cout << std::endl;
//         convergence_table.write_text(std::cout);
//         std::cout << std::endl;
//       }
//   }

// } // namespace Step85

// int main()
// {
//   const int dim = 3;

//   Step85::LaplaceSolver<dim> laplace_solver;
//   laplace_solver.run();
// }
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
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>


#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/base/timer.h>
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
#include <deal.II/cgal/triangulation.h>
#include "../exact_non_matching/NonMatching_utilities.h"


namespace Step85
{
  using namespace dealii;

  const double Cx = .5;
  const double Cy = .5;
  const double Cz = .5;
  const double R  = .3;

  class ImplicitFunction : public Function<3>
  {
  public:
    virtual double value(const Point<3> &   p,
                         const unsigned int component = 0) const override
    {
      // const double xc = 0.2 / std::sqrt(20.);
      // const double yc = 0.2 / std::sqrt(20.);
      // const double w  = 12.;
      // const double R = .3;
      // const double r = .1;
      const double x = p[0];
      const double y = p[1];
      const double z = p[2];
      // const double oscillating_term =
      //   (1. - 2. * (y * y) / (x * x + y * y)) *
      //   (1. - 16. * (x * x * y * y) / ((x * x + y * y) * (x * x + y * y)));
      // return std::sqrt(p[0]*p[0] + p[1]*p[1]) - r*(p[0]*p[0]*p[0]
      // - 3.*p[0]*p[1]*p[1])*std::pow(p[0]*p[0] + p[1]*p[1],-3./2.) - R;
      // return std::sqrt(x * x + y * y) - r * oscillating_term - R;

      return std::sqrt((x - Cx) * (x - Cx) + (y - Cy) * (y - Cy) +
                       (z - Cz) * (z - Cz)) -
             R;
    }
  };


  template <typename Tester>
  class Discrete_signed_distance : public Function<3>
  {
    using K       = CGAL::Exact_predicates_inexact_constructions_kernel;
    using Point_3 = K::Point_3;

  public:
    Discrete_signed_distance(Tester tester)
      : is_in_tester(tester)
    {
      is_in_tester(K::Point_3(Cx, Cy, Cz));
    };

    virtual double value(const Point<3> &   p,
                         const unsigned int component = 0) const override
    {
      // const double x = p[0];
      // const double y = p[1];
      // const double z = p[2];


      // return std::sqrt((x - Cx) * (x - Cx) + (y - Cy) * (y - Cy) +
      //                  (z - Cz) * (z - Cz)) -
      //        R;

      // std::cout << is_in_tester(K::Point_3(Cx, Cy, Cz)) << std::endl;

      if (*is_in_tester(
            CGALWrappers::dealii_point_to_cgal_point<Point_3, 3>(p)))
        return -1.;
      else
        return +1.;
    }

    Tester is_in_tester;
  };


  // void create_cgal_mesh_domain()
  // {
  //   ImplicitFunction dealii_implicit_function;
  //   auto             cgal_implicit_function = [&](const Point_3 &p) {
  //     return NumberType(
  //       dealii_implicit_function.value(Point<3>(p.x(), p.y(), p.z())));
  //   };

  //   Mesh_domain domain = Mesh_domain::create_implicit_mesh_domain(
  //     cgal_implicit_function, K::Sphere_3(Point_3(Cx, Cy, Cz), 10));

  //   auto                                test =
  //     Mesh_domain::Is_in_domain(domain).operator()(Point_3(Cx, Cy, Cz));
  //   auto                                test_false =
  //     Mesh_domain::Is_in_domain(domain).operator()(Point_3(5., 6., 8.));
  //   MeshDomain::Is_in_domain(domain) is_in_tester;
  //   auto other_test = is_in_tester(Point_3(5., 6., 8.));


  //   std::cout << "Testing: " << test << std::endl;
  //   std::cout << "Testing: " << *test_false << std::endl;
  //   std::cout << "Testing: " << *other_test << std::endl;
  // }


  template <int dim>
  class LaplaceSolver
  {
  public:
    LaplaceSolver();

    void run();

  private:
    using K          = CGAL::Exact_predicates_inexact_constructions_kernel;
    using NumberType = K::FT;
    using Point_3    = K::Point_3;
    using Sphere_3   = K::Sphere_3;

    using Mesh_domain   = CGAL::Labeled_mesh_domain_3<K>;
    using Tr            = CGAL::Mesh_triangulation_3<Mesh_domain,
                                          CGAL::Default,
                                          CGALWrappers::ConcurrencyTag>::type;
    using C3t3          = CGAL::Mesh_complex_3_in_triangulation_3<Tr, int, int>;
    using Mesh_criteria = CGAL::Mesh_criteria_3<Tr>;

    void make_grid();

    void setup_discrete_level_set(const unsigned int cycle);

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve();

    void output_results() const;

    double compute_L2_error_from_inside() const;

    double compute_L2_error_from_outside() const;

    double compute_H1_error_from_inside() const;

    double compute_H1_error_from_outside() const;

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

    parallel::shared::Triangulation<dim> triangulation;
    Triangulation<2, 3>                  embedded_tria;

    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double>  level_set;

    DoFHandler<dim> dof_handler;

    FESystem<dim>         fe_in;
    FESystem<dim>         fe_surf;
    FESystem<dim>         fe_out;
    hp::FECollection<dim> fe_collection;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern                  sparsity_pattern;
    PETScWrappers::MPI::SparseMatrix stiffness_matrix;
    PETScWrappers::MPI::Vector       rhs;
    PETScWrappers::MPI::Vector       solution;
    MPI_Comm                         mpi_communicator;

    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    mutable TimerOutput timer;
  };

  template <int dim>
  LaplaceSolver<dim>::LaplaceSolver()
    : fe_degree(1)
    , triangulation(MPI_COMM_WORLD)
    , fe_level_set(fe_degree)
    , level_set_dof_handler(triangulation)
    , dof_handler(triangulation)
    , fe_in(FE_Q<dim>(fe_degree), 1, FE_Nothing<dim>(), 1)
    , fe_surf(FE_Q<dim>(fe_degree), 1, FE_Q<dim>(fe_degree), 1)
    , fe_out(FE_Nothing<dim>(), 1, FE_Q<dim>(fe_degree), 1)
    , mesh_classifier(level_set_dof_handler, level_set)
    , mpi_communicator(MPI_COMM_WORLD)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator))
    , timer(std::cout,
            TimerOutput::every_call_and_summary,
            TimerOutput::cpu_times)
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
    triangulation.refine_global(2);
  }

  template <int dim>
  void LaplaceSolver<dim>::setup_discrete_level_set(const unsigned int cycle)
  {
    std::cout << "Setting up discrete level set function" << std::endl;
    TimerOutput::Scope timer_section(timer, "Setup discrete level set");
    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    const Functions::SignedDistance::Sphere<dim> signed_distance_sphere(
      {Cx, Cy, Cz}, R);
    // ImplicitFunction implicit_function;

    // using CGALMesh = CGAL::Surface_mesh<Point_3>;
    // CGALMesh surface_mesh;

    // // Triangulation<2, 3> tria_out;
    // // if (cycle == 0)
    // //   {
    // //     GridGenerator::hyper_ball(embedded_tria, {Cx, Cy, Cz}, R);
    // //     embedded_tria.refine_global(3);
    // //   }
    // // else
    // //   {
    // //     embedded_tria.refine_global(1);
    // //   }

    // // Move to CGAL surfaces
    // dealii::CGALWrappers::dealii_tria_to_cgal_surface_mesh(embedded_tria,
    //                                                        surface_mesh);
    // // CGAL::Polygon_mesh_processing::stitch_borders(surface_mesh);
    // CGAL::Polygon_mesh_processing::triangulate_faces(surface_mesh);
    // // Now back to deal.II
    // // dealii::CGALWrappers::cgal_surface_mesh_to_dealii_triangulation(
    // //   surface_mesh, tria_out);


    // // auto cgal_implicit_function = [&](const Point_3 &p) {
    // //   return NumberType(implicit_function.value(Point<3>(p.x(), p.y(),
    // //   p.z())));
    // // };

    // // Mesh_domain domain = Mesh_domain::create_implicit_mesh_domain(
    // //   cgal_implicit_function, K::Sphere_3(Point_3(Cx, Cy, Cz), 10));
    // // const auto &is_in_tester = Mesh_domain::Is_in_domain(domain);
    // // Discrete_signed_distance<Mesh_domain::Is_in_domain>
    // //   discrete_signed_distance(is_in_tester);

    // typedef CGAL::Polyhedral_mesh_domain_3<CGALMesh, K> Mesh_domain_poly;
    // Mesh_domain_poly                                    domain(surface_mesh);
    // const auto &is_in_tester = Mesh_domain_poly::Is_in_domain(domain);

    // Discrete_signed_distance<Mesh_domain_poly::Is_in_domain>
    //   discrete_signed_distance(is_in_tester);

    // std::ofstream out_name_spheres("hyper_sphere_test.vtk");
    // GridOut       go;
    // go.write_vtk(tria_out, out_name_spheres);


    if (cycle == 0)
      {
        GridGenerator::hyper_sphere(embedded_tria, {Cx, Cy, Cz}, R);
        embedded_tria.refine_global(5);
      }
    else
      {
        embedded_tria.refine_global(1);
      }
    GridTools::Cache<2, 3> cache(embedded_tria);
    auto                   tree = cache.get_cell_bounding_boxes_rtree();



    // Triangulation<2, 3> tria_out;
    // Move to CGAL surfaces
    NonMatchingUtilities::CGALMesh surface_mesh;
    dealii::CGALWrappers::dealii_tria_to_cgal_surface_mesh(embedded_tria,
                                                           surface_mesh);
    // CGAL::Polygon_mesh_processing::stitch_borders(surface_mesh);
    CGAL::Polygon_mesh_processing::triangulate_faces(surface_mesh);
    // // Now back to deal.II
    // dealii::CGALWrappers::cgal_surface_mesh_to_dealii_triangulation(surface_mesh,
    //                                                                 tria_out);


    typedef CGAL::Polyhedral_mesh_domain_3<NonMatchingUtilities::CGALMesh,
                                           NonMatchingUtilities::CGALKernel>
                     Mesh_domain_poly;
    Mesh_domain_poly domain(surface_mesh);
    auto             is_in_tester = Mesh_domain_poly::Is_in_domain(domain);


    NonMatchingUtilities::
      DiscreteLevelSet<3, decltype(tree), Mesh_domain_poly::Is_in_domain>
        discrete_level_set(&tree, is_in_tester);


    VectorTools::interpolate(level_set_dof_handler,
                             discrete_level_set,
                             //  signed_distance_sphere,
                             level_set);
  }

  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim> & point,
                 const unsigned int component = 0) const override;

    Tensor<1, dim> gradient(const Point<dim> & point,
                            const unsigned int component = 0) const override;
  };

  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> & point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;
    const Point<3> xc{Cx, Cy, Cz}; // center of the sphere
    const double   r = (point - xc).norm();

    switch (dim)
      {
        case 2:
          return std::sin(2. * numbers::PI * point[0]) *
                 std::sin(2. * numbers::PI * point[1]);
        case 3:
          return std::sin(2. * numbers::PI * point[0]) *
                 std::sin(2. * numbers::PI * point[1]) *
                 std::sin(2. * numbers::PI * point[2]);
          // return r <= R ? 1. / R : 1. / r;
        default:
          Assert(false, ExcNotImplemented());
      }

    // 1. - 2. / dim * (point.norm_square() - 1.);
  }

  template <int dim>
  Tensor<1, dim>
  AnalyticalSolution<dim>::gradient(const Point<dim> & point,
                                    const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    const Point<3> xc{Cx, Cy, Cz};
    const double   r = (point - xc).norm();

    Tensor<1, dim> grad;
    switch (dim)
      {
        case 2:
          grad[0] = 2. * M_PI * std::cos(2. * M_PI * point[0]) *
                    std::sin(2. * M_PI * point[1]);
          grad[1] = 2. * M_PI * std::cos(2. * M_PI * point[1]) *
                    std::sin(2. * M_PI * point[0]);
          return grad;
          break;
        case 3:

          grad[0] = 2. * M_PI * std::cos(2. * M_PI * point[0]) *
                    std::sin(2. * M_PI * point[1]) *
                    std::sin(2. * M_PI * point[2]);
          grad[1] = 2. * M_PI * std::sin(2. * M_PI * point[0]) *
                    std::cos(2. * M_PI * point[1]) *
                    std::sin(2. * M_PI * point[2]);
          grad[2] = 2. * M_PI * std::sin(2. * M_PI * point[0]) *
                    std::sin(2. * M_PI * point[1]) *
                    std::cos(2. * M_PI * point[2]);
          // grad[0] = (r <= R) ? 0. : -(point[0] - Cx) / (std::pow(r *
          // r, 1.5)); grad[1] = (r <= R) ? 0. : -(point[1] - Cy) / (std::pow(r
          // * r, 1.5)); grad[2] = (r <= R) ? 0. : -(point[2] - Cz) /
          // (std::pow(r * r, 1.5));
          return grad;
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
  }



  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues()
      : Function<dim>(dim - 1)
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
          case 3:
            return AnalyticalSolution<dim>().value(p);
          default:
            Assert(false, ExcNotImplemented());
        }

    // return 0;
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
    switch (dim)
      {
        case 2:
          return 8. * numbers::PI * numbers::PI *
                 std::sin(2. * numbers::PI * p[0]) *
                 std::sin(2. * numbers::PI * p[1]);
        case 3:
          return 12. * numbers::PI * numbers::PI *
                 std::sin(2. * numbers::PI * p[0]) *
                 std::sin(2. * numbers::PI * p[1]) *
                 std::sin(2. * numbers::PI * p[2]);
          // return 0.;
        default:
          Assert(false, ExcMessage("Nonsense"));
      };
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
    TimerOutput::Scope timer_section(timer, "Distribute DoFs");
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
    std::vector<bool>                bools{true, true};
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             BoundaryValues<dim>(),
                                             constraints,
                                             fe_collection[0].component_mask(
                                               bools));
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
    TimerOutput::Scope timer_section(timer, "Initializing matrices");
    const auto         face_has_flux_coupling = [&](const auto &       cell,
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

    // stiffness_matrix.reinit(sparsity_pattern);
    // solution.reinit(dof_handler.n_dofs());
    // rhs.reinit(dof_handler.n_dofs());
    const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

    stiffness_matrix.reinit(locally_owned_dofs,
                            locally_owned_dofs,
                            sparsity_pattern,
                            mpi_communicator);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    rhs.reinit(locally_owned_dofs, mpi_communicator);
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
    Assert(!(fe_face0.dofs_per_cell == 8 && fe_face1.dofs_per_cell == 8),
           ExcMessage("They cannot have both 8 DoFs."));

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
                  if (fe_face0.dofs_per_cell == 8 * 2)
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
                  // std::cout
                  //   << "fe_face0.dofs_per_cell = " << fe_face0.dofs_per_cell
                  //   << std::endl;

                  // std::cout
                  //   << "fe_face1.dofs_per_cell = " << fe_face1.dofs_per_cell
                  //   << std::endl;

                  if (fe_face1.dofs_per_cell == 8 * 2 &&
                      fe_face0.dofs_per_cell == 4 * 2)
                    {
                      if (j % 2 == reminder)
                        {
                          A_01(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 4 * 2 &&
                           fe_face0.dofs_per_cell == 8 * 2)
                    {
                      if (i % 2 == reminder)
                        {
                          A_01(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face0.shape_grad(i, q) * normal *
                            fe_face1.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 8 * 2 &&
                           fe_face0.dofs_per_cell == 8 * 2)
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
                  if (fe_face1.dofs_per_cell == 8 * 2 &&
                      fe_face0.dofs_per_cell == 4 * 2)
                    {
                      if (i % 2 == reminder)
                        {
                          A_10(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 4 * 2 &&
                           fe_face0.dofs_per_cell == 8 * 2)
                    {
                      if (j % 2 == reminder)
                        {
                          A_10(i, j) +=
                            -.5 * ghost_parameter * cell_side_length * normal *
                            fe_face1.shape_grad(i, q) * normal *
                            fe_face0.shape_grad(j, q) * interface_JxW;
                        }
                    }
                  else if (fe_face1.dofs_per_cell == 8 * 2 &&
                           fe_face0.dofs_per_cell == 8 * 2)
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
                  if (fe_face1.dofs_per_cell == 8 * 2)
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
    TimerOutput::Scope timer_section(timer, "Assembling");
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
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

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

        for (unsigned int f : cell->face_indices())
          {
            if (face_has_ghost_penalty_outside(cell, f))
              {
                distribute_penalty_terms(cell,
                                         f,
                                         ghost_parameter,
                                         cell_side_length,
                                         1); // reminder == 1
              }
          }
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

        if (inside_fe_values)
          {
            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                for (const unsigned int i : inside_fe_values->dof_indices())
                  {
                    for (const unsigned int j : inside_fe_values->dof_indices())
                      {
                        if (i % 2 == 0 && j % 2 == 0)
                          {
                            local_stiffness_surf(i, j) +=
                              inside_fe_values->shape_grad(i, q) *
                              inside_fe_values->shape_grad(j, q) *
                              inside_fe_values->JxW(q);
                          }
                      }

                    if (i % 2 == 0)
                      {
                        local_rhs_surf(i) +=
                          rhs_function.value(point) *
                          inside_fe_values->shape_value(i, q) *
                          inside_fe_values->JxW(q);
                      }
                  }
              }
          }

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
            if (face_has_ghost_penalty_outside(cell, f))
              {
                distribute_penalty_terms(cell,
                                         f,
                                         ghost_parameter,
                                         cell_side_length,
                                         1); // reminder == 1
              }
          }

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
                        if (i % 2 == 1 && j % 2 == 1)
                          {
                            local_stiffness_surf(i, j) +=
                              outside_fe_values->shape_grad(i, q) *
                              outside_fe_values->shape_grad(j, q) *
                              outside_fe_values->JxW(q);
                          }
                      }
                    if (i % 2 == 1)
                      {
                        local_rhs_surf(i) +=
                          rhs_function.value(point) *
                          outside_fe_values->shape_value(i, q) *
                          outside_fe_values->JxW(q);
                      }
                  }
              }
          }

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
                        // std::cout
                        //   << "Test: "
                        //   << surface_fe_values->shape_value_component(i, q,
                        //   1)
                        //   << std::endl;
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
    stiffness_matrix.compress(VectorOperation::add);
    rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void LaplaceSolver<dim>::solve()
  {
    std::cout << "Solving system" << std::endl;
    TimerOutput::Scope timer_section(timer, "Solve");
    // const unsigned int max_iterations = solution.size();
    // SolverControl      solver_control(max_iterations);
    // SolverCG<>         solver(solver_control);
    // solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
    // std::cout << "Solved in " << solver_control.last_step() << " iterations."
    //           << std::endl;
    // constraints.distribute(solution);
    PETScWrappers::PreconditionBoomerAMG                 preconditioner;
    PETScWrappers::PreconditionBoomerAMG::AdditionalData data;
    data.symmetric_operator = true;
    preconditioner.initialize(stiffness_matrix, data);
    SolverControl           solver_control(solution.size(), 1e-12);
    PETScWrappers::SolverCG solver(solver_control);
    solver.solve(stiffness_matrix, solution, rhs, preconditioner);
    std::cout << "Solved in " << solver_control.last_step() << " iterations."
              << std::endl;
    constraints.distribute(solution);
  }

  template <int dim>
  void LaplaceSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution_sphere_3D");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) !=
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output_inside_inter("cutFEM_inside_intersected.vtu");
    data_out.write_vtu(output_inside_inter);

    data_out.set_cell_selection(
      [this](const typename Triangulation<dim>::cell_iterator &cell) {
        return cell->is_active() &&
               mesh_classifier.location_to_level_set(cell) ==
                 NonMatching::LocationToLevelSet::outside;
      });

    data_out.build_patches();
    std::ofstream output_outside("cutFEM_outside.vtu");
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

    const QGauss<1> quadrature_1D(2 * fe_degree + 1);

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
        /*
                const
           std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
                  &surface_fe_values =
           non_matching_fe_values.get_surface_fe_values(); if
           (surface_fe_values)
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
                          std::pow(error_at_point, 2) *
           surface_fe_values->JxW(q);
                      }
                  }*/
      }

    return std::sqrt(error_L2_squared);
  }



  template <int dim>
  double LaplaceSolver<dim>::compute_H1_error_from_inside() const
  {
    std::cout << "Computing H1 error from inside" << std::endl;

    const QGauss<1> quadrature_1D(2 * fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;

    region_update_flags.inside = update_values | update_gradients |
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

    const AnalyticalSolution<dim> analytical_solution;
    double                        error_H1_squared = 0.;
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
            std::vector<Tensor<1, dim>> solution_gradients(
              fe_values->n_quadrature_points);
            (*fe_values)[interior].get_function_gradients(solution,
                                                          solution_gradients);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  (analytical_solution.gradient(point) - solution_gradients[q])
                    .norm_square();
                error_H1_squared += error_at_point * fe_values->JxW(q);
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
            std::vector<Tensor<1, dim>> solution_gradients(
              inside_fe_values->n_quadrature_points);
            (*inside_fe_values)[interior].get_function_gradients(
              solution, solution_gradients);

            for (const unsigned int q :
                 inside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = inside_fe_values->quadrature_point(q);
                const double      error_at_point =
                  (analytical_solution.gradient(point) - solution_gradients[q])
                    .norm_square();
                error_H1_squared += error_at_point * inside_fe_values->JxW(q);
              }
          }

        /*
                const
           std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
                  &surface_fe_values =
           non_matching_fe_values.get_surface_fe_values(); if
           (surface_fe_values)
                  {
                    std::vector<double> solution_values(
                      surface_fe_values->n_quadrature_points);
                    (*surface_fe_values)[exterior].get_function_values(solution,
                                                                       solution_values);

                    std::vector<Tensor<1, dim>> solution_gradients(
                      surface_fe_values->n_quadrature_points);
                    (*surface_fe_values)[exterior].get_function_gradients(
                      solution, solution_gradients);

                    for (const unsigned int q :
                         surface_fe_values->quadrature_point_indices())
                      {
                        const Point<dim> &point =
                          surface_fe_values->quadrature_point(q);
                        const double error_at_point =
                          (analytical_solution.gradient(point) -
           solution_gradients[q]) .norm_square(); error_H1_squared +=
           error_at_point * surface_fe_values->JxW(q);
                      }
                  }*/
      }
    const double sqrtL2error = compute_L2_error_from_inside();
    return std::sqrt(error_H1_squared + sqrtL2error);
  }



  template <int dim>
  double LaplaceSolver<dim>::compute_H1_error_from_outside() const
  {
    std::cout << "Computing H1 error from outside" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;

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

    const AnalyticalSolution<dim> analytical_solution;
    double                        error_H1_squared = 0.;
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
            std::vector<Tensor<1, dim>> solution_gradients(
              fe_values->n_quadrature_points);
            (*fe_values)[exterior].get_function_gradients(solution,
                                                          solution_gradients);

            for (const unsigned int q : fe_values->quadrature_point_indices())
              {
                const Point<dim> &point = fe_values->quadrature_point(q);
                const double      error_at_point =
                  (analytical_solution.gradient(point) - solution_gradients[q])
                    .norm_square();
                error_H1_squared += error_at_point * fe_values->JxW(q);
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
            std::vector<Tensor<1, dim>> solution_gradients(
              outside_fe_values->n_quadrature_points);
            (*outside_fe_values)[exterior].get_function_gradients(
              solution, solution_gradients);

            for (const unsigned int q :
                 outside_fe_values->quadrature_point_indices())
              {
                const Point<dim> &point =
                  outside_fe_values->quadrature_point(q);
                const double error_at_point =
                  (analytical_solution.gradient(point) - solution_gradients[q])
                    .norm_square();
                error_H1_squared += error_at_point * outside_fe_values->JxW(q);
              }
          }

        /*
                const
           std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
                  &surface_fe_values =
           non_matching_fe_values.get_surface_fe_values(); if
           (surface_fe_values)
                  {
                    std::vector<double> solution_values(
                      surface_fe_values->n_quadrature_points);
                    (*surface_fe_values)[exterior].get_function_values(solution,
                                                                       solution_values);

                    std::vector<Tensor<1, dim>> solution_gradients(
                      surface_fe_values->n_quadrature_points);
                    (*surface_fe_values)[exterior].get_function_gradients(
                      solution, solution_gradients);

                    for (const unsigned int q :
                         surface_fe_values->quadrature_point_indices())
                      {
                        const Point<dim> &point =
                          surface_fe_values->quadrature_point(q);
                        const double error_at_point =
                          (analytical_solution.gradient(point) -
           solution_gradients[q]) .norm_square(); error_H1_squared +=
           error_at_point * surface_fe_values->JxW(q);
                      }
                  }*/
      }
    const double sqrtL2error = compute_L2_error_from_outside();
    return std::sqrt(error_H1_squared + sqrtL2error);
  }


  template <int dim>
  void LaplaceSolver<dim>::run()
  {
    ConvergenceTable   convergence_table;
    const unsigned int n_refinements = 5;

    make_grid();
    for (unsigned int cycle = 0; cycle <= n_refinements; cycle++)
      {
        {
          TimerOutput::Scope timer_section(timer,
                                           "Total time cycle " +
                                             std::to_string(cycle));

          std::cout << "Refinement cycle " << cycle << std::endl;
          triangulation.refine_global(1);
          setup_discrete_level_set(cycle);
          std::cout << "Classifying cells" << std::endl;

          {
            TimerOutput::Scope timer_section(timer, "Mesh classifier");

            mesh_classifier.reclassify();
          }
          distribute_dofs();
          initialize_matrices();
          assemble_system();
          solve();
          // if (cycle == 3)
        }
        output_results();
        const double error_L2_inside  = compute_L2_error_from_inside();
        const double error_L2_outside = compute_L2_error_from_outside();
        const double error_L2 = std::sqrt(error_L2_outside * error_L2_outside +
                                          error_L2_inside * error_L2_inside);

        const double error_H1_inside  = compute_H1_error_from_inside();
        const double error_H1_outside = compute_H1_error_from_outside();
        const double error_H1 = std::sqrt(error_H1_outside * error_H1_outside +
                                          error_H1_inside * error_H1_inside);
        const double cell_side_length =
          triangulation.begin_active()->minimum_vertex_distance();

        convergence_table.add_value("Cycle", cycle);
        convergence_table.add_value("Mesh size", cell_side_length);
        convergence_table.add_value("Cells",
                                    triangulation.n_global_active_cells());
        convergence_table.add_value("dofs", dof_handler.n_dofs());
        convergence_table.add_value("L2-Error", error_L2);
        convergence_table.add_value("H1-Error", error_H1);

        convergence_table.set_scientific("L2-Error", true);
        convergence_table.set_scientific("H1-Error", true);
        convergence_table.evaluate_convergence_rates(
          "L2-Error", "dofs", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.evaluate_convergence_rates(
          "H1-Error", "dofs", ConvergenceTable::reduction_rate_log2, dim);

        std::cout << std::endl;
        convergence_table.write_text(std::cout);
        std::cout << std::endl;
      }
  }

} // namespace Step85

int main(int argc, char *argv[])
{
  const int dim = 3;
  // Step85::create_cgal_mesh_domain();
  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  Step85::LaplaceSolver<dim>               laplace_solver;
  laplace_solver.run();
}