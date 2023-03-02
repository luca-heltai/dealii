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

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/cgal/triangulation.h>

#include <deal.II/dofs/dof_tools.h>


#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/tria.h>


#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <vector>

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/mesh_classifier.h>
#include "../exact_non_matching/NonMatching_utilities.h"
#include "../../tests/tests.h"


const double Cx = .5;
const double Cy = .5;
const double Cz = .5;
const double R  = .3;

using namespace dealii;

template <int dim>
struct TestPoints
{
  TestPoints() = default;

  void make_grid();

  void setup_discrete_level_set();

  // const unsigned int fe_degree;

  Triangulation<dim>          triangulation;
  Triangulation<dim - 1, dim> embedded_tria;

  // const FE_Q<dim> fe_level_set;
  // DoFHandler<dim> level_set_dof_handler;
  // Vector<double>  level_set;

  // DoFHandler<dim> dof_handler;

  // NonMatching::MeshClassifier<dim> mesh_classifier;
};

// template <int dim>
// TestPoints<dim>::TestPoints()
// // : fe_degree(1)
// // , fe_level_set(fe_degree)
// // , level_set_dof_handler(triangulation)
// // , dof_handler(triangulation)
// // , mesh_classifier(level_set_dof_handler, level_set)
// {}

template <int dim>
void TestPoints<dim>::make_grid()
{
  std::cout << "Creating background mesh" << std::endl;

  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(2);
}

template <>
void TestPoints<2>::setup_discrete_level_set()
{
  std::cout << "Setting up discrete level set function" << std::endl;

  // level_set_dof_handler.distribute_dofs(fe_level_set);
  // level_set.reinit(level_set_dof_handler.n_dofs());

  const Functions::SignedDistance::Sphere<2> signed_distance_sphere({Cx, Cy},
                                                                    R);
  // ImplicitFunction                             implicit_function;
  // VectorTools::interpolate(level_set_dof_handler,
  //                          implicit_function,
  //                          level_set);
  GridGenerator::hyper_sphere(embedded_tria, {Cx, Cy}, R);
  embedded_tria.refine_global(10);

  NonMatchingUtilities::CDT tr;
  using Point2 = NonMatchingUtilities::Point2;
  Point<2> center{Cx, Cy};
  for (const auto &cell : embedded_tria.active_cell_iterators())
    {
      tr.insert_constraint(
        CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(0)),
        CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(1)));

      tr.insert_constraint(
        CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(1)),
        CGALWrappers::dealii_point_to_cgal_point<Point2>(center));

      tr.insert_constraint(
        CGALWrappers::dealii_point_to_cgal_point<Point2>(center),
        CGALWrappers::dealii_point_to_cgal_point<Point2>(cell->vertex(0)));
    }

  NonMatchingUtilities::mark_domains(tr);

  GridTools::Cache<1, 2> cache(embedded_tria);
  auto                   tree = cache.get_cell_bounding_boxes_rtree();

  NonMatchingUtilities::DiscreteLevelSet<2, decltype(tree)> discrete_level_set(
    &tree, tr);

  // { // Sanity checks
  //   std::cout << signed_distance_sphere.value(Point<2>{}) << " and "
  //             << discrete_level_set.value(Point<2>{}) << std::endl;

  //   std::cout << discrete_level_set.value(Point<2>{1.1, 1.1}) << " and "
  //             << signed_distance_sphere.value(Point<2>{1.1, 1.1}) <<
  //             std::endl;

  //   std::cout << discrete_level_set.value(Point<2>{1.2, 1.2}) << " and "
  //             << signed_distance_sphere.value(Point<2>{1.2, 1.2}) <<
  //             std::endl;

  //   std::cout << discrete_level_set.value(Point<2>{0.5, 0.5}) << " and "
  //             << signed_distance_sphere.value(Point<2>{0.5, 0.5}) <<
  //             std::endl;

  //   std::cout << discrete_level_set.value(Point<2>{0.8, 0.8}) << " and "
  //             << signed_distance_sphere.value(Point<2>{0.8, 0.8}) <<
  //             std::endl;
  // }



  // Create randomly distributed Points in the mesh

  unsigned int          n_points = 500;
  std::vector<Point<2>> test_points;
  std::vector<double>   distances;
  for (unsigned int i = 0; i < n_points; ++i)
    {
      const auto p = random_point<2>(-1.1, 1.1);
      test_points.push_back(p);
      distances.push_back((p - center).norm());
    }

  std::vector<double> rel_errors(n_points);
  std::vector<double> abs_errors(n_points);
  unsigned int        j = 0;
  unsigned int        k = 0;
  for (const auto &p : test_points)
    {
      const double correct = signed_distance_sphere.value(p);
      rel_errors[j++] =
        std::abs(correct - discrete_level_set.value(p)) / std::abs(correct);
      abs_errors[k++] = std::abs(correct - discrete_level_set.value(p));
    }

  // Check the errors
  std::cout << "Check the errors:" << std::endl;
  for (unsigned int i = 0; i < n_points; ++i)
    std::cout << "Relative error: " << rel_errors[i] << "\t" << distances[i]
              << "\t"
              << "\t Abolute error: " << abs_errors[i] << std::endl;

  std::cout << "Max relative error:"
            << *std::max_element(rel_errors.begin(), rel_errors.end())
            << std::endl;

  // VectorTools::interpolate(level_set_dof_handler,
  //                          discrete_level_set,
  //                          level_set);
}


template <>
void TestPoints<3>::setup_discrete_level_set()
{
  std::cout << "Setting up discrete level set function" << std::endl;

  const Functions::SignedDistance::Sphere<3> signed_distance_sphere(
    {Cx, Cy, Cz}, R);
  Triangulation<2, 3> embedded_tria;
  GridGenerator::hyper_sphere(embedded_tria, {Cx, Cy, Cz}, R);
  embedded_tria.refine_global(9);
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


  // Create randomly distributed Points in the mesh
  unsigned int          n_points = 200;
  std::vector<Point<3>> test_points;
  std::vector<double>   distances;
  Point<3>              center{Cx, Cy, Cz};
  for (unsigned int i = 0; i < n_points; ++i)
    {
      const auto p = random_point<3>(-1.1, 1.1);
      test_points.push_back(p);
      distances.push_back((p - center).norm());
    }

  std::vector<double> rel_errors(n_points);
  std::vector<double> abs_errors(n_points);
  unsigned int        j = 0;
  unsigned int        k = 0;
  for (const auto &p : test_points)
    {
      const double correct = signed_distance_sphere.value(p);
      std::cout << "Correct: " << correct << "\t"
                << "Approximate: " << discrete_level_set.value(p) << std::endl;
      rel_errors[j++] =
        std::abs(correct - discrete_level_set.value(p)) / std::abs(correct);
      abs_errors[k++] = std::abs(correct - discrete_level_set.value(p));
    }

  // Check the errors
  std::cout << "Check the errors:" << std::endl;
  for (unsigned int i = 0; i < n_points; ++i)
    std::cout << "Relative error: " << rel_errors[i] << "\t" << distances[i]
              << "\t"
              << "\t Abolute error: " << abs_errors[i] << std::endl;

  std::cout << "Max relative error:"
            << *std::max_element(rel_errors.begin(), rel_errors.end())
            << std::endl;
}



int main(int argc, char *argv[])
{
  TestPoints<2> manufactured_test;
  manufactured_test.setup_discrete_level_set();

  TestPoints<3> manufactured_test_3D;
  manufactured_test_3D.setup_discrete_level_set();
}
