#ifndef dealii_singular_integral_tools_h
#define dealii_singular_integral_tools_h

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria_accessor.h>


DEAL_II_NAMESPACE_OPEN

namespace SingularIntegralTools
{
  namespace internal
  {
    std::
      tuple<std::vector<Point<2>>, std::vector<Point<2>>, std::vector<double>>
      simplices_identical_panels(
        const std::vector<std::shared_ptr<Quadrature<1>>> &quadrature)
    {
      const unsigned int spacedim = 2;
      AssertDimension(quadrature.size(), 2 * spacedim);

      unsigned int tensor_quad_size = 1;
      for (unsigned int k = 0; k < quadrature.size(); ++k)
        tensor_quad_size *= quadrature[k]->size();
      unsigned int quad_size = 6 * tensor_quad_size;

      std::vector<Point<spacedim>> px_hat(quad_size);
      std::vector<Point<spacedim>> py_hat(quad_size);
      std::vector<double>          jxw_hat(quad_size);

      unsigned int counter = 0;

      double              jxw;
      double              xi;
      std::vector<double> eta(2 * spacedim - 1);
      for (unsigned int i = 0; i < quadrature[0]->size(); ++i)
        for (unsigned int j1 = 0; j1 < quadrature[1]->size(); ++j1)
          for (unsigned int j2 = 0; j2 < quadrature[2]->size(); ++j2)
            for (unsigned int j3 = 0; j3 < quadrature[3]->size(); ++j3)
              {
                xi     = quadrature[0]->point(i)[0];
                eta[0] = quadrature[1]->point(j1)[0];
                eta[1] = quadrature[2]->point(j2)[0];
                eta[2] = quadrature[3]->point(j3)[0];

                jxw = quadrature[0]->weight(i) * quadrature[1]->weight(j1) *
                      quadrature[2]->weight(j2) * quadrature[3]->weight(j3);

                px_hat[counter][0] = xi;
                px_hat[counter][1] = xi * (1 - eta[0] + eta[0] * eta[1]);
                py_hat[counter][0] = xi * (1 - eta[0] * eta[1] * eta[2]);
                py_hat[counter][1] = xi * (1 - eta[0]);

                px_hat[counter + 1][0] = py_hat[counter][0];
                px_hat[counter + 1][1] = py_hat[counter][1];
                py_hat[counter + 1][0] = px_hat[counter][0];
                py_hat[counter + 1][1] = px_hat[counter][1];

                px_hat[counter + 2][0] = xi;
                px_hat[counter + 2][1] =
                  xi * eta[0] * (1 - eta[0] + eta[1] * eta[2]);
                py_hat[counter + 2][0] = xi * (1 - eta[0] * eta[1]);
                py_hat[counter + 2][1] = xi * eta[0] * (1 - eta[1]);

                px_hat[counter + 3][0] = py_hat[counter + 2][0];
                px_hat[counter + 3][1] = py_hat[counter + 2][1];
                py_hat[counter + 3][0] = px_hat[counter + 2][0];
                py_hat[counter + 3][1] = px_hat[counter + 2][1];

                px_hat[counter + 4][0] = xi * (1 - eta[0] * eta[1] * eta[2]);
                px_hat[counter + 4][1] = xi * eta[0] * (1 - eta[0] * eta[1]);
                py_hat[counter + 4][0] = xi;
                py_hat[counter + 4][1] = xi * eta[0] * (1 - eta[1]);

                px_hat[counter + 5][0] = py_hat[counter + 4][0];
                px_hat[counter + 5][1] = py_hat[counter + 4][1];
                py_hat[counter + 5][0] = px_hat[counter + 4][0];
                py_hat[counter + 5][1] = px_hat[counter + 4][1];

                jxw *= pow(xi, 3) * eta[0] * eta[0] * eta[1];

                for (unsigned int k = 0; k < 6; ++k)
                  jxw_hat[counter + k] = jxw;

                counter += 6;
              }

      for (unsigned int i = 0; i < quad_size; ++i)
        {
          px_hat[i][0] = 1. - px_hat[i][0];
          py_hat[i][0] = 1. - py_hat[i][0];
        }

      return std::make_tuple(px_hat, py_hat, jxw_hat);
    }


    std::
      tuple<std::vector<Point<2>>, std::vector<Point<2>>, std::vector<double>>
      simplices_common_face(
        const std::vector<std::shared_ptr<Quadrature<1>>> &quadrature)
    {
      const unsigned int spacedim = 2;
      AssertDimension(quadrature.size(), 2 * spacedim);

      unsigned int tensor_quad_size = 1;
      for (unsigned int k = 0; k < quadrature.size(); ++k)
        tensor_quad_size *= quadrature[k]->size();
      unsigned int quad_size = 5 * tensor_quad_size;

      std::vector<Point<spacedim>> px_hat(quad_size);
      std::vector<Point<spacedim>> py_hat(quad_size);
      std::vector<double>          jxw_hat(quad_size);

      unsigned int counter = 0;

      double              jxw;
      double              xi;
      std::vector<double> eta(2 * spacedim - 1);
      for (unsigned int i = 0; i < quadrature[0]->size(); ++i)
        for (unsigned int j1 = 0; j1 < quadrature[1]->size(); ++j1)
          for (unsigned int j2 = 0; j2 < quadrature[2]->size(); ++j2)
            for (unsigned int j3 = 0; j3 < quadrature[3]->size(); ++j3)
              {
                xi     = quadrature[0]->point(i)[0];
                eta[0] = quadrature[1]->point(j1)[0];
                eta[1] = quadrature[2]->point(j2)[0];
                eta[2] = quadrature[3]->point(j3)[0];

                jxw = quadrature[0]->weight(i) * quadrature[1]->weight(j1) *
                      quadrature[2]->weight(j2) * quadrature[3]->weight(j3);

                px_hat[counter][0] = xi;
                px_hat[counter][1] = xi * eta[0] * eta[2];
                py_hat[counter][0] = xi * (1 - eta[0] * eta[1]);
                py_hat[counter][1] = xi * eta[0] * (1 - eta[1]);

                px_hat[counter + 1][0] = xi;
                px_hat[counter + 1][1] = xi * eta[0];
                py_hat[counter + 1][0] = xi * (1 - eta[0] * eta[1] * eta[2]);
                py_hat[counter + 1][1] = xi * eta[0] * eta[1] * (1 - eta[2]);

                px_hat[counter + 2][0] = xi * (1 - eta[0] * eta[1]);
                px_hat[counter + 2][1] = xi * eta[0] * (1 - eta[1]);
                py_hat[counter + 2][0] = xi;
                py_hat[counter + 2][1] = xi * eta[0] * eta[1] * eta[2];

                px_hat[counter + 3][0] = xi * (1 - eta[0] * eta[1] * eta[2]);
                px_hat[counter + 3][1] = xi * eta[0] * eta[1] * (1 - eta[2]);
                py_hat[counter + 3][0] = xi;
                py_hat[counter + 3][1] = xi * eta[0];

                px_hat[counter + 4][0] = xi * (1 - eta[0] * eta[1] * eta[2]);
                px_hat[counter + 4][1] = xi * eta[0] * (1 - eta[1] * eta[2]);
                py_hat[counter + 4][0] = xi;
                py_hat[counter + 4][1] = xi * eta[0] * eta[1];

                jxw_hat[counter] = jxw * pow(xi, 3) * eta[0] * eta[0];

                jxw *= pow(xi, 3) * eta[0] * eta[0] * eta[1];
                for (unsigned int k = 1; k < 5; ++k)
                  jxw_hat[counter + k] = jxw;

                counter += 5;
              }

      for (unsigned int i = 0; i < quad_size; ++i)
        {
          px_hat[i][0] = 1. - px_hat[i][0];
          py_hat[i][0] = 1. - py_hat[i][0];
        }

      return std::make_tuple(px_hat, py_hat, jxw_hat);
    }

    std::
      tuple<std::vector<Point<2>>, std::vector<Point<2>>, std::vector<double>>
      simplices_common_vertex(
        const std::vector<std::shared_ptr<Quadrature<1>>> &quadrature)
    {
      const unsigned int spacedim = 2;
      AssertDimension(quadrature.size(), 2 * spacedim);

      unsigned int tensor_quad_size = 1;
      for (unsigned int k = 0; k < quadrature.size(); ++k)
        tensor_quad_size *= quadrature[k]->size();
      unsigned int quad_size = 2 * tensor_quad_size;

      std::vector<Point<spacedim>> px_hat(quad_size);
      std::vector<Point<spacedim>> py_hat(quad_size);
      std::vector<double>          jxw_hat(quad_size);

      unsigned int counter = 0;

      double              jxw;
      double              xi;
      std::vector<double> eta(2 * spacedim - 1);
      for (unsigned int i = 0; i < quadrature[0]->size(); ++i)
        for (unsigned int j1 = 0; j1 < quadrature[1]->size(); ++j1)
          for (unsigned int j2 = 0; j2 < quadrature[2]->size(); ++j2)
            for (unsigned int j3 = 0; j3 < quadrature[3]->size(); ++j3)
              {
                xi     = quadrature[0]->point(i)[0];
                eta[0] = quadrature[1]->point(j1)[0];
                eta[1] = quadrature[2]->point(j2)[0];
                eta[2] = quadrature[3]->point(j3)[0];

                jxw = quadrature[0]->weight(i) * quadrature[1]->weight(j1) *
                      quadrature[2]->weight(j2) * quadrature[3]->weight(j3);

                px_hat[counter][0] = xi;
                px_hat[counter][1] = xi * eta[0];
                py_hat[counter][0] = xi * eta[1];
                py_hat[counter][1] = xi * eta[1] * eta[2];

                px_hat[counter + 1][0] = py_hat[counter][0];
                px_hat[counter + 1][1] = py_hat[counter][1];
                py_hat[counter + 1][0] = px_hat[counter][0];
                py_hat[counter + 1][1] = px_hat[counter][1];

                jxw *= pow(xi, 3) * eta[1];

                for (unsigned int k = 0; k < 2; ++k)
                  jxw_hat[counter + k] = jxw;

                counter += 2;
              }

      for (unsigned int i = 0; i < quad_size; ++i)
        {
          px_hat[i][0] = 1. - px_hat[i][0];
          py_hat[i][0] = 1. - py_hat[i][0];
        }

      return std::make_tuple(px_hat, py_hat, jxw_hat);
    }


    std::
      tuple<std::vector<Point<2>>, std::vector<Point<2>>, std::vector<double>>
      simplices_disjoint(
        const std::vector<std::shared_ptr<Quadrature<1>>> &quadrature)
    {
      const unsigned int spacedim = 2;
      AssertDimension(quadrature.size(), 2 * spacedim);

      unsigned int tensor_quad_size = 1;
      for (unsigned int k = 0; k < quadrature.size(); ++k)
        tensor_quad_size *= quadrature[k]->size();
      unsigned int quad_size = tensor_quad_size;

      std::vector<Point<spacedim>> px_hat(quad_size);
      std::vector<Point<spacedim>> py_hat(quad_size);
      std::vector<double>          jxw_hat(quad_size);

      unsigned int counter = 0;

      double              jxw;
      double              xi;
      std::vector<double> eta(2 * spacedim - 1);
      for (unsigned int i = 0; i < quadrature[0]->size(); ++i)
        for (unsigned int j1 = 0; j1 < quadrature[1]->size(); ++j1)
          for (unsigned int j2 = 0; j2 < quadrature[2]->size(); ++j2)
            for (unsigned int j3 = 0; j3 < quadrature[3]->size(); ++j3)
              {
                xi     = quadrature[0]->point(i)[0];
                eta[0] = quadrature[1]->point(j1)[0];
                eta[1] = quadrature[2]->point(j2)[0];
                eta[2] = quadrature[3]->point(j3)[0];

                jxw = quadrature[0]->weight(i) * quadrature[1]->weight(j1) *
                      quadrature[2]->weight(j2) * quadrature[3]->weight(j3);

                px_hat[counter][0] = xi;
                px_hat[counter][1] = xi * eta[0];
                py_hat[counter][0] = eta[1];
                py_hat[counter][1] = eta[1] * eta[2];

                jxw_hat[counter] = jxw * xi * eta[1];

                counter++;
              }

      for (unsigned int i = 0; i < quad_size; ++i)
        {
          px_hat[i][0] = 1. - px_hat[i][0];
          py_hat[i][0] = 1. - py_hat[i][0];
        }

      return std::make_tuple(px_hat, py_hat, jxw_hat);
    }

    std::pair<QSimplex<2>, QSimplex<2>>
    get_duffy_coupling_quadrature(
      const std::vector<unsigned int> &quadrature_order,
      const unsigned int               n_common_vertices)
    {
      const unsigned int spacedim = 2;
      AssertDimension(quadrature_order.size(), 2 * spacedim);

      std::vector<std::shared_ptr<Quadrature<1>>> tensor_quadrature(
        quadrature_order.size());
      for (unsigned int k = 0; k < quadrature_order.size(); ++k)
        tensor_quadrature[k] = std::make_shared<QGauss<1>>(quadrature_order[k]);
      Quadrature<spacedim> quad_left, quad_right;

      switch (n_common_vertices)
        {
            case 0: {
              const auto quad_tuple =
                internal::simplices_disjoint(tensor_quadrature);
              const unsigned int  quad_size = std::get<2>(quad_tuple).size();
              std::vector<double> unit_weights(quad_size, 1.0);

              quad_left.initialize(std::get<0>(quad_tuple),
                                   std::get<2>(quad_tuple));
              quad_right.initialize(std::get<1>(quad_tuple), unit_weights);

              break;
            }
            case 1: {
              const auto quad_tuple =
                internal::simplices_common_vertex(tensor_quadrature);
              const unsigned int  quad_size = std::get<2>(quad_tuple).size();
              std::vector<double> unit_weights(quad_size, 1.0);

              quad_left.initialize(std::get<0>(quad_tuple),
                                   std::get<2>(quad_tuple));
              quad_right.initialize(std::get<1>(quad_tuple), unit_weights);

              break;
            }
            case 2: {
              const auto quad_tuple = simplices_common_face(tensor_quadrature);
              const unsigned int  quad_size = std::get<2>(quad_tuple).size();
              std::vector<double> unit_weights(quad_size, 1.0);

              quad_left.initialize(std::get<0>(quad_tuple),
                                   std::get<2>(quad_tuple));
              quad_right.initialize(std::get<1>(quad_tuple), unit_weights);

              break;
            }
            case 3: {
              const auto quad_tuple =
                simplices_identical_panels(tensor_quadrature);
              const unsigned int  quad_size = std::get<2>(quad_tuple).size();
              std::vector<double> unit_weights(quad_size, 1.0);

              quad_left.initialize(std::get<0>(quad_tuple),
                                   std::get<2>(quad_tuple));
              quad_right.initialize(std::get<1>(quad_tuple), unit_weights);

              break;
            }
            default: {
              Assert(false,
                     ExcMessage("The possible number of common vertices"
                                "for two triangles is 0, 1, 2 or 3."));
              break;
            }
        }

      return std::make_pair(quad_left, quad_right);
    }
  } // namespace internal



  // template <int dim, int spacedim>
  // std::vector<std::tuple<types::global_dof_index, unsigned int, unsigned
  // int>> get_cell_coupling_info(
  //   const typename DoFHandler<dim, spacedim>::active_cell_iterator
  //   &left_cell, const typename DoFHandler<dim,
  //   spacedim>::active_cell_iterator &right_cell);

  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  get_cell_coupling_reordering(
    const std::vector<
      std::tuple<types::global_dof_index, unsigned int, unsigned int>>
                      &coupling_info,
    const unsigned int n_left_vertices,
    const unsigned int n_right_vertices);

  // template <int dim, int spacedim>
  // std::pair<QSimplex<dim>, QSimplex<dim>>
  // get_duffy_coupling_quadrature_at_reference(
  //   const typename DoFHandler<dim, spacedim>::active_cell_iterator
  //   &left_cell, const typename DoFHandler<dim,
  //   spacedim>::active_cell_iterator &right_cell, const std::vector<unsigned
  //   int> &quadrature_order);

  // template <int dim, int spacedim>
  // std::pair<Quadrature<spacedim>, Quadrature<spacedim>>
  // get_duffy_coupling_quadrature(
  //   const typename DoFHandler<dim, spacedim>::active_cell_iterator
  //   &left_cell, const typename DoFHandler<dim,
  //   spacedim>::active_cell_iterator &right_cell, const std::vector<unsigned
  //   int> &quadrature_order);



  std::vector<std::tuple<types::global_dof_index, unsigned int, unsigned int>>
  get_cell_coupling_info(
    const typename DoFHandler<2, 3>::active_cell_iterator &left_cell,
    const typename DoFHandler<2, 3>::active_cell_iterator &right_cell)
  {
    std::vector<std::tuple<types::global_dof_index, unsigned int, unsigned int>>
      shared_vertices;
    for (const auto v1 : left_cell->vertex_indices())
      for (const auto v2 : right_cell->vertex_indices())
        {
          if (left_cell->vertex_index(v1) == right_cell->vertex_index(v2))
            shared_vertices.push_back(
              std::make_tuple(left_cell->vertex_index(v1), v1, v2));
        }
    return shared_vertices;
  }


  std::pair<std::vector<unsigned int>, std::vector<unsigned int>>
  get_cell_coupling_reordering(
    const std::vector<
      std::tuple<types::global_dof_index, unsigned int, unsigned int>>
                      &coupling_info,
    const unsigned int n_left_vertices,
    const unsigned int n_right_vertices)
  {
    std::vector<unsigned int> left_reordering;
    std::vector<unsigned int> right_reordering;

    std::vector<unsigned int> left_normal_reordering(n_left_vertices);
    std::vector<unsigned int> right_normal_reordering(n_right_vertices);
    std::iota(left_normal_reordering.begin(), left_normal_reordering.end(), 0);
    std::iota(right_normal_reordering.begin(),
              right_normal_reordering.end(),
              0);

    // simplex case only
    unsigned int n_shared_vertices = coupling_info.size();
    switch (n_shared_vertices)
      {
          case 0: {
            left_reordering  = left_normal_reordering;
            right_reordering = right_normal_reordering;
            break;
          }
          case 1: {
            left_reordering  = left_normal_reordering;
            right_reordering = right_normal_reordering;

            unsigned int local_left_index  = std::get<1>(coupling_info[0]);
            unsigned int local_right_index = std::get<2>(coupling_info[0]);

            std::swap(left_reordering[0], left_reordering[local_left_index]);
            std::swap(right_reordering[0], right_reordering[local_right_index]);
            break;
          }
          case 2: {
            for (const auto &info : coupling_info)
              {
                left_reordering.push_back(std::get<1>(info));
                right_reordering.push_back(std::get<2>(info));
              }
            for (const auto i : left_normal_reordering)
              if (std::find(left_reordering.begin(),
                            left_reordering.end(),
                            i) == left_reordering.end())
                left_reordering.push_back(i);
            for (const auto i : right_normal_reordering)
              if (std::find(right_reordering.begin(),
                            right_reordering.end(),
                            i) == right_reordering.end())
                right_reordering.push_back(i);
            break;
          }
          case 3: {
            left_reordering  = left_normal_reordering;
            right_reordering = right_normal_reordering;
            break;
          }
          default: {
            Assert(false,
                   ExcMessage("The possible number of common vertices"
                              "for two triangles is 0, 1, 2 or 3."));
            break;
          }
      }
    return std::make_pair(left_reordering, right_reordering);
  }


  std::pair<QSimplex<2>, QSimplex<2>>
  get_duffy_coupling_quadrature_at_reference(
    const typename DoFHandler<2, 3>::active_cell_iterator &left_cell,
    const typename DoFHandler<2, 3>::active_cell_iterator &right_cell,
    const std::vector<unsigned int>                       &quadrature_order)
  {
    const unsigned int dim = 2;
    // const unsigned int spacedim = 3;

    Assert(dim == 2, ExcNotImplemented());
    AssertDimension(quadrature_order.size(), 2 * dim);

    const auto coupling_info = get_cell_coupling_info(left_cell, right_cell);
    const auto coupling_reordering =
      get_cell_coupling_reordering(coupling_info,
                                   left_cell->n_vertices(),
                                   right_cell->n_vertices());

    const auto unordered_quadrature =
      internal::get_duffy_coupling_quadrature(quadrature_order,
                                              coupling_info.size());
    const auto left_reordering  = coupling_reordering.first;
    const auto right_reordering = coupling_reordering.second;


    const auto         simplex = ReferenceCells::get_simplex<dim>();
    Triangulation<dim> simplex_grid;
    GridGenerator::reference_cell(simplex_grid, simplex);
    const auto &simplex_vertices = simplex_grid.get_vertices();

    std::array<Point<dim>, dim + 1> left_reordered_vertices;
    std::array<Point<dim>, dim + 1> right_reordered_vertices;
    for (unsigned int i = 0; i < dim + 1; ++i)
      {
        left_reordered_vertices[i]  = simplex_vertices[left_reordering[i]];
        right_reordered_vertices[i] = simplex_vertices[right_reordering[i]];
      }

    QSimplex<dim> left_reference_quad =
      unordered_quadrature.first.compute_affine_transformation(
        left_reordered_vertices);

    QSimplex<dim> right_reference_quad =
      unordered_quadrature.second.compute_affine_transformation(
        right_reordered_vertices);

    return std::make_pair(left_reference_quad, right_reference_quad);
  }


  template <int dim, int spacedim>
  std::pair<Quadrature<spacedim>, Quadrature<spacedim>>
  get_duffy_coupling_quadrature(
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &left_cell,
    const typename DoFHandler<dim, spacedim>::active_cell_iterator &right_cell,
    const std::vector<unsigned int> &quadrature_order)
  {
    auto reference_quad =
      get_duffy_coupling_quadrature_at_reference(left_cell,
                                                 right_cell,
                                                 quadrature_order);

    std::array<Point<spacedim>, dim + 1> left_vertices;
    std::array<Point<spacedim>, dim + 1> right_vertices;
    for (unsigned int i = 0; i < dim + 1; ++i)
      {
        left_vertices[i]  = left_cell->vertex(i);
        right_vertices[i] = right_cell->vertex(i);
      }

    Quadrature<spacedim> left_quad =
      reference_quad.first.compute_affine_transformation(left_vertices);

    Quadrature<spacedim> right_quad =
      reference_quad.second.compute_affine_transformation(right_vertices);

    return std::make_pair(left_quad, right_quad);
  }

} // namespace SingularIntegralTools


DEAL_II_NAMESPACE_CLOSE

#endif