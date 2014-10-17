#include <deal.II/opencascade/boundary_lib.h>

#ifdef DEAL_II_WITH_OPENCASCADE

#include <GCPnts_AbscissaPoint.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_CompCurve.hxx>
#include <BRepAdaptor_HCurve.hxx>
#include <BRepAdaptor_HCompCurve.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <ShapeAnalysis_Curve.hxx>
#include <BRep_Tool.hxx>
#include <TopoDS.hxx>
#include <Adaptor3d_HCurve.hxx>
#include <Handle_Adaptor3d_HCurve.hxx>

DEAL_II_NAMESPACE_OPEN


namespace OpenCASCADE
{


  namespace
  {
    /**
     * Return a Geometrical curve representation for the given
     * TopoDS_Shape. This function will fail when the given shape is
     * not of topological dimension one.
     */
    Handle_Adaptor3d_HCurve curve_adaptor(const TopoDS_Shape &shape)
    {
      Assert( (shape.ShapeType() == TopAbs_WIRE) ||
              (shape.ShapeType() == TopAbs_EDGE),
              ExcUnsupportedShape());
      if (shape.ShapeType() == TopAbs_WIRE)
        return (Handle(BRepAdaptor_HCompCurve(new BRepAdaptor_HCompCurve(TopoDS::Wire(shape)))));
      else if (shape.ShapeType() == TopAbs_EDGE)
        return (Handle(BRepAdaptor_HCurve(new BRepAdaptor_HCurve(TopoDS::Edge(shape)))));

      Assert(false, ExcInternalError());
      return Handle(BRepAdaptor_HCurve(new BRepAdaptor_HCurve()));
    }



    // Helper internal functions.
    double shape_length(const TopoDS_Shape &sh)
    {
      Handle_Adaptor3d_HCurve adapt = curve_adaptor(sh);
      return GCPnts_AbscissaPoint::Length(adapt->GetCurve());
    }
  }

  /*============================== NormalProjectionBoundary ==============================*/
  template <int dim, int spacedim>
  NormalProjectionBoundary<dim,spacedim>::NormalProjectionBoundary(const TopoDS_Shape &sh,
      const double tolerance) :
    sh(sh),
    tolerance(tolerance)
  {
    Assert(spacedim == 3, ExcNotImplemented());
  }


  template <int dim, int spacedim>
  Point<spacedim>  NormalProjectionBoundary<dim,spacedim>::
  project_to_manifold (const std::vector<Point<spacedim> > &surrounding_points,
                       const Point<spacedim> &candidate) const
  {
    TopoDS_Shape out_shape;
    double u=0, v=0;
    for (unsigned int i=0; i<surrounding_points.size(); ++i)
      Assert(closest_point(sh, surrounding_points[i], out_shape, u, v)
             .distance(surrounding_points[i]) <
             std::max(tolerance*surrounding_points[i].norm(), tolerance),
             ExcPointNotOnManifold(surrounding_points[i]));

    return closest_point(sh, candidate, out_shape, u, v);
  }


  /*============================== DirectionalProjectionBoundary ==============================*/
  template <int dim, int spacedim>
  DirectionalProjectionBoundary<dim,spacedim>::DirectionalProjectionBoundary(const TopoDS_Shape &sh,
      const Tensor<1,spacedim> &direction,
      const double tolerance) :
    sh(sh),
    direction(direction),
    tolerance(tolerance)
  {
    Assert(spacedim == 3, ExcNotImplemented());
  }


  template <int dim, int spacedim>
  Point<spacedim>  DirectionalProjectionBoundary<dim,spacedim>::
  project_to_manifold (const std::vector<Point<spacedim> > &surrounding_points,
                       const Point<spacedim> &candidate) const
  {
    TopoDS_Shape out_shape;
    double u=0, v=0;
    for (unsigned int i=0; i<surrounding_points.size(); ++i)
      Assert(closest_point(sh, surrounding_points[i], out_shape, u, v)
             .distance(surrounding_points[i]) <
             std::max(tolerance*surrounding_points[i].norm(), tolerance),
             ExcPointNotOnManifold(surrounding_points[i]));

    return line_intersection(sh, candidate, direction, tolerance);

  }



  /*============================== NormalToMeshProjectionBoundary ==============================*/
  template <int dim, int spacedim>
  NormalToMeshProjectionBoundary<dim,spacedim>::NormalToMeshProjectionBoundary(const TopoDS_Shape &sh,
      const double tolerance) :
    sh(sh),
    tolerance(tolerance)
  {
    Assert(spacedim == 3, ExcNotImplemented());

    unsigned int n_faces;
    unsigned int n_edges;
    unsigned int n_vertices;
    count_elements(sh,
                   n_faces,
                   n_edges,
                   n_vertices);

    Assert(n_faces > 0, ExcMessage("NormalToMeshProjectionBoundary needs a shape containing faces to operate."));
  }


  template <int dim, int spacedim>
  Point<spacedim>  NormalToMeshProjectionBoundary<dim,spacedim>::
  project_to_manifold (const std::vector<Point<spacedim> > &surrounding_points,
                       const Point<spacedim> &candidate) const
  {
    TopoDS_Shape out_shape;
    double u=0, v=0;
    Point<3> average_normal(0.0,0.0,0.0);
    for (unsigned int i=0; i<surrounding_points.size(); ++i)
      {
        Assert(closest_point(sh, surrounding_points[i], out_shape, u, v)
               .distance(surrounding_points[i]) <
               std::max(tolerance*surrounding_points[i].norm(), tolerance),
               ExcPointNotOnManifold(surrounding_points[i]));
      }

    switch (surrounding_points.size())
      {
      case 2:
      {
        for (unsigned int i=0; i<surrounding_points.size(); ++i)
          {
            Point<3> surface_normal;
            double mean_curvature;
            closest_point_and_differential_forms(sh, surrounding_points[i], surface_normal, mean_curvature);
            average_normal += surface_normal;
          }

        average_normal/=2.0;

        Assert(average_normal.norm() > 1e-4,
               ExcMessage("Failed to refine cell: the average of the surface normals at the surrounding edge turns out to be a null vector, making the projection direction undetermined."));

        Point<3> P = (surrounding_points[0]+surrounding_points[1])/2;
        Point<3> N = surrounding_points[0]-surrounding_points[1];
        N = N/sqrt(N.square());
        average_normal = average_normal-(average_normal*N)*N;
        average_normal = average_normal/average_normal.norm();
        break;
      }
      case 8:
      {
        Point<3> u = surrounding_points[1]-surrounding_points[0];
        Point<3> v = surrounding_points[2]-surrounding_points[0];
        Point<3> n1(u(1)*v(2)-u(2)*v(1),u(2)*v(0)-u(0)*v(2),u(1)*v(1)-u(1)*v(0));
        n1 = n1/n1.norm();
        u = surrounding_points[2]-surrounding_points[3];
        v = surrounding_points[1]-surrounding_points[3];
        Point<3> n2(u(1)*v(2)-u(2)*v(1),u(2)*v(0)-u(0)*v(2),u(1)*v(1)-u(1)*v(0));
        n2 = n2/n2.norm();
        u = surrounding_points[4]-surrounding_points[7];
        v = surrounding_points[6]-surrounding_points[7];
        Point<3> n3(u(1)*v(2)-u(2)*v(1),u(2)*v(0)-u(0)*v(2),u(1)*v(1)-u(1)*v(0));
        n3 = n3/n3.norm();
        u = surrounding_points[6]-surrounding_points[7];
        v = surrounding_points[5]-surrounding_points[7];
        Point<3> n4(u(1)*v(2)-u(2)*v(1),u(2)*v(0)-u(0)*v(2),u(1)*v(1)-u(1)*v(0));
        n4 = n4/n4.norm();

        average_normal = (n1+n2+n3+n4)/4.0;

        Assert(average_normal.norm() > 1e-4,
               ExcMessage("Failed to refine cell: the average of the surface normals at the surrounding edge turns out to be a null vector, making the projection direction undetermined."));

        average_normal = average_normal/average_normal.norm();
        break;
      }
      default:
      {
        AssertThrow(false, ExcNotImplemented());
        break;
      }
      }

    return line_intersection(sh, candidate, average_normal, tolerance);
  }


  /*============================== ArclengthProjectionLineManifold ==============================*/
  template <int dim, int spacedim>
  ArclengthProjectionLineManifold<dim,spacedim>::ArclengthProjectionLineManifold(const TopoDS_Shape &sh,
      const double tolerance):

    ChartManifold<dim,spacedim,1>(sh.Closed() ?
                                  Point<1>(shape_length(sh)) :
                                  Point<1>()),
    curve(curve_adaptor(sh)),
    tolerance(tolerance),
    length(shape_length(sh))
  {
    Assert(spacedim == 3, ExcNotImplemented());
  }


  template <int dim, int spacedim>
  Point<1>
  ArclengthProjectionLineManifold<dim,spacedim>::pull_back(const Point<spacedim> &space_point) const
  {
    ShapeAnalysis_Curve curve_analysis;
    gp_Pnt proj;
    double t;
    double dist = curve_analysis.Project(curve->GetCurve(), point(space_point), tolerance, proj, t, true);
    Assert(dist < tolerance*length, ExcPointNotOnManifold(space_point));
    return Point<1>(GCPnts_AbscissaPoint::Length(curve->GetCurve(),curve->GetCurve().FirstParameter(),t));
  }



  template <int dim, int spacedim>
  Point<spacedim>
  ArclengthProjectionLineManifold<dim,spacedim>::push_forward(const Point<1> &chart_point) const
  {
    GCPnts_AbscissaPoint AP(curve->GetCurve(), chart_point[0], curve->GetCurve().FirstParameter());
    gp_Pnt P = curve->GetCurve().Value(AP.Parameter());
    return point(P);
  }


  // Explicit instantiations
#include "boundary_lib.inst"

} // end namespace OpenCASCADE

DEAL_II_NAMESPACE_CLOSE

#endif
