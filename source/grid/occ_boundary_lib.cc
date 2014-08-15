#include <deal.II/grid/occ_boundary_lib.h>

#ifdef DEAL_II_WITH_OPENCASCADE

#include <GCPnts_AbscissaPoint.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <GCPnts_AbscissaPoint.hxx>
#include <ShapeAnalysis_Curve.hxx>
#include <BRep_Tool.hxx>

DEAL_II_NAMESPACE_OPEN

namespace OpenCASCADE
{

  /*============================== NormalProjectionBoundary ==============================*/
  template <int dim, int spacedim>
  NormalProjectionBoundary<dim,spacedim>::NormalProjectionBoundary(const TopoDS_Shape sh, 
								   const double tolerance) :
    sh(sh),
    tolerance(tolerance) 
  {
    Assert(spacedim == 3, ExcImpossibleInDim(spacedim));
  }
  
  
  template <int dim, int spacedim>
  Point<spacedim>  NormalProjectionBoundary<dim,spacedim>::
  project_to_manifold (const std::vector<Point<spacedim> > &surrounding_points,
		       const Point<spacedim> &candidate) const {
    TopoDS_Shape out_shape;
    double u=0, v=0;
    for(unsigned int i=0; i<surrounding_points.size(); ++i)
      Assert(closest_point(sh, surrounding_points[i], out_shape, u, v)
	     .distance(surrounding_points[i]) < 
	     std::max(tolerance*surrounding_points[i].norm(), tolerance),
	     ExcPointNotOnManifold(surrounding_points[i]));
    
    return closest_point(sh, candidate, out_shape, u, v);
  }

  /*============================== AxisProjectionBoundary ==============================*/
  template <int dim, int spacedim>
  AxisProjectionBoundary<dim,spacedim>::AxisProjectionBoundary(const TopoDS_Shape sh, 
							       const Point<3> direction,
							       const double tolerance) :
    sh(sh),
    direction(direction),
    tolerance(tolerance) 
  {
    Assert(spacedim == 3, ExcImpossibleInDim(spacedim));
  }
  
  
  template <int dim, int spacedim>
  Point<spacedim>  AxisProjectionBoundary<dim,spacedim>::
  project_to_manifold (const std::vector<Point<spacedim> > &surrounding_points,
		       const Point<spacedim> &candidate) const {
    TopoDS_Shape out_shape;
    double u=0, v=0;
    for(unsigned int i=0; i<surrounding_points.size(); ++i)
      Assert(closest_point(sh, surrounding_points[i], out_shape, u, v)
	     .distance(surrounding_points[i]) < (surrounding_points[i].norm()>0 ? 
						 tolerance*surrounding_points[i].norm() :
						 tolerance),
	     ExcPointNotOnManifold(surrounding_points[i]));
    
    return axis_intersection(sh, candidate, direction, tolerance);
  }

  
  /*============================== ArclengthProjectionLineManifold ==============================*/
  
  template <int dim, int spacedim>
  ArclengthProjectionLineManifold<dim,spacedim>::ArclengthProjectionLineManifold(const TopoDS_Edge &sh,
										 const double tolerance):
    
    ChartManifold<dim,spacedim,1>(sh.Closed() ? 
				  Point<1>(OpenCASCADE::length(sh)) :
				  Point<1>()),
    curve(sh), 
    tolerance(tolerance),
    length(OpenCASCADE::length(sh))
  {
    Assert(spacedim == 3, ExcImpossibleInDim(spacedim));
    Assert(!BRep_Tool::Degenerated(sh), ExcEdgeIsDegenerate());
  }

  
  template <int dim, int spacedim>
  Point<1> 
  ArclengthProjectionLineManifold<dim,spacedim>::pull_back(const Point<spacedim> &space_point) const {
    ShapeAnalysis_Curve curve_analysis;
    gp_Pnt proj;
    double t;
    double dist = curve_analysis.Project(curve, Pnt(space_point), tolerance, proj, t, true);
    Assert(dist < tolerance*length, ExcPointNotOnManifold(space_point));
    return Point<1>(GCPnts_AbscissaPoint::Length(curve,curve.FirstParameter(),t));
  }

  
  
  template <int dim, int spacedim>
  Point<spacedim> 
  ArclengthProjectionLineManifold<dim,spacedim>::push_forward(const Point<1> &chart_point) const {
    GCPnts_AbscissaPoint AP(curve, chart_point[0], curve.FirstParameter());
    gp_Pnt P = curve.Value(AP.Parameter());
    return Pnt(P);
  }
  

  // Explicit instantiations
#include "occ_boundary_lib.inst"  
  
} // end namespace OpenCASCADE

DEAL_II_NAMESPACE_CLOSE

#endif
