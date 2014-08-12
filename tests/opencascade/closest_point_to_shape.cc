// Makes a OpenCASCADE cicrular arc, and project a few points onto it.


#include "../tests.h"

#include <deal.II/grid/occ_utilities.h>

#include <base/logstream.h>
#include <grid/tria.h>
#include <grid/tria_accessor.h>
#include <grid/grid_out.h>
#include <grid/tria_iterator.h>
#include <grid/grid_generator.h>
#include <grid/tria_boundary_lib.h>

#include <gp_Pnt.hxx>
#include <gp_Dir.hxx>
#include <gp_Ax2.hxx>
#include <GC_MakeCircle.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <TopoDS_Edge.hxx>

using namespace OpenCASCADE;


int main () 
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.depth_console(0);

				   // A unit circle
  gp_Dir z_axis(0.,0.,1.);
  gp_Pnt center(0.,0.,0.);
  gp_Ax2 axis(center, z_axis);
  Standard_Real radius(1.);
  
  Handle(Geom_Curve) circle = GC_MakeCircle(axis, radius);
  TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(circle);
  
  
				   // Now get a few points and project
				   // them on the circle
  std::vector<Point<3> > points;
  
  points.push_back(Point<3>(3,0,0));
  points.push_back(Point<3>(0,3,0));
				   // This one is tricky... the point
				   // is not on the xy plane. If we
				   // put it on the axis (x=0, y=0),
				   // there should not exist a
				   // projection. Any value for z
				   // should give the same result.  
  points.push_back(Point<3>(.1,0,3));
  points.push_back(Point<3>(.1,0,4)); 
 

  double u, v;
  TopoDS_Shape sh;
  for(unsigned int i=0; i<points.size(); ++i) 
    {
      Point<3> pp = closest_point(edge, points[i], sh, u, v);
      
      deallog << "Origin: " << points[i]
	      << ", on unit circle: " << pp
	      << ", with local coordinates (u, v): (" << u
	      << ", " << v << ")" << std::endl;
    }
  return 0;
}

