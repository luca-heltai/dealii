
//----------------------------  occ_circle_projection.cc  ---------------------------
//    $Id: testsuite.html 13373 2006-07-13 13:12:08Z kanschat $
//    Version: $Name$ 
//
//    Copyright (C) 2005 by the deal.II authors 
//
//    This file is subject to QPL and may not be  distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//----------------------------  occ_circle_projection.cc  ---------------------------


// Create a circle, a Triangulation, and try to project normally on
// it.

#include "../tests.h"

#include <deal.II/grid/occ_boundary_lib.h>

#include <fstream>
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
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>

using namespace OpenCASCADE;

int main () 
{
  std::ofstream logfile("output");
  // deallog.attach(logfile);
  // deallog.depth_console(0);

  gp_Pnt center(.5,.5,.5);
  Standard_Real radius(Point<3>().distance(Pnt(center)));

  
  TopoDS_Face face = BRepPrimAPI_MakeSphere(center, radius);
  
				   // Create a boundary projector.
  NormalProjectionBoundary<3,3> sphere(face);
  
  
				   // The unit cube.
  Triangulation<3,3> tria;
  GridGenerator::hyper_cube(tria);

				   // Set the exterior boundary
  tria.set_manifold(1, sphere);

				   // This is here to ignore the
				   // points created in the interior
				   // of the face.
  tria.begin()->set_all_manifold_ids(1);
  tria.begin()->set_manifold_id(0);

  tria.refine_global(2);


  // You can open the generated file with gmsh
  GridOut gridout;
  gridout.write_msh (tria, logfile);
  
  return 0;
}
                  
