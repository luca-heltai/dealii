
//----------------------------  iges_read.cc  ---------------------------
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
//----------------------------  iges_read.cc  ---------------------------


// Read goteborg.iges and dump its topological structure to the logfile.

#include "../tests.h"
#include <fstream>
#include <base/logstream.h>

#include <deal.II/grid/occ_utilities.h>
#include <deal.II/grid/occ_boundary_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>

#include <TopTools.hxx>
#include <TopoDS_Shape.hxx>
#include <Standard_Stream.hxx>

using namespace OpenCASCADE;

int main () 
{
  std::ofstream logfile("output");
  // deallog.attach(logfile);
  // deallog.depth_console(0);
  
  // Create a bspline passign through the points
  std::vector<Point<3> > pts;
  pts.push_back(Point<3>(0,0,0));
  pts.push_back(Point<3>(0,1,0));
  pts.push_back(Point<3>(1,1,0));
  pts.push_back(Point<3>(1,0,0));

  TopoDS_Edge edge = interpolation_curve(pts);
  ArclengthProjectionLineManifold<1,3> manifold(edge);
  
  Triangulation<1,3> tria;
  GridGenerator::hyper_cube(tria);

  tria.begin_active()->set_all_manifold_ids(1);
  tria.set_manifold(1, manifold);
  
  tria.refine_global(4);
  GridOut gridout;
  gridout.write_msh(tria, logfile);

  return 0;
}
                  
