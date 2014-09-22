//-----------------------------------------------------------
//
//    Copyright (C) 2014 by the deal.II authors 
//
//    This file is subject to LGPL and may not be distributed
//    without copyright and license information. Please refer
//    to the file deal.II/doc/license.html for the  text  and
//    further information on this license.
//
//-----------------------------------------------------------

// Read the first face of goteborg.iges, attach it to a projector,
// create a single-cell Triangulation, and refine it with the new
// projector.

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
  
  TopoDS_Shape sh = read_IGES(SOURCE_DIR "/iges_files/goteborg.iges");
  std::vector<TopoDS_Face> faces;
  std::vector<TopoDS_Edge> edges;
  std::vector<TopoDS_Vertex> vertices;
  
  extract_geometrical_shapes(sh, faces, edges, vertices);

  // Create a boundary projector on the first face.
  NormalProjectionBoundary<2,3> boundary(faces[0], 1e-9);
  
  // Create a Triangulation with a single cell
  Triangulation<2,3> tria;
  create_triangulation(faces[0], tria);

  // Set the boundary
  tria.set_manifold(1, boundary);
  tria.begin()->set_all_manifold_ids(1);

  tria.refine_global(2);

  // You can open the generated file with gmsh
  GridOut gridout;
  gridout.write_msh (tria, logfile);
  
  return 0;
}
                  
