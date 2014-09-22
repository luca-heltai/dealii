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

// Read goteborg.iges and dump its topological structure to the
// logfile.

#include "../tests.h"
#include <fstream>
#include <base/logstream.h>

#include <deal.II/grid/occ_utilities.h>
#include <TopTools.hxx>
#include <TopoDS_Shape.hxx>
#include <Standard_Stream.hxx>

using namespace OpenCASCADE;

int main () 
{
  std::ofstream logfile("output");
  deallog.attach(logfile);
  deallog.depth_console(0);
  
  TopoDS_Shape sh = read_IGES(SOURCE_DIR "/iges_files/goteborg.iges");
  unsigned int nf=0, ne=0, nv=0;
  count_elements(sh, nf, ne, nv);
  deallog << "Shape contains " << nf << " faces, "
	  << ne << " edges, and " 
	  << nv << " vertices." << std::endl;

  return 0;
}
                  
