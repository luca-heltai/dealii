/* ---------------------------------------------------------------------
 * $Id$
 *
 * Copyright (C) 2009 - 2013 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *  Authors: Andrea Mola, Luca Heltai, 2014
 */


// @sect3{Include files}

// We start with including a bunch of files that we will use
// in the various parts of the program. Most of them have been discussed in
// previous tutorials already:
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// And here are the headers of the opencascade support classes and functions:
#include <deal.II/opencascade/boundary_lib.h>
#include <deal.II/opencascade/utilities.h>


// Finally, a few C++ standard header files that we will need:
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace Step54
{
  using namespace dealii;



  // @sect3{The TriangulationOnCAD class}

  // The structure of this class is very small. Since we only want
  // to show how a triangulation can be refined onto a CAD surface, the
  // arguments of this class are basically just the input and output file
  // names, and a flag indicating the surface projection strategy we want to
  // test. Along with the input argument, the only other member of the class
  // is the triangulation we want to play with. 
  // The member functions of this class are similar to those that in most of the
  // other tutorial programs deal with the setup of the grid for the
  // simulations.

  class TriangulationOnCAD
  {
  public:
    TriangulationOnCAD(const std::string &initial_mesh_filename,
                       const std::string &output_filename,
                       const unsigned int &surface_projection_kind = 0);


    ~TriangulationOnCAD();

    void run();

  private:

    void read_domain();

    void refine_and_resize();

    void output_results(const unsigned int cycle);

    Triangulation<2, 3>   tria;

    const std::string &initial_mesh_filename;
    const std::string &output_filename;
    const unsigned int &surface_projection_kind;

  };


  // @sect4{TriangulationOnCAD::TriangulationOnCAD }

  // The constructor of the TriangulationOnCAD class is very simple.
  // The input arguments are strings for the input and output file
  // names, and an unsigned int flag which can only assume values
  // 0,1,2 and determines which kind of surface projector is used in
  // the mesh refinement cycles (see below for details).

  TriangulationOnCAD::TriangulationOnCAD(const std::string &initial_mesh_filename,
                                         const std::string &output_filename,
                                         const unsigned int &surface_projection_kind)
    :
    initial_mesh_filename(initial_mesh_filename),
    output_filename(output_filename),
    surface_projection_kind(surface_projection_kind)
  {
  }

  TriangulationOnCAD::~TriangulationOnCAD()
  {
  }



  // @sect4{TriangulationOnCAD::read_domain}


  // The following function represents the core of thhis program.
  // In this function we in fact import the CAD shape upon which we want to generate
  // and refine our triangulation. Such CAD surface is contained in the IGES
  // file "DTMB-5415_bulbous_bow.iges", and represents the bulbous bow of a ship.
  // The presence of several convex and concave high curvature regions makes this
  // geometry a particularly meaningful example.
  //
  // So, after importing the hull bow surface, we extract some of the curves and surfaces
  // composing it, and use them to generate a set of projectors. Such projectors substantially
  // define the rules the Triangulation has to follow to position each new node during the cell
  // refinement.
  //
  // To initialize the Triangulation, as done in previous tutorial programs, we import a
  // pre-existing grid saved in VTK format. The imported mesh is composed of a single
  // quadrilateral cell the vertices of which have been placed on the CAD shape. 
  //
  // So, after importing both the IGES geometry and the initial mesh, we assign the projectors
  // previously discussed to each of the edges and cells which will have to be
  // refined on the CAD surface.
  //
  // In this tutorial, we will test the three different CAD surface projectors described in
  // the introduction, and will analyze the results obtained with each of them.
  // As mentioned, each of such projection strategiy has been
  // implemented in a different class, which can be assigned to the set_manifold method
  // of a Triangulation class.
  //




  void TriangulationOnCAD::read_domain()
  {

    // The following function allows for the CAD file of interest (in IGES format) to be imported.
    // The function arguments are a string containing the desired file name, and
    // a scale factor. In this example, such scale factor is set to 1e-3, as the original
    // geometry is written in millimeters, while we prefer to work in meters.
    // The output of the function is an object of OpenCASCADE generic topological shape
    // class, namely a TopoDS_Shape.

    TopoDS_Shape bow_surface = OpenCASCADE::read_IGES("DTMB-5415_bulbous_bow.iges",1e-3);

    // Each CAD geometrical object is defined along with a tolerance, which indicates
    // possible inaccuracy of its placement. For instance, the tolerance tol of a vertex
    // indicates that it can be located in any point contained in a sphere centered
    // in the nominal position and having radius tol. While projecting a point onto a
    // surface (which will in turn have its tolerance) we must keep in mind that the
    // precision of the projection will be limited by the tolerance with which the
    // surface is built.

    // Thus, we use a method that extracts the tolerance of a desired shape

    double tolerance = OpenCASCADE::get_shape_tolerance(bow_surface);


    // To stay out of trouble, we make this tolerance a bit bigger
    tolerance*=5.0;

    // We now want to extract from the generic shape, a set of composite sub-shapes (we are
    // in particular interested in the single wire contained in the CAD file, which will
    // allow us to define a line projector).
    // To extract all these sub-shapes, we resort to a method of the OpenCASCADE namespace.
    // The input of extract_compound_shapes is a shape and a set of empty std::vectors
    // of subshapes.
    std::vector<TopoDS_Compound> compounds;
    std::vector<TopoDS_CompSolid> compsolids;
    std::vector<TopoDS_Solid> solids;
    std::vector<TopoDS_Shell> shells;
    std::vector<TopoDS_Wire> wires;

    OpenCASCADE::extract_compound_shapes(bow_surface,
                                         compounds,
                                         compsolids,
                                         solids,
                                         shells,
                                         wires);

    // The next few steps are more familiar, and allow us to import an existing
    // mesh from an external VTK file, and convert it to a deal triangulation.
    std::ifstream in;

    in.open(initial_mesh_filename.c_str());

    GridIn<2,3> gi;
    gi.attach_triangulation(tria);
    gi.read_vtk(in);

    // We output this initial mesh saving it as the refinement step 0.
    output_results(0);

    // The mesh imported has a single cell. So, we get an iterator to that cell.
    // and assgin it the manifold_id 1
    Triangulation<2,3>::active_cell_iterator cell = tria.begin_active();
    cell->set_manifold_id(1);

    // We also get an iterator to its faces, and assign each of them to manifold_id 2.

    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
      cell->face(f)->set_manifold_id(2);

    // Once both the CAD geometry and the initial mesh have been imported and digested, we
    // use the CAD surfaces and curves to define the projectors and assign them to the
    // manifold ids just specified.

    // A first projector is defined using the single wire contained in our CAD file.
    // The ArclengthProjectionLineManifold will make sure that every mesh edge located
    // on the wire is refined with a point that lies on the wire and splits in two equal arcs
    // the wire portion lying between the edge vertices.

    static OpenCASCADE::ArclengthProjectionLineManifold<2,3> line_projector(wires[0], tolerance);

    // Once the projector is created, we assign it to all the edges with manifold_id = 2
    tria.set_manifold(2, line_projector);

    // The surface projector is created according to what specified with the surface_projection_kind
    // option of the constructor.
    switch (surface_projection_kind)
      {
      case 0:
        // If surface_projection_kind value is 0, we select the NormalProjectionBoundary. The new mesh points will initially
        // generated at the baricenter of the cell/edge considered, and then projected
        // on the CAD surface along its normal direction.
        // The NormalProjectionBoundary constructor only needs a shape and a tolerance.
        static OpenCASCADE::NormalProjectionBoundary<2,3> normal_projector(bow_surface, tolerance);
        // Once created, the normal projector is assigned to the manifold having id 1.
        tria.set_manifold(1,normal_projector);
        break;
      case 1:
        // If surface_projection_kind value is 1, we select the DirectionalProjectionBoundary. The new mesh points will initially
        // generated at the baricenter of the cell/edge considere, and then projected
        // on the CAD surface along a direction that is specified to the DirectionalProjectionBoundary
        // constructor. In this case, the projection is done along the y-axis.
        static OpenCASCADE::DirectionalProjectionBoundary<2,3> directional_projector(bow_surface, Point<3>(0.0,1.0,0.0), tolerance);
        tria.set_manifold(1,directional_projector);
        break;
      case 2:
        // If surface_projection_kind value is 2, we select the NormaToMeshlProjectionBoundary. The new mesh points will initially
        // generated at the baricenter of the cell/edge considere, and then projected
        // on the CAD surface along a direction that is an estimate of the mesh normal direction.
        // The NormalToMeshProjectionBoundary constructor only requires a shape (containing at least a face)
        // and a tolerance.
        static OpenCASCADE::NormalToMeshProjectionBoundary<2,3> normal_to_mesh_projector(bow_surface, tolerance);
        tria.set_manifold(1,normal_to_mesh_projector);
        break;
      default:
        AssertThrow(false, ExcMessage("No valid projector selected: surface_projection_kind must be 0,1 or 2."));
        break;
      }

  }


  // @sect4{TriangulationOnCAD::refine_and_resize}

  // This function globally refines the mesh. In other tutorials, it tipically also distributes degrees
  // of freedom, and resizes matrices and vectors. These tasks are not carried out
  // here, since we are not running any simulation on the Triangulation produced.


  void TriangulationOnCAD::refine_and_resize()
  {
    tria.refine_global(1);
  }




  // @sect4{TriangulationOnCAD::output_results}

  // Outputting the results of our computations is a rather mechanical
  // tasks. All the components of this function have been discussed before.

  void TriangulationOnCAD::output_results(const unsigned int cycle)
  {

    std::string filename = ( output_filename + "_" +
                             Utilities::int_to_string(cycle) +
                             ".vtk" );
    std::ofstream logfile(filename.c_str());
    GridOut grid_out;
    grid_out.write_vtk(tria, logfile);


  }


  // @sect4{TriangulationOnCAD::run}

  // This is the main function. It should be self explanatory in its
  // briefness:

  void TriangulationOnCAD::run()
  {


    read_domain();
    unsigned int n_cycles = 5;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
      {
        refine_and_resize();
        output_results(cycle+1);
      }

  }
}


// @sect3{The main() function}

// This is the main function of this program. It is exactly like all previous
// tutorial programs:
int main ()
{
  try
    {
      using namespace dealii;
      using namespace Step54;


      deallog.depth_console (3);

      std::string in_mesh_filename = "initial_mesh_3d.vtk";

      cout<<"----------------------------------------------------------"<<endl;
      cout<<"Testing projection in direction normal to CAD surface"<<endl;
      cout<<"----------------------------------------------------------"<<endl;
      std::string out_mesh_filename = ( "3d_mesh_normal_projection" );
      TriangulationOnCAD tria_on_cad_norm(in_mesh_filename,out_mesh_filename,0);
      tria_on_cad_norm.run();
      cout<<"----------------------------------------------------------"<<endl;
      cout<<endl;
      cout<<endl;

      cout<<"----------------------------------------------------------"<<endl;
      cout<<"Testing projection in y-axis direction"<<endl;
      cout<<"----------------------------------------------------------"<<endl;
      out_mesh_filename = ( "3d_mesh_directional_projection" );
      TriangulationOnCAD tria_on_cad_dir(in_mesh_filename,out_mesh_filename,1);
      tria_on_cad_dir.run();
      cout<<"----------------------------------------------------------"<<endl;
      cout<<endl;
      cout<<endl;

      cout<<"----------------------------------------------------------"<<endl;
      cout<<"Testing projection in direction normal to mesh elements"<<endl;
      cout<<"----------------------------------------------------------"<<endl;
      out_mesh_filename = ( "3d_mesh_normal_to_mesh_projection" );
      TriangulationOnCAD tria_on_cad_norm_to_mesh(in_mesh_filename,out_mesh_filename,2);
      tria_on_cad_norm_to_mesh.run();
      cout<<"----------------------------------------------------------"<<endl;
      cout<<endl;
      cout<<endl;



    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}

