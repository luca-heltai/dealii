// ---------------------------------------------------------------------
// $Id$
//
// Copyright (C) 2014 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


#ifndef __deal2__occ_utilities_h
#define __deal2__occ_utilities_h

#ifdef DEAL_II_WITH_OPENCASCADE

#include <string>
#include <TopoDS_Shape.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Geom_Plane.hxx>
#include <Geom_Curve.hxx>
#include <gp_Pnt.hxx>

#include <base/point.h>

DEAL_II_NAMESPACE_OPEN

/**
 * @addtogroup OpenCASCADE
 * @{
 * We collect in this namespace all utilities which operate on
 * OpenCASCADE entities. OpenCASCADE splits every object into a
 * topological description and a geometrical entity. The basic
 * topological description is a TopoDS_Shape. TopoDS_Shapes are light
 * objects, and can be copied around. The closest deal.II analog is a
 * TriaIterator.
 *
 * The OpenCASCADE topology is designed with reference to the STEP
 * standard ISO-10303-42.  The structure is an oriented one-way graph,
 * where parents refer to their children, and there are no back
 * references. Abstract structure is implemented as C++ classes from
 * the TopoDS package. A TopoDS_Shape is manipulated by value and
 * contains 3 fields: location, orientation and a myTShape handle (of
 * the TopoDS_TShape type). According to OpenCASCADE documentation,
 * myTShape and Location are used to share data between various shapes
 * and thus save huge amounts of memory. For example, an edge
 * belonging to two faces has equal Locations and myTShape fields but
 * different Orientations (Forward in context of one face and Reversed
 * in one of the other).
 *
 * Valid shapes include collection of other shapes, solids, faces,
 * edges, vertices, etc.
 *
 * Once a topological description is available, if a concrete
 * geometrical object can be created, the BRep classes allow one to
 * extract the actual geometrical information from a shape.
 *
 * This is done by inheriting abstract topology classes from the
 * TopoDS package by those implementing a boundary representation
 * model (from the BRep package). Only 3 types of topological objects
 * have geometric representations – vertex, edge, and face.
 *
 * Every TopoDS_Shape can be queried to figure out what type of shape
 * it is, and actual geometrical objects, like surfaces, curves or
 * points, can be extracted using BRepTools.
 *
 * In this namespace we provide readers and writers that read standard
 * CAD files, and return a TopoDS_Shape, or that write a CAD file,
 * given a TopoDS_Shape. Most of the functions in the OpenCASCADE
 * namespace deal with TopoDS_Shapes of one type or another, and
 * provide interfaces to common deal.II objects, like Triangulation,
 * Manifold, and so on.
 *
 * Notice that these tools are only useful when spacedim is equal to
 * three, since OpenCASCADE only operates in three-dimensional mode.
 *
 * @author Luca Heltai, Andrea Mola, 2011--2014.
 */
namespace OpenCASCADE 
{
  /**
   * Count the subobjects of a shape. This function just outputs some
   * information about the TopoDS_Shape passed as argument. It counts
   * the number of faces, edges and vertices (the only topological
   * entities associated with actual geometries) which are contained
   * in the given shape.
   */
  void count_elements(const TopoDS_Shape &shape,
		      unsigned int &n_faces,
		      unsigned int &n_edges,
		      unsigned int &n_vertices);
  
  /**
   * Read IGES files and translate their content into openCascade
   * topological entities. The option scale_factor is used to
   * compensate for different units being used in the IGES files and
   * in the target application. The standard unit for IGES files is
   * millimiters. The return object is a TopoDS_Shape which contains
   * all objects from the file. 
   */
  TopoDS_Shape read_IGES(const std::string &filename, 
			 const double scale_factor=1e-3);
  
  /**
   * Perform the intersection of the given topological shape with the
   * plane $c_x x + c_y y + c_z z +c = 0$. The returned topological
   * shape will contain as few bsplines as possible. An exception is
   * thrown if the intersection produces an empty shape.
   */
  TopoDS_Shape  intersect_plane(const TopoDS_Shape &in_shape,
				const double c_x,
				const double c_y,
				const double c_z,
				const double c,
				const double tolerance=1e-7);

  
  /**
   * Creates a 3D smooth BSpline curve passing through the points in
   * the assigned vector. The points are reordered internally
   * according to their scalar product with the direction, if
   * direction is different from zero, otherwise they are used as
   * passed. Notice that this function changes the input points if
   * required by the algorithm.
   *
   * This class is used to interpolate a BsplineCurve passing through
   * an array of points, with a C2 Continuity.
   */
  TopoDS_Shape interpolation_curve(std::vector<dealii::Point<3> >  &curve_points,
				   const dealii::Point<3> direction=dealii::Point<3>(), 
				   const double tolerance=1e-7);

  /**
   * Convert OpenCASCADE point into a Point<3>.
   */
  inline dealii::Point<3> Pnt(const gp_Pnt &p)
  {
    dealii::Point<3> P(p.X(), p.Y(), p.Z());
    return P;
  }

  /**
   * Convert Point<3> into OpenCASCADE point.
   */
  inline gp_Pnt Pnt(const dealii::Point<3> &p);
  
  /**
   * Sort two points according to their scalar product with
   * direction. If the norm of the direction is zero, then use
   * lexycographical ordering.
   */
  inline bool point_compare(const dealii::Point<3> &p1, const dealii::Point<3> &p2,
			    const dealii::Point<3> direction=Point<3>(),
			    const double tolerance=1e-10);
  
  DeclException1(ExcOpenCASCADEStatus, IFSelect_ReturnStatus,<<"Error! OpenCASCADE function returned "<<arg1);
				      
}
/*@}*/

DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_WITH_OPENCASCADE

/*------------------------------ occ_utilities.h ------------------------------*/
#endif
/*------------------------------ occ_utilities.h ------------------------------*/
