#include <deal.II/grid/occ_utilities.h>


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <set>

#include <IGESControl_Reader.hxx>
#include <IGESControl_Controller.hxx>

// #include <IGESControl_Writer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
// #include <TopoDS_Shell.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Edge.hxx>
#include <BRepTools.hxx>

// #include <XSControl_Reader.hxx>
// #include <TopTools_SequenceOfShape.hxx>
// #include <Handle_Standard_Transient.hxx>
// #include <TColStd_SequenceOfTransient.hxx>
// #include <TColStd_HSequenceOfTransient.hxx>
#include <TopExp_Explorer.hxx>
// #include <gp_Pnt.hxx>
// #include <gp_Vec.hxx>
// #include <GeomAPI_ProjectPointOnSurf.hxx>
// #include <GeomAPI_ProjectPointOnCurve.hxx>
// #include <Standard_Real.hxx>
// #include <Standard_Integer.hxx>
// #include <BRep_Tool.hxx>
// #include <Geom_Surface.hxx>
// #include <Geom_Plane.hxx>
// #include <Prs3d_ShapeTool.hxx>
// #include <GeomAPI_IntSS.hxx>
// #include <Bnd_Box.hxx>
// #include <gp_Trsf.hxx>
// #include <gp_Ax3.hxx>
// #include <gp_Pln.hxx>
// #include <BRepBuilderAPI_Transform.hxx>
// #include <GeomConvert_CompCurveToBSplineCurve.hxx>
// #include <BRepBuilderAPI_MakeEdge.hxx>
// #include <TColGeom_Array1OfCurve.hxx>
// #include <TColgp_HArray1OfPnt.hxx>
// #include <Geom_Curve.hxx>
// #include <Geom_BoundedCurve.hxx>
// #include <Geom_TrimmedCurve.hxx>
// #include <Geom_BSplineCurve.hxx>
// #include <GeomAPI_Interpolate.hxx>
// #include <BRepAlgo_Section.hxx>
// #include <GeomLib_Tool.hxx>
// #include <TColGeom_Array2OfBezierSurface.hxx>
// #include <ProjLib_ProjectOnPlane.hxx>
// #include <Adaptor3d_HCurve.hxx>
// #include <GeomAdaptor_HCurve.hxx>
// #include <ShapeAnalysis_Surface.hxx>
// #include <GeomLProp_SLProps.hxx>
// #include <BRepExtrema_DistShapeShape.hxx>
// #include <BRepBuilderAPI_MakeVertex.hxx>
// #include <GCPnts_AbscissaPoint.hxx>
// #include <GeomLProp_CLProps.hxx>
#include <deal.II/base/utilities.h>

#include <boost/bind.hpp>



#include <BRep_Builder.hxx>

#define FALSE 0
#define TRUE 1


#include <deal.II/base/exceptions.h>
#include <vector>
#include <algorithm>

using namespace dealii;
using namespace std;

namespace OpenCascade 
{
  
  TopoDS_Shape read_IGES(const string filename, 
			 const double scale_factor)
  {
    IGESControl_Reader reader;
    IFSelect_ReturnStatus stat;
    stat = reader.ReadFile(filename.c_str()); 
    Assert(stat == IFSelect_RetDone, ExcOpenCASCADEStatus(stat));
	   
    Standard_Boolean failsonly = Standard_False;
    IFSelect_PrintCount mode = IFSelect_ItemsByEntity;
    reader.PrintCheckLoad (failsonly, mode);
    Standard_Integer nIgesFaces,nTransFaces;

    Handle(TColStd_HSequenceOfTransient) myList = reader.GiveList("iges-faces");
				     //selects all IGES faces in the
				     //file and puts them into a list
				     //called MyList,
    nIgesFaces = myList->Length(); 
    nTransFaces = reader.TransferList(myList);

    AssertThrow(nTransFaces > 0, ExcMessage("Read nothing from file."));

				     // Handle IGES Scale here.
    gp_Pnt Origin;
    gp_Trsf scale;
    scale.SetScale (Origin, scale_factor);

    TopoDS_Shape sh = reader.OneShape();
    BRepBuilderAPI_Transform trans(sh, scale);
    
    return trans.Shape();   // this is the actual translation
  }
  

   TopoDS_Shape intersect_plane(const TopoDS_Shape &in_shape,
				const double c_x,
				const double c_y,
				const double c_z,
				const double c,
				const double tolerance) 
   {
     TopoDS_Shape out_shape;
     Handle(Geom_Plane) plane = new Geom_Plane(c_x,c_y,c_z,c);

     // This is to loop on the faces,
     // that extracts all
     // intersections.
   
     //TopExp_Explorer faceExplorer(in_shape , TopAbs_FACE);

     std::vector<Handle_Geom_BoundedCurve> intersections;
 
     BRepAlgo_Section section(in_shape, plane);
     TopoDS_Shape edges = section.Shape();

     TopoDS_Edge edge;
     TopLoc_Location L;
     Standard_Real First;
     Standard_Real Last;   
     gp_Pnt PIn(0.0,0.0,0.0);
     gp_Pnt PFin(0.0,0.0,0.0);
     gp_Pnt PMid(0.0,0.0,0.0); 
     TopExp_Explorer edgeExplorer(edges , TopAbs_EDGE);
     unsigned int count = 0;
     while (edgeExplorer.More())
       {
	 edge = TopoDS::Edge(edgeExplorer.Current());
	 Handle(Geom_Curve) curve = BRep_Tool::Curve(edge,L,First,Last);
	 intersections.push_back(Handle(Geom_BoundedCurve)::DownCast(curve));
	 edgeExplorer.Next();
       }
  
     // Now we build a single bspline out of all the geometrical
     // curves
     unsigned int numIntersEdges = intersections.size();  
     for (unsigned int i = 0;i<intersections.size();++i)
       {
	 if (intersections[i]->Value(intersections[i]->FirstParameter()).X() > 
	     intersections[i]->Value(intersections[i]->LastParameter()).X()  )
	   intersections[i]->Reverse();
       }

     GeomConvert_CompCurveToBSplineCurve
       convert_bspline(intersections[0], Convert_TgtThetaOver2);
     bool check = false, one_added = true, one_failed=true;
     vector<bool> added(numIntersEdges, false);
     added[0] = true;
     while(one_added == true) 
       {
	 one_added = false;
	 one_failed = false;
	 for (unsigned int i=1; i<numIntersEdges; ++i)
	   if(added[i] == false) 
	     { 
	       Handle(Geom_Curve) curve = intersections[i];
	       Handle(Geom_BoundedCurve) bcurve = Handle(Geom_BoundedCurve)::DownCast(curve);
	       check = convert_bspline.Add(bcurve,tolerance,0,1,0);
	       one_failed = one_failed || (check == false);
	       one_added = one_added || (check == true);
	       added[i] = check;
	       //cout<<i<<" -->  "<<added[i]<<"  "<<false<<endl;
	     }
       }
 
     Assert(one_failed == false,
	    ExcMessage("Bspline convertion of intersection with plane has failed."));

     Handle(Geom_Curve) bspline = convert_bspline.BSplineCurve();

     out_shape = BRepBuilderAPI_MakeEdge(bspline);

     if (bspline->IsCN(1))
       cout<<"Intersection with plane is at least a C1 curve"<<endl;
     else
       cout<<"Intersection with plane is not a C1 curve "<<endl;
     
   }



  TopoDS_Shape interpolation_curve(std::vector<Point<3> > &curve_points,
				   const Point<3> direction,
				   const double tolerance)
  {

    unsigned int n_vertices = curve_points.size();

    if(direction*direction > 0) 
      {
	std::sort(curve_points.begin(), curve_points.end(),
		  boost::bind(&point_compare, _1, _2, direction));
      }

				     // set up array of vertices
    Handle(TColgp_HArray1OfPnt) vertices = new TColgp_HArray1OfPnt(1,n_vertices);
    for (unsigned int vertex=0; vertex<n_vertices; ++vertex)
      {
	vertices->SetValue(vertex+1,OpenCascade::Pnt(curve_points[vertex]));
      }


    GeomAPI_Interpolate bspline_generator(vertices, false, tolerance);
    bspline_generator.Perform();
    Assert( (bspline_generator.IsDone()), ExcMessage("Interpolated bspline generation failed"));
    
    Handle(Geom_BSplineCurve) bspline = bspline_generator.Curve();
    TopoDS_Shape out_shape = BRepBuilderAPI_MakeEdge(bspline);
    return out_shape;
  }

} // end namespace


