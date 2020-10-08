#include "libmd/libmd.h"
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <vtkCellArrayIterator.h>
#include <string>
using namespace std;

// dimensionality of the sim
const ui dim = 2;
// Simulation parameters
ui run_time = 1000000;
ui LogInterval=1000;
ldf time_step = 1E-3; 
// Input and Output
const string InputMeshFilename = "Disk.vtu";
const string DataDir="Data/";
const string OutputMeshPrefix="Disk";

int main(int argc,char *argv[])
{
  // Read in the mesh
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName (InputMeshFilename.c_str());
  reader->Update();
  auto InputMesh = reader->GetOutput();

  // Initialise the MD simulation
  int N = InputMesh->GetNumberOfPoints();
	md<dim> sys(N);
	sys.index();
	sys.network.update = false;

  // tell the MD sim about particle positions
  std::vector<double>x(N);
  std::vector<double>y(N);
  std::vector<double>z(N);
  for (vtkIdType i = 0; i < InputMesh->GetNumberOfPoints() ; i++)
  {
    double temp[3];
    InputMesh->GetPoint(i,temp);
    x[i]=temp[0];
    y[i]=temp[1];
    z[i]=temp[2];
  }
  sys.import_pos(x.data(),y.data());

  // tell the MD sim about initial particle velocities
  std::fill(x.begin(), x.end(), 0);
  std::fill(y.begin(), y.end(), 0);
  std::fill(z.begin(), z.end(), 0);
  sys.import_vel(x.data(),y.data());

  for (ui i = 0; i < sys.particles.size(); i++) 
  {
    ldf x[3];
    x[2]=0;
    for(int d=0;d<dim;d++)
    {
      x[d]=sys.particles[i].x[d];
      cout << x[d] << "\t" ; 
    }
    for(int d=0;d<dim;d++)
    {
      x[d]=sys.particles[i].dx[d];
      cout << x[d] << "\t" ; 
    }
    cout << endl;
  }

  /*
    if(2==dim)
    {

      sys.import_pos(i,x[0],x[1]);
    }
    else if(3==dim)
    {
      sys.import_pos(i,x[0],x[1],x[2]);
    }
  }
  */

  /*
  for (vtkIdType i = 0; i < InputMesh->GetNumberOfPoints() ; i++)
  {
    double x[3];
    InputMesh->GetPoint(i,x);

    if(2==dim)
    {
      sys.import_pos(i,x[0],x[1]);
    }
    else if(3==dim)
    {
      sys.import_pos(i,x[0],x[1],x[2]);
    }
  }
  */

  // tell the MD sim about particle bonds

  
  auto cellArray = InputMesh->GetCells();
  auto iter = vtk::TakeSmartPointer(cellArray->NewIterator());
  auto BoundaryFlags = InputMesh->GetCellData()->GetScalars();
  for (iter->GoToFirstCell(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    // get the num of points in the cell, and those points
    vtkIdType npts;
    vtkIdType const *pts;
    iter->GetCurrentCell(npts,pts);

    // get whether its a boundary cell or not
    // 
    
    int i = iter->GetCurrentCellId();
    double* x= BoundaryFlags->GetTuple(i);
    int bflag= round(x[0]);

    if(2==npts) // i.e if the cell is an edge
    {
      double r0[3];
      InputMesh->GetPoint(pts[0],r0);
      double r1[3]; 
      InputMesh->GetPoint(pts[1],r1);
      double restlength = sqrt((r1[0]-r0[0])*(r1[0]-r0[0])+(r1[1]-r0[1])*(r1[1]-r0[1])+(r1[2]-r0[2])*(r1[2]-r0[2]));
      bflag=0;
      if(!bflag)
      {
        cout << pts[0] << "\t" << pts[1] << endl;
        cout << restlength << endl;
        vector<ldf> params = {0.1,2*restlength};
        bool made = sys.add_bond(pts[0], pts[1], POT::HOOKEAN, params); 
      }
      if(bflag)
      {
        cout << "hi" << endl;
        vector<ldf> params = {0.1,2*restlength+0.0};
        bool made = sys.add_bond(pts[0], pts[1], POT::HOOKEAN, params); 
      }
    }
    if(3==npts)
    {
      vector<ldf> params = {0.1,1.0};
      sys.add_bond(pts[0], pts[1], POT::HOOKEAN, params); 
      sys.add_bond(pts[0], pts[2], POT::HOOKEAN, params); 
      sys.add_bond(pts[1], pts[2], POT::HOOKEAN, params);
    }
    
  }
  
  
  /*
  auto cellArray = InputMesh->GetCells();
  cellArray->InitTraversal();
  vtkIdType npts;
 const vtkIdType *pts;
  while(cellArray->GetNextCell(npts, pts))
  {
    if(2==npts) // i.e if the cell is an edge
    {

      double r0[3];
      InputMesh->GetPoint(pts[0],r0);
      double r1[3]; 
      InputMesh->GetPoint(pts[1],r1);

      double restlength = sqrt((r1[0]-r0[0])*(r1[0]-r0[0])+(r1[1]-r0[1])*(r1[1]-r0[1])+(r1[2]-r0[2])*(r1[2]-r0[2]));

      vector<ldf> params = {0.1,restlength};
      bool made = sys.add_bond(pts[0], pts[1], POT::HOOKEAN, params); 

    }
    if(3==npts)
    {
      vector<ldf> params = {0.1,1.0};
     
      sys.add_bond(pts[0], pts[1], POT::HOOKEAN, params); 
      sys.add_bond(pts[0], pts[2], POT::HOOKEAN, params); 
      sys.add_bond(pts[1], pts[2], POT::HOOKEAN, params);
    }
  }
  */
  
  
  // Tell the MD sim what kind of damping/dissipation we want
  /*
  vector<ldf> dampingcoeff = {1};
  ui dampingForceIndex = sys.add_forcetype(EXTFORCE::DAMPING,dampingcoeff);
  sys.assign_all_forcetype(dampingForceIndex);
  */

  // Run the sim
  

	sys.integrator.h = time_step;
	for (ui i = 0; i < run_time; i++) 
  {
    // log output
    if(0==i%100)
    {
      for (ui i = 0; i < sys.particles.size(); i++) 
      {
        ldf x[3];
        x[2]=0;
        for(int d=0;d<dim;d++)
        {
          x[d]=sys.particles[i].x[d];
          cout << x[d] << "\t" ; 
        }
        cout << endl;
        InputMesh->GetPoints()->SetPoint(i,x);
      }
      string outputname = DataDir+OutputMeshPrefix+std::to_string(i)+".vtu";
      auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
      writer->SetFileName(outputname.c_str());
      writer->SetInputData(InputMesh);
      writer->Write();
    }
		sys.timestep(); 
  }
  

  return 0;
}





  
  //output



  //  cout <<npts << endl;
  //  cout <<pts[0] << "\t" << pts[1] << "\t" << endl;

  /*
  auto cellArray = InputMesh->GetCells();
   vtkIdType numCells = cellArray->GetNumberOfCells();
   vtkIdType cellLocation = 0; // the index into the cell array

   for (vtkIdType i = 0; i < numCells; i++)
     {
     vtkIdType numIds; // to hold the size of the cell
     vtkIdType *pointIds; // to hold the ids in the cell

     cellArray->GetCell(cellLocation, numIds, pointIds);
     cellLocation += 1 + numIds;
     cout << pointIds[0] << pointIds[1]<< endl;
   }
   */



/*  
  auto cellArray = InputMesh->GetCells();
   vtkIdType numCells = cellArray->GetNumberOfCells();
   vtkIdType cellLocation = 0; // the index into the cell array

   for (vtkIdType i = 0; i < numCells; i++)
     {
     vtkIdType numIds; // to hold the size of the cell
     vtkIdType *pointIds; // to hold the ids in the cell

     cellArray->GetCell(cellLocation, numIds, pointIds);
     cellLocation += 1 + numIds;
     cout << pointIds[0] << pointIds[1]<< endl;
   }
   */

  /*
  auto cellArray = InputMesh->GetCellData();
  //cout << cellArray->GetClassName();
  //InputMesh->GetCellData()->GetScalars();

 for (vtkIdType i = 0; i <InputMesh->GetCellData()->GetScalars()->GetNumberOfTuples(); i++)
   {
       double* x= InputMesh->GetCellData()->GetScalars()->GetTuple(i);
       cout << *x <<endl;
   }
   */


/*
  auto cellArray = InputMesh->GetCells();
  cellArray->InitTraversal();
  vtkIdType npts, *pts;
  while(cellArray->GetNextCell(npts, pts))
  {
    cout <<npts << endl;
    cout <<pts[0] << "\t" << pts[1] << "\t" << endl;

  }
  */
  /*
  for (iter->GoToFirstCell(); !iter->IsDoneWithTraversal(); iter->GoToNextCell())
  {
    // do work with iter
    iter->GetCurrentCell(numCellPts, cellPts);
    cout << numCellPts;
  }
  */



/*
  auto CellTypeData = InputMesh->GetCellTypesArray();
  CellTypeData->Print(std::cout);
  //cout<< CellData->GetNumberOfCells();
  for(vtkIdType t=0; t<CellData->GetNumberOfCells(); t++)
  {
    //CellData->GetCell(t);
    auto tup = CellTypeData->GetTuple(t);
    cout << *tup << endl ;
  }

  */

  //double x[3];
  //data->GetPoint(3,x);
   // ->Print(std::cout);

  // Tell the MD simulation about the initial positions of the nodes
  // 

  // Tell the MD simulation about the mesh edges, i.e. the bonds 

    // Write file
//  string outputname = "out.vtu";
//  vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
//  writer->SetFileName(outputname.c_str());
//  writer->SetInputData(unstructuredGrid);
//  writer->Write();
//
//
//      cout <<r0[0] << "\t" << r0[1] << "\t" << r0[2] << "\t" << endl;
//      cout <<r1[0] << "\t" << r1[1] << "\t" << r1[2] << "\t" << endl;
//      cout <<restlength << endl<<endl;
//
