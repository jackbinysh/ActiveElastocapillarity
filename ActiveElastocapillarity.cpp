#include "libmd/libmd.h"
#include <vtkSmartPointer.h>
#include <vtkXMLUnstructuredGridReader.h>
#include <vtkXMLUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkPoints.h>
#include <vtkUnsignedCharArray.h>
#include <string>
using namespace std;

int main(int argc,char *argv[])
{

  int system_size=6;
  string fileName = "TestMesh.vtu";
  const ui dim=3;

  // Initialise the MD simulation
	md<dim> sys(system_size);
	sys.index();
	sys.network.update = false;
  

  // Read in the mesh
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName (fileName.c_str());
  reader->Update();
  auto InputMesh = reader->GetOutput();

  // tell the MD sim about particle positions
  for (vtkIdType i = 0; i < InputMesh->GetNumberOfPoints() ; i++)
  {
    double* x;
    x=InputMesh->GetPoint(i);

    if(2==dim)
    {
      sys.import_pos(i,x[0],x[1]);
    }
    else if(3==dim)
    {
      sys.import_pos(i,x[0],x[1],x[2]);
    }
  }

  // tell the MD sim about particle bonds
  auto cellArray = InputMesh->GetCells();
  cellArray->InitTraversal();
  vtkIdType npts, *pts;
  while(cellArray->GetNextCell(npts, pts))
  {
    if(2==npts) // i.e if the cell is an edge
    {
      //cout <<pts[0] << "\t" <<pts[1] << "\n";
      bool made = sys.add_bond(pts[0], pts[1], POT::HOOKEAN); //add spring, neohookean potential
      cout << made;
    }
  }

  // Run the sim
  ui run_time = 1000000;
	ldf time_step = 1E-7; 
	sys.integrator.h = time_step;
	for (ui i = 0; i < run_time; i++) 
  {
		sys.timestep(); 
  }
  
  //output
	for (ui i = 0; i < sys.particles.size(); i++) 
  {
    ldf x[3];
    x[2]=0;
    for(int d=0;d<dim;d++)
    {
      x[d]=sys.particles[i].x[d];
      cout <<x[d] << "\t";
    }
    InputMesh->GetPoints()->SetPoint(i,x);
  }
  string outputname = "out.vtu";
  auto writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
  writer->SetFileName(outputname.c_str());
  writer->SetInputData(InputMesh);
  writer->Write();



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

  return 0;
}