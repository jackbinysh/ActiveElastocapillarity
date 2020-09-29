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

  // Initialise the MD simulation
	//md<2> sys(system_size);
	//sys.index();
	//sys.network.update = false;

  // Read in the mesh
  string fileName = "TestMesh.vtu";
  vtkSmartPointer<vtkUnstructuredGrid> InputMesh;
  auto reader = vtkSmartPointer<vtkXMLUnstructuredGridReader>::New();
  reader->SetFileName (fileName.c_str());
  reader->Update();
  InputMesh = reader->GetOutput();

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

  auto cellArray = InputMesh->GetCellData();
  //cout << cellArray->GetClassName();
  //InputMesh->GetCellData()->GetScalars();

 for (vtkIdType i = 0; i <InputMesh->GetCellData()->GetScalars()->GetNumberOfTuples(); i++)
   {
       double* x= InputMesh->GetCellData()->GetScalars()->GetTuple(i);
       cout << *x <<endl;
   }


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
