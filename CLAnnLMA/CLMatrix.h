//
//  CLMatrix.h
//  CLAnnLMA
//
//  Created by Albertomac on 11/5/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef CLMatrix_h
#define CLMatrix_h

#include "CLManager.h"

typedef enum _CLMatrixTranspose {
	CLMatrixTrans,
	CLMatrixNoTrans,

} CLMatrixTranspose;

typedef struct {

	CLString name;
	CLSize offsetMem;
	CLMem mem;
	CLUInt rows;
	CLUInt columns;
	CLUInt elements;
	CLSize size;
	CLNetDataType * values;

} CLMatrix;

//Init
void CLMatrixInit(CLMatrix * matrix, CLUInt rows, CLUInt columns, CLStringConst name);
void CLMatrixInitWithCLMem(CLMatrix * matrix, CLMem mem, CLUInt rows, CLUInt columns, CLStringConst name);
void CLMatrixInitWithValues(CLMatrix * matrix, CLNetDataType * values, CLUInt rows, CLUInt columns, CLUInt copyValues, CLStringConst name);

//CSV
void CLMatrixInitWithCSV(CLMatrix * matrix, CLStringConst file);
void CLMatrixSaveCSV(CLMatrix * matrix, CLStringConst file);

//Values
void CLMatrixFillRandom(CLMatrix * matrix);
void CLMatrixFillRandomWithMult(CLMatrix * matrix, CLNetDataType mult, CLNetDataType shift);
void CLMatrixFillValue(CLMatrix * matrix, CLNetDataType value);
void CLMatrixUpdateValues(CLMatrix * matrix, const CLNetDataType * newValues);
void CLMatrixNormalize(CLMatrix * matrix);

//CLMem
void CLMatrixCreateMem(CLMatrix * matrix, CLContext context, CLMemFlags flags);
void CLMatrixCreateMemHostVar(CLMatrix * matrix, CLContext context, CLMemFlags flags);
void CLMatrixUpdateValuesFromMem(CLMatrix * matrix, CLQueue queue);

//Operations
void CLMatrixMultiply(CLMatrix * A, CLMatrix * B, CLMatrix * C, CLMatrixTranspose aTranspose, CLMatrixTranspose bTranspose);
CLUInt CLMatrixCompare(CLMatrix * A, CLMatrix * B);

//Useful stuffs
void CLMatrixPrint(CLMatrix * matrix, CLMatrixTranspose transpose);
void CLMatrixPrintStats(CLMatrix * matrix);

//Release
void CLMatrixCleanUp(CLMatrix * matrix);
#define CLMatrixRelease(matrix) do {CLMatrixCleanUp(matrix); matrix = NULL;} while(0);

#endif /* CLMatrix_h */
