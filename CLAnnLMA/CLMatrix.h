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
	CLMem mem;
	CLUInt rows;
	CLUInt columns;
	CLUInt elements;
	CLSize size;
	CLFloat * values;

} CLMatrix;

void CLMatrixInit(CLMatrix * matrix, CLUInt rows, CLUInt columns, CLStringConst name);
void CLMatrixInitWithCLMem(CLMatrix * matrix, CLMem mem, CLUInt rows, CLUInt columns, CLStringConst name);
void CLMatrixInitWithValues(CLMatrix * matrix, CLFloat * values, CLUInt rows, CLUInt columns, CLUInt copyValues, CLStringConst name);

void CLMatrixFillRandom(CLMatrix * matrix);
void CLMatrixFillValue(CLMatrix * matrix, CLFloat value);
void CLMatrixUpdateValues(CLMatrix * matrix, const CLFloat * newValues);

void CLMatrixCreateMem(CLMatrix * matrix, CLContext context, CLMemFlags flags);
void CLMatrixCreateMemHostVar(CLMatrix * matrix, CLContext context, CLMemFlags flags);
void CLMatrixUpdateValuesFromMem(CLMatrix * matrix, CLQueue queue);

CLMatrix * CLMatrixMultiply(CLMatrix * A, CLMatrix * B, CLMatrixTranspose aTranspose, CLMatrixTranspose bTranspose);
CLUInt CLMatrixCompare(CLMatrix * A, CLMatrix * B);

void CLMatrixPrint(CLMatrix * matrix, CLMatrixTranspose transpose);
void CLMatrixPrintStats(CLMatrix * matrix);

void CLMatrixReleaseMem(CLMatrix * matrix);
void CLMatrixRelease(CLMatrix * matrix);

#endif /* CLMatrix_h */
