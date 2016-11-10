//
//  CLMatrix.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/5/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLMatrix.h"
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <strings.h>
#include "CLRandom.h"

void CLMatrixInit(CLMatrix * matrix, CLUInt rows, CLUInt columns, CLStringConst name)
{
	matrix->name = malloc(sizeof(CLChar) * 1024);
	strcpy(matrix->name, name);
	matrix->rows = rows;
	matrix->columns = columns;
	matrix->elements = matrix->rows * matrix->columns;
	matrix->size = sizeof(CLFloat) * matrix->elements;

	matrix->values = malloc(matrix->size);

}

void CLMatrixInitWithCLMem(CLMatrix * matrix, CLMem mem, CLUInt rows, CLUInt columns, CLStringConst name)
{
	CLMatrixInit(matrix, rows, columns, name);
	matrix->mem = mem;
}

void CLMatrixInitWithValues(CLMatrix * matrix, CLFloat * values, CLUInt rows, CLUInt columns, CLUInt copyValues, CLStringConst name)
{
	if (copyValues) {
		CLMatrixInit(matrix, rows, columns, name);
		memcpy(matrix->values, values, matrix->size);
	} else {
		matrix->name = malloc(sizeof(CLChar) * 1024);
		strcpy(matrix->name, name);		matrix->rows = rows;
		matrix->columns = columns;
		matrix->elements = matrix->rows * matrix->columns;
		matrix->size = sizeof(CLFloat) * matrix->elements;

		matrix->values = values;
	}
}

void CLMatrixFillRandom(CLMatrix * matrix)
{
	for (CLUInt i = 0; i < matrix->elements; ++i) {
		matrix->values[i] = CLRandomValue();
	}
}

void CLMatrixFillValue(CLMatrix * matrix, CLFloat value)
{
	for (CLUInt i = 0; i < matrix->elements; ++i) {
		matrix->values[i] = value;
	}
}

void CLMatrixUpdateValues(CLMatrix * matrix, const CLFloat * newValues)
{
	for (CLUInt i = 0; i < matrix->elements; ++i) {
		matrix->values[i] = newValues[i];
	}
}

void CLMatrixCreateMem(CLMatrix * matrix, CLContext context, CLMemFlags flags)
{
	matrix->mem = CLCreateBuffer(context, flags, matrix->size, matrix->name);
}

void CLMatrixCreateMemHostVar(CLMatrix * matrix, CLContext context, CLMemFlags flags)
{
	matrix->mem = CLCreateBufferHostVar(context, flags, matrix->size, matrix->values, matrix->name);
}

void CLMatrixUpdateValuesFromMem(CLMatrix * matrix, CLQueue queue)
{
	matrix->values = CLEnqueueReadBuffer(queue, matrix->mem, matrix->size, matrix->name);
}

CLMatrix * CLMatrixMultiply(CLMatrix * A, CLMatrix * B, CLMatrixTranspose aTranspose, CLMatrixTranspose bTranspose)
{
	CLUInt aRows = (aTranspose == CLMatrixTrans) ? A->columns : A->rows;
	CLUInt aColumns = (aTranspose == CLMatrixTrans) ? A->rows : A->columns;
	CLUInt bColumns = (bTranspose == CLMatrixTrans) ? B->rows : B->columns;

	CLMatrix * C = malloc(sizeof(CLMatrix));
	CLMatrixInit(C, aRows, bColumns, "C");

	for (CLUInt i = 0; i < aRows; ++i) {
		for (CLUInt j = 0; j < bColumns; ++j) {
			CLDouble sum = 0.0;
			for (CLUInt k = 0; k < aColumns; ++k) {
				CLFloat a = A->values[i * aColumns + k];
				CLFloat b = B->values[k * bColumns + j];
				sum += a * b;
			}
			C->values[i * bColumns + j] = sum;
		}
	}
	return C;
}

CLUInt CLMatrixCompare(CLMatrix * A, CLMatrix * B)
{
	if (A->elements != B->elements)
		return CLFalse;

	for (CLUInt i = 0; i < A->elements; ++i) {
		if (fabs(A->values[i] - B->values[i]) > CLFTollerance) {
			return CLFalse;
		}
	}
	
	return CLTrue;
}

void CLMatrixPrint(CLMatrix * matrix, CLMatrixTranspose transpose)
{
	printf("\n/** %s **/\n", matrix->name);
	CLUInt rows = (transpose == CLMatrixTrans) ? matrix->columns : matrix->rows;
	CLUInt cols = (transpose == CLMatrixTrans) ? matrix->rows : matrix->columns;

	for (CLUInt r = 0; r < rows; ++r) {
		for (CLUInt c = 0; c < cols; ++c) {
			CLUInt index = (transpose == CLMatrixTrans) ? c * cols + r : r * cols + c;
			printf("%8.5g ", matrix->values[index]);
//			printf("%+0.2f ", matrix->values[index]);

		}
		printf("\n");
	}
	printf("/** %s - END **/\n", matrix->name);
}

void CLMatrixPrintStats(CLMatrix * matrix)
{
	printf("%s: %d x %d = %d - %zu\n", matrix->name, matrix->rows, matrix->columns, matrix->elements, matrix->size);
}

void CLMatrixReleaseMem(CLMatrix * matrix)
{
	if (matrix->mem != NULL) {
		CLReleaseMemObject(matrix->mem, matrix->name);
		matrix->mem = NULL;
	}
}

void CLMatrixRelease(CLMatrix * matrix)
{
	CLMatrixReleaseMem(matrix);
	free(matrix->name);
	matrix->name = NULL;
	free(matrix->values);
	matrix->values = NULL;
	free(matrix);
	matrix = NULL;
}