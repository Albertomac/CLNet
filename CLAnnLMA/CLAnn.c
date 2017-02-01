//
//  CLAnn.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/6/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLAnn.h"
#include <math.h>

#ifdef __APPLE__
#include </usr/local/include/clBLAS.h>
#else
#include <clBLAS.h>
#endif

#include "CLBenchmark.h"
#include "CLRandom.h"

#define kClean "clean"

#define kActivationSigmoid "activationSigmoid"
#define kActivationTansig "activationTansig"
#define kActivationRadbas "activationRadbas"

#define kChiSquared "chiSquared"
#define kChiSquaredReduce "chiSquaredReduce"
#define kJacobian "jacobian"
#define kDelta "delta"
#define kCholeskyDecomposition "choleskyDecomposition"

CLUInt BLOCK_SIZE = 32;

void CLAnnInit(CLAnn * nn, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddenLayers, CLActivation * activationPerLayer,  CLUInt * neuronsPerLayer, CLUInt nTargets, CLStringConst name)
{
	nn->name = malloc(sizeof(CLChar) * 1024);
	strcpy(nn->name, name);
	nn->nPatterns = nPatterns;
	nn->nInputs = nInputs;
	nn->nHiddenLayers = nHiddenLayers;
	nn->activationPerLayer = malloc(sizeof(CLActivation) * nn->nHiddenLayers);
	memcpy(nn->activationPerLayer, activationPerLayer, sizeof(CLActivation) * nn->nHiddenLayers);
	nn->neuronsPerLayer = malloc(sizeof(CLUInt) * nn->nHiddenLayers);
	memcpy(nn->neuronsPerLayer, neuronsPerLayer, sizeof(CLUInt) * nn->nHiddenLayers);
	nn->nTargets = nTargets;

	printf("/** %s **/\n"
		   "nPatters: %d\n"
		   "%d ", nn->name, nn->nPatterns, nn->nInputs);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		printf("x %d ", nn->neuronsPerLayer[i]);
	}
	printf("x %d\n/** %s **/\n", nn->nTargets, nn->name);

	nn->inputs = malloc(sizeof(CLMatrix));
	nn->targets = malloc(sizeof(CLMatrix));

	nn->weights = malloc(sizeof(CLMatrix));
	nn->weightsTemp = malloc(sizeof(CLMatrix));

	nn->weightsForLayer = malloc(sizeof(CLMatrix) * nn->nHiddenLayers + 1);
	for (CLUInt i = 0; i < nn->nHiddenLayers + 1; ++i) {
		nn->weightsForLayer[i] = malloc(sizeof(CLMatrix));
	}

	nn->outputs = malloc(sizeof(CLMatrix));

	nn->hActivations = malloc(sizeof(CLMatrix));
	nn->jacobian = malloc(sizeof(CLMatrix));
	nn->hessian = malloc(sizeof(CLMatrix));
	nn->d = malloc(sizeof(CLMatrix));
	nn->delta = malloc(sizeof(CLMatrix));
	nn->cholesky = malloc(sizeof(CLMatrix));

	nn->verbose = CLTrue;
	nn->maxIteration = 10000;
	nn->initialLambda = 0.0001;
	nn->upFactor = 10.0f;
	nn->downFactor = 10.0f;
	nn->targetDeltaError = 1e-12f;
	nn->finalError = 0.0f;
	nn->finalDeltaError = 0.0f;
	nn->inputsCopiedIntoJacobian = CLFalse;

	CLUInt nWeights = nn->nInputs * nn->neuronsPerLayer[0];

	for (CLUInt i = 1; i < nn->nHiddenLayers; ++i) {
		nWeights += nn->neuronsPerLayer[i - 1] * nn->neuronsPerLayer[i];
	}
	nWeights += nn->neuronsPerLayer[nn->nHiddenLayers - 1] * nn->nTargets;

	CLMatrixInit(nn->inputs, nn->nPatterns, nn->nInputs, "inputs");
	CLMatrixInit(nn->targets, nn->nPatterns, nn->nTargets, "targets");

	CLMatrixInit(nn->weights, 1, nWeights, "weights");
	CLMatrixInit(nn->weightsTemp, 1, nWeights, "weightsTemp");

	CLString weightsName = malloc(sizeof(CLChar) * 32);

	//Inputs x layer[0]
	snprintf(weightsName, 31, "weightsForLayer[%d]", 0);
	CLMatrixInit(nn->weightsForLayer[0], nn->inputs->columns, nn->neuronsPerLayer[0], weightsName);

	//layer[i-1] x layer[i]
	for (CLUInt i = 1; i < nn->nHiddenLayers; ++i) {
		snprintf(weightsName, 31, "weightsForLayer[%d]", i);
		CLMatrixInit(nn->weightsForLayer[i], nn->neuronsPerLayer[i - 1], nn->neuronsPerLayer[i], weightsName);
	}

	snprintf(weightsName, 31, "weightsForLayer[%d]", nn->nHiddenLayers);
	CLMatrixInit(nn->weightsForLayer[nn->nHiddenLayers], nn->neuronsPerLayer[nn->nHiddenLayers - 1], nn->targets->columns, weightsName);

	CLMatrixInit(nn->outputs, nn->nPatterns, nn->nTargets, "outputs");

	nn->hActivations = malloc(sizeof(CLMatrix) * nn->nHiddenLayers);

	CLString hActivationName = malloc(sizeof(CLChar) * 32);
	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		snprintf(hActivationName, 31, "hActivation[%d]", i);
		nn->hActivations[i] = malloc(sizeof(CLMatrix));
		CLMatrixInit(nn->hActivations[i], nn->nPatterns, nn->neuronsPerLayer[i], hActivationName);
	}

	CLMatrixInit(nn->jacobian, nn->nPatterns * nn->nTargets, nWeights, "jacobian");
	CLMatrixInit(nn->hessian, nWeights, nWeights, "hessian");
	CLMatrixInit(nn->d, 1, nWeights, "d");
	CLMatrixInit(nn->delta, 1, nWeights, "delta");
	CLMatrixInit(nn->cholesky, nWeights, nWeights, "cholesky");

	CLMatrixPrintStats(nn->inputs);
	CLMatrixPrintStats(nn->targets);
	CLMatrixPrintStats(nn->weights);

	for (CLUInt i = 0; i < nn->nHiddenLayers + 1; ++i) {
		CLMatrixPrintStats(nn->weightsForLayer[i]);
	}

	CLMatrixPrintStats(nn->outputs);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixPrintStats(nn->hActivations[i]);
	}

	CLMatrixPrintStats(nn->jacobian);
	CLMatrixPrintStats(nn->hessian);
	CLMatrixPrintStats(nn->d);
	CLMatrixPrintStats(nn->delta);
	CLMatrixPrintStats(nn->cholesky);


	//TODO: Controllare se è giusta
//	CLSize totalSize = nn->inputs->size + nn->targets->size + nn->weights->size + nn->weightsTemp->size + nn->outputs->size + nn->jacobian->size + nn->hessian->size + nn->delta->size + nn->cholesky->size;
//	printf("Memory usage (aprox) : %0.2f MB\n", (float)totalSize / 1e6);
}

void CLAnnUpdateWithRandomWeights(CLAnn * nn)
{
	CLMatrixFillRandom(nn->weights);
}

void swapRow(CLMatrix * matrix, CLUInt aRow, CLUInt bRow)
{
	CLUInt cols = matrix->columns;
	for (CLUInt i = 0; i < matrix->columns; ++i) {
		CLFloat tmp = matrix->values[aRow * cols + i];
		matrix->values[aRow * cols + i] = matrix->values[bRow * cols + i];
		matrix->values[bRow * cols + i] = tmp;
	}
}

void CLAnnShufflePatterns(CLAnn * nn)
{
	CLUInt rows = nn->inputs->rows;
	if (rows > 1) {
		for (CLUInt i = rows - 1; i > 0; --i) {
			CLUInt j = CLRandomValue() * (i + 1);
			swapRow(nn->inputs, i, j);
			swapRow(nn->targets, i, j);
		}
	}
}

void CLAnnSetupTrainingFor(CLAnn * nn, CLPlatform platform, CLDevice device)
{
	nn->platform = platform;
	nn->device = device;
	nn->context = CLCreateContext(platform, device);
	nn->queue = CLCreateQueue(nn->context, nn->device);

	nn->program = CLCreateProgram(nn->context, nn->device, "Kernels.ocl");
	nn->kernelClean = CLCreateKernel(nn->program, kClean);

	nn->kernelActivation = malloc(sizeof(CLKernel) * 3);
	nn->kernelActivation[CLActivationSigmoid] = CLCreateKernel(nn->program, kActivationSigmoid);
	nn->kernelActivation[CLActivationTansig] = CLCreateKernel(nn->program, kActivationTansig);
	nn->kernelActivation[CLActivationRadbas] = CLCreateKernel(nn->program, kActivationRadbas);

	nn->kernelChiSquared = CLCreateKernel(nn->program, kChiSquared);
	nn->kernelChiSquaredReduce = CLCreateKernel(nn->program, kChiSquaredReduce);
	nn->kernelJacobian = CLCreateKernel(nn->program, kJacobian);
	nn->kernelDelta = CLCreateKernel(nn->program, kDelta);
	nn->kernelCholeskyDecomposition = CLCreateKernel(nn->program, kCholeskyDecomposition);

	clblasSetup();

	CLMatrixCreateMemHostVar(nn->inputs, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMemHostVar(nn->targets, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_WRITE);
	CLMatrixUpdateValues(nn->weightsTemp, nn->weights->values);
	CLMatrixCreateMemHostVar(nn->weightsTemp, nn->context, CL_MEM_READ_WRITE);

	//TODO: metterlo nell'init
	CLSize offset = 0;
	for (CLUInt i = 0; i < nn->nHiddenLayers + 1; ++i) {
		nn->weightsForLayer[i]->offsetMem = offset;
		offset += nn->weightsForLayer[i]->elements;
		nn->weightsForLayer[i]->mem = nn->weights->mem;
		clRetainMemObject(nn->weights->mem);
	}

	CLMatrixCreateMem(nn->outputs, nn->context, CL_MEM_READ_ONLY);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixCreateMem(nn->hActivations[i], nn->context, CL_MEM_READ_WRITE);
	}

	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->targets->elements, lws[0])};
	CLSize nwg = divUpSize(gws[0], lws[0]);

	//TODO: evitare di creare il mem ogni volta che chiamo questa funzione
	nn->chiSquaredError = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(CLFloat) * nwg, "chiSquaredError");

	CLMatrixCreateMem(nn->jacobian, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->hessian, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->d, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->delta, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->cholesky, nn->context, CL_MEM_READ_WRITE);
	nn->illMem = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(CLUInt), "ill");

	CLDeviceType deviceType = CLGetDeviceType(nn->device);

	switch (deviceType) {
		case CL_DEVICE_TYPE_CPU:
			BLOCK_SIZE = 1;
			break;
		case CL_DEVICE_TYPE_GPU:
			BLOCK_SIZE = 32;
			break;

		default:
			break;
	}
}

void CLAnnCleanMatrix(CLAnn * nn, CLMatrix * matrix)
{
	CLEvent event;
	CLUInt workDim = 1;
	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(matrix->elements, lws[0])};

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelClean, nArg++, sizeof(matrix->mem), &matrix->mem, matrix->name);
	CLSetKernelArg(nn->kernelClean, nArg++, sizeof(CLUInt), &matrix->elements, "elements");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelClean, workDim, 0, gws, lws, 0, NULL, &event, kClean);
	CLWaitForEvent(&event, kClean);
	CLReleaseEvent(event, kClean);
}

void CLAnnMatrixMultiply(CLAnn * nn, CLMatrix * matrixA, CLMatrix * matrixB, CLMatrix * matrixResult, CLStringConst nameEvent)
{
	CLSize m = matrixA->rows;
	CLSize n = matrixB->columns;
	CLSize k = matrixA->columns;

	CLEvent event;
	clblasStatus status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, 1,
									  matrixA->mem, matrixA->offsetMem, k,
									  matrixB->mem, matrixB->offsetMem, n,
									  0, matrixResult->mem, 0, n,
									  1, &nn->queue, 0, NULL, &event);

	if (status != CL_SUCCESS) {
		debugLog("SGEMM %s errorCode: %d", nameEvent, status);
		exit(status);
	}
	CLWaitForEvent(&event, nameEvent);
	//printf("MatrixMultiply(%s): %f ms\n", nameEvent, timeBetweenEventsMS(event, event));
	CLReleaseEvent(event, nameEvent);
}

void CLAnnActivation(CLAnn * nn, CLMatrix * matrix, CLActivation activationFunction, CLStringConst nameEvent)
{
	if (activationFunction == CLActivationLinear) return;

	CLEvent event;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(matrix->rows, lws[0]), CLGetOptimalGlobalWorkItemsSize(matrix->columns, lws[1])};

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelActivation[activationFunction], nArg++, sizeof(matrix->mem), &matrix->mem, matrix->name);
	CLSetKernelArg(nn->kernelActivation[activationFunction], nArg++, sizeof(CLUInt), &matrix->rows, "rows");
	CLSetKernelArg(nn->kernelActivation[activationFunction], nArg++, sizeof(CLUInt), &matrix->columns, "columns");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelActivation[activationFunction], 2, NULL, gws, lws, 0, NULL, &event, nameEvent);
	CLWaitForEvent(&event, nameEvent);
	//printf("Activation(%s): %f ms\n", nameEvent, timeBetweenEventsMS(event, event));
	CLReleaseEvent(event, "eventActivation");
}

void CLAnnForward(CLAnn * nn, CLUInt updateWeightsFromHost, CLUInt printOutputs)
{
	if (updateWeightsFromHost == CLTrue) {
		CLEvent eventCopyBuffer;
		clblasStatus status = clEnqueueCopyBuffer(nn->queue, nn->weightsTemp->mem, nn->weights->mem, 0, 0, nn->weightsTemp->size, 0, NULL, &eventCopyBuffer);
		CLErrorCheck(status, "clEnququeCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);
		CLWaitForEvent(&eventCopyBuffer, "eventCopyBuffer");
	}

	CLAnnMatrixMultiply(nn, nn->inputs, nn->weightsForLayer[0], nn->hActivations[0], "inputsXhWeightsForLayer[0]");
	CLAnnActivation(nn, nn->hActivations[0], nn->activationPerLayer[0], nn->hActivations[0]->name);

	//MULTILAYER
	for (CLUInt i = 1; i < nn->nHiddenLayers; ++i) {

		CLAnnMatrixMultiply(nn, nn->hActivations[i-1], nn->weightsForLayer[i], nn->hActivations[i], nn->hActivations[i]->name);
		CLAnnActivation(nn, nn->hActivations[i], nn->activationPerLayer[i], nn->hActivations[i]->name);
	}
	//END MULTILAYER

	CLAnnMatrixMultiply(nn, nn->hActivations[nn->nHiddenLayers - 1], nn->weightsForLayer[nn->nHiddenLayers], nn->outputs, nn->outputs->name);

#if DEBUG_FORWARD
	CLMatrixPrint(nn->inputs, CLMatrixNoTrans);
	CLMatrixPrint(nn->weights, CLMatrixNoTrans);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixUpdateValuesFromMem(nn->hActivations[i], nn->queue);
		CLMatrixPrint(nn->hActivations[i], CLMatrixNoTrans);
	}

	CLMatrixUpdateValuesFromMem(nn->outputs, nn->queue);
	CLMatrixPrint(nn->outputs, CLMatrixNoTrans);

	CLMatrixPrint(nn->targets, CLMatrixNoTrans);
#endif

	if (printOutputs && ! DEBUG_LOG) {
		CLAnnPrintResults(nn);
	}
}

CLFloat CLAnnChiSquared(CLAnn * nn)
{
	if (nn->outputs->mem == NULL) {
		fprintf(stderr, "Call CLAnnForward() before!");
		exit(-1);
	}

	//ChiSquared
	CLEvent eventChiSquared;
	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->targets->elements, lws[0])};
	CLSize nwg = divUpSize(gws[0], lws[0]);

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(nn->outputs->mem), &nn->outputs->mem, nn->outputs->name);
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(nn->targets->mem), &nn->targets->mem, nn->targets->name);
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(CLFloat) * BLOCK_SIZE, NULL, "localSums");
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(nn->chiSquaredError), &nn->chiSquaredError, "chiSquaredError");
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(CLUInt), &nn->targets->elements, "elements");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelChiSquared, 1, NULL, gws, lws, 0, NULL, &eventChiSquared, kChiSquared);
	CLWaitForEvent(&eventChiSquared, "eventChiSquared");

	//ChiSquaredReduce
	CLEvent eventChiSquaredReduce;
	gws[0] = nwg;
	lws[0] = nwg;
	nArg = 0;
	CLSetKernelArg(nn->kernelChiSquaredReduce, nArg++, sizeof(nn->chiSquaredError), &nn->chiSquaredError, "partialSums");
	CLSetKernelArg(nn->kernelChiSquaredReduce, nArg++, sizeof(CLFloat) * lws[0], NULL, "localSums");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelChiSquaredReduce, 1, NULL, gws, lws, 0, NULL, &eventChiSquaredReduce, kChiSquaredReduce);
	CLWaitForEvent(&eventChiSquaredReduce, "eventChiSquaredReduce");

	CLFloat error = *((CLFloat *)CLEnqueueReadBuffer(nn->queue, nn->chiSquaredError, sizeof(CLFloat), "chiSquaredError"));


#if DEBUG_CHI_SQUARED
	printf("ChiSquared: %f\n", error);
#endif


	//printf("ChiSquared: %f ms\n", timeBetweenEventsMS(eventChiSquared, eventChiSquaredReduce));
	CLReleaseEvent(eventChiSquared, "eventChiSquared");
	CLReleaseEvent(eventChiSquaredReduce, "eventChiSquaredReduce");
	return error;
}

void CLAnnJacobian(CLAnn * nn)
{
	CLUInt offset = 0;
	CLUInt slope = nn->neuronsPerLayer[0];
	CLUInt yTimes = nn->nTargets;
	CLUInt nArg = 0;

	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->inputs->columns, lws[0]), CLGetOptimalGlobalWorkItemsSize(nn->inputs->rows, lws[1])};



	if (nn->inputsCopiedIntoJacobian == CLFalse) {

		CLEvent eventJacobian;

		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->inputs->mem), &nn->inputs->mem, nn->inputs->name);
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->jacobian->columns, "jacobianColumns");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->rows, "rowsI");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->columns, "columnsI");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

		CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobian, "jacobian");
		CLWaitForEvent(&eventJacobian, "eventJacobian");
		CLReleaseEvent(eventJacobian, "eventJacobian");

		nn->inputsCopiedIntoJacobian = CLTrue;
	}

	offset = nn->nInputs * slope;

	CLEvent * eventJacobianX = malloc(sizeof(CLEvent) * nn->nHiddenLayers);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {

		slope = (i == nn->nHiddenLayers - 1 ? nn->nTargets : nn->neuronsPerLayer[i + 1]);

		nArg = 0;
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->hActivations[i]->mem), &nn->hActivations[i]->mem, nn->hActivations[i]->name);
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->jacobian->columns, "jacobianColumns");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->hActivations[i]->rows, "rowsI");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->hActivations[i]->columns, "columnsI");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
		CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

		gws[0] = CLGetOptimalGlobalWorkItemsSize(nn->hActivations[i]->columns, lws[0]);
		gws[1] = CLGetOptimalGlobalWorkItemsSize(nn->hActivations[i]->rows, lws[1]);

		CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, eventJacobianX+i, "jacobianX");
		CLWaitForEvent(eventJacobianX+i, "eventJacobianX");
		offset += nn->neuronsPerLayer[i] * slope;
	}

#if DEBUG_JACOBIAN
	debugLog("offset: %d\nslope: %d\n", offset, slope);
	CLMatrixUpdateValuesFromMem(nn->jacobian, nn->queue);
	CLMatrixPrint(nn->jacobian, CLMatrixNoTrans);
#endif

	//printf("Jacobian: %f ms\n", timeBetweenEventsMS(eventJacobianX[0], eventJacobianX[nn->nHiddenLayers-1]));

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLReleaseEvent(eventJacobianX[i], "eventJacobianX");
	}

}

//Hessian = J^T J
void CLAnnHessian(CLAnn * nn)
{
	CLEvent eventHessian;
	CLSize m = nn->jacobian->columns;
	CLSize n = nn->jacobian->columns;
	CLSize k = nn->jacobian->rows;

	clblasStatus status = clblasSgemm(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k, 1,
									  nn->jacobian->mem, 0, m,
									  nn->jacobian->mem, 0, n,
									  0, nn->hessian->mem, 0, n,
									  1, &nn->queue, 0, NULL, &eventHessian);

	if (status != CL_SUCCESS) {
		debugLog("hessian errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventHessian, "eventHessian");

	//printf("Hessian: %f ms\n", timeBetweenEventsMS(eventHessian, eventHessian));

	CLReleaseEvent(eventHessian, "eventHessian");

#if DEBUG_HESSIAN
	CLMatrixUpdateValuesFromMem(nn->hessian, nn->queue);
	CLMatrixPrint(nn->hessian, CLMatrixNoTrans);
#endif

	//Delta
	CLEvent eventDelta;
	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->d->mem), &nn->d->mem, nn->d->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->targets->mem), &nn->targets->mem, nn->targets->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->outputs->mem), &nn->outputs->mem, nn->outputs->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(CLUInt), &nn->targets->elements, "ny");
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(CLUInt), &nn->weights->elements, "npar");

	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->d->columns, BLOCK_SIZE)};

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelDelta, 1, NULL, gws, lws, 0, NULL, &eventDelta, kDelta);
	CLWaitForEvent(&eventDelta, "eventDelta");

	//printf("Delta: %f ms\n", timeBetweenEventsMS(eventDelta, eventDelta));
	CLReleaseEvent(eventDelta, "eventDelta");

#if DEBUG_DELTA
	CLMatrixUpdateValuesFromMem(nn->d, nn->queue);
	CLMatrixPrint(nn->d, CLMatrixNoTrans);
#endif
}

void CLAnnCholeskyDecomposition(CLAnn * nn, CLFloat mult)
{
	CLAnnCleanMatrix(nn, nn->cholesky);

	nn->ill = CLFalse;
	clblasStatus status = clEnqueueWriteBuffer(nn->queue, nn->illMem, CLTrue, 0, sizeof(CLInt), &nn->ill, 0, NULL, NULL);
	CLErrorCheck(status, "clEnqueueWriteBuffer", "copy ill", CHECK_EXIT);

	//CholeskyDecomposition
#if 0
	CLEvent eventCholeskyDecomposition;

	nArg = 0;
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->cholesky->mem), &nn->cholesky->mem, nn->cholesky->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->hessian->mem), &nn->hessian->mem, nn->hessian->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLUInt), &nn->weights->elements, "npar");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLFloat), &mult, "alpha");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->illMem), &nn->illMem, "ill");

	gws[0] = CLGetOptimalGlobalWorkItemsSize(nn->weights->elements, BLOCK_SIZE);
	lws[0] = BLOCK_SIZE;

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelCholeskyDecomposition, 1, NULL, gws, lws, 0, NULL, &eventCholeskyDecomposition, kCholeskyDecomposition);
	CLWaitForEvent(&eventCholeskyDecomposition, "eventCholeskyDecomposition");
	CLReleaseEvent(eventCholeskyDecomposition, "eventCholeskyDecomposition");

	nn->ill = ((CLUInt *)CLEnqueueReadBuffer(nn->queue, nn->illMem, sizeof(CLUInt), "ill"))[0];
#else

	CLSize gws[] = {nn->weights->elements};//CLGetOptimalGlobalWorkItemsSize(nn->weights->elements, BLOCK_SIZE);
	CLSize lws[] = {nn->weights->elements};//BLOCK_SIZE;

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->cholesky->mem), &nn->cholesky->mem, nn->cholesky->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->hessian->mem), &nn->hessian->mem, nn->hessian->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLUInt), &nn->weights->elements, "npar");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLFloat), &mult, "alpha");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->illMem), &nn->illMem, "ill");

	CLEvent * eventCholeskyDecomposition = malloc(sizeof(CLEvent) * nn->cholesky->rows);

	CLUInt lastKernelCall = 0;
	for (CLUInt i = 0; i < nn->cholesky->rows; ++i, ++lastKernelCall) {

		CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg, sizeof(CLUInt), &i, "row");

		CLEnqueueNDRangeKernel(nn->queue, nn->kernelCholeskyDecomposition, 1, NULL, gws, lws, 0, NULL, eventCholeskyDecomposition+i, kCholeskyDecomposition);
		CLWaitForEvent(eventCholeskyDecomposition+i, "eventCholeskyDecomposition");
	}
#endif

	nn->ill = ((CLBool *)CLEnqueueReadBuffer(nn->queue, nn->illMem, sizeof(CLBool), "ill"))[0];

#if DEBUG_CHOLESKY_DECOMPOSITION
	printf("Ill: %d\n", nn->ill);
	CLMatrixUpdateValuesFromMem(nn->cholesky, nn->queue);
	CLMatrixPrint(nn->cholesky, CLMatrixTrans);
#endif

	//printf("CholeskyDecomposition: %f ms\n", timeBetweenEventsMS(eventCholeskyDecomposition[0], eventCholeskyDecomposition[lastKernelCall - 1]));

	for (CLUInt i = 0; i < nn->cholesky->rows; ++i) {
		CLReleaseEvent(eventCholeskyDecomposition[i], "eventCholeskyDecomposition");
	}

}

void CLAnnSolveTriangular(CLAnn * nn, CLMatrix * cholesky, CLMatrix * delta, clblasTranspose uplo, CLStringConst eventName)
{
	CLEvent eventSolveTriangular;
	clblasStatus status = clblasStrsv(clblasRowMajor, clblasUpper, uplo, clblasNonUnit,
									  cholesky->rows, cholesky->mem, 0,
									  cholesky->rows, delta->mem, 0, 1,
									  1, &nn->queue, 0, NULL, &eventSolveTriangular);
	if (status != CL_SUCCESS) {
		debugLog("%s errorCode: %d", eventName, status);
		exit(status);
	}

	CLWaitForEvent(&eventSolveTriangular, eventName);
	//printf("SolveTriangular(%s): %f ms\n", eventName, timeBetweenEventsMS(eventSolveTriangular, eventSolveTriangular));
	CLReleaseEvent(eventSolveTriangular, eventName);
}

void CLAnnCholeskySolve(CLAnn * nn)
{
	CLEvent eventCopyDelta;
	clblasStatus status = clEnqueueCopyBuffer(nn->queue, nn->d->mem, nn->delta->mem, 0, 0, nn->d->size, 0, NULL, &eventCopyDelta);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "copyDelta", CHECK_EXIT);

#if DEBUG_CHOLESKY_SOLVE
	printf("\n\n### BEFORE ###\n\n");
	CLMatrixUpdateValuesFromMem(nn->d, nn->queue);
	CLMatrixPrint(nn->d, CLMatrixNoTrans);
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif

	CLAnnSolveTriangular(nn, nn->cholesky, nn->delta, clblasTrans, "choleskySolve[0]");
	CLAnnSolveTriangular(nn, nn->cholesky, nn->delta, clblasNoTrans, "choleskySolve[1]");

#if DEBUG_CHOLESKY_SOLVE
	printf("\n\n### AFTER ###\n\n");

	CLMatrixUpdateValuesFromMem(nn->d, nn->queue);
	CLMatrixPrint(nn->d, CLMatrixNoTrans);
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif
}

void CLAnnUpdateWeights(CLAnn * nn)
{
	if (nn->delta->mem == NULL) {
		fprintf(stderr, "Call CLAnnCholeskySolve() before!");
		return;
	}

	CLEvent eventCopyBuffer;
	clblasStatus status = clEnqueueCopyBuffer(nn->queue, nn->weightsTemp->mem, nn->weights->mem, 0, 0, nn->weightsTemp->size, 0, NULL, &eventCopyBuffer);
	CLErrorCheck(status, "clEnququeCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);
	CLWaitForEvent(&eventCopyBuffer, "eventCopyBuffer");
	CLReleaseEvent(eventCopyBuffer, "eventCopyBuffer");

	CLEvent eventUpdateWeights;
	status = clblasSaxpy(nn->weights->elements, 1, nn->delta->mem, 0, 1, nn->weights->mem, 0, 1, 1, &nn->queue, 0, NULL, &eventUpdateWeights);
	if (status != CL_SUCCESS) {
		debugLog("UpdateWeights errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventUpdateWeights, "eventUpdateWeights");
	CLReleaseEvent(eventUpdateWeights, "eventUpdateWeights");

#if DEBUG_UPDATE_WEIGHTS
	CLMatrixUpdateValuesFromMem(nn->weights, nn->queue);
	CLMatrixPrint(nn->weights, CLMatrixNoTrans);
#endif
}


void CLAnnUpdateLocalWeights(CLAnn * nn) {
	CLEvent eventCopyBuffer;
	clblasStatus status = clEnqueueCopyBuffer(nn->queue, nn->weights->mem, nn->weightsTemp->mem, 0, 0, nn->weightsTemp->size, 0, NULL, &eventCopyBuffer);
	CLErrorCheck(status, "clEnququeCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);
	CLWaitForEvent(&eventCopyBuffer, "eventCopyBuffer");
	CLReleaseEvent(eventCopyBuffer, "eventCopyBuffer");
}

CLUInt CLAnnTraining(CLAnn * nn) {

	CLUInt it;
	CLFloat mult;
	CLFloat lambda = nn->initialLambda;
	CLUInt ill;
	CLFloat error = -1.0f;
	CLFloat newError = -1.0f;
	CLFloat deltaError = -1.0f;


	CLAnnForward(nn, CLTrue, CLFalse);					//Forward per calcolare l'errore iniziale
	error = CLAnnChiSquared(nn);						//Calcolo dell'errore iniziale

	/* main iteration */
	for (it = 0; it < nn->maxIteration; ++it) {

		CLAnnForward(nn, CLFalse, CLFalse);				//Forward all'inizio dell'iterazione per ricalcolare le matrici Jacobian e Hessian
		CLAnnJacobian(nn);								//Calcolo matrice Jacobian
		CLAnnHessian(nn);								//Calcolo matrice Hessian

		mult = 1 + lambda;								//Aggiornamento moltiplicatore
		ill = 1;										//ill settato a 1 per entrare almeno una volta nel while
														//Non si può sostituire con il do/while per via della seconda condizione
		while (ill && (it < nn->maxIteration)) {

			CLAnnCholeskyDecomposition(nn, mult);		//Calcolo della decomposizione di Cholesky
			ill = nn->ill;								//Aggiornamento di ill

			if (!ill) {
				CLAnnCholeskySolve(nn);					//Risoluzione di Cholesky per il calcolo dei delta dei pesi
				CLAnnUpdateWeights(nn);					//Aggiornamento dei pesi con i delta calcolati nello step precedente

				CLAnnForward(nn, CLFalse, CLFalse);		//Forward per ricalcolare l'errore
				newError = CLAnnChiSquared(nn);			//Calcolo del nuovo errore

				deltaError = newError - error;			//Calcolo del delta error
				ill = (deltaError > 0);					//Aggiornamento di ill a 0 se il delta error è negativo
			}

			if (nn->verbose == CLTrue) printf("it = %4d,   lambda = %10g,   err = %10g,   derr = %10g\n", it, lambda, error, deltaError);

			if (isnan(newError) || lambda > 1e9){
				return (it == nn->maxIteration);
			}
			
			if (ill) {									//Se ill è ancora 1, vengono aggiornati i moltiplicatori
				mult = (1 + lambda * nn->upFactor)/(1 + lambda);
				lambda *= nn->upFactor;
				it++;
			}
		}
		CLAnnUpdateLocalWeights(nn);					//I nuovi pesi vengono salvati

		error = newError;
		lambda /= nn->downFactor;

		if ((!ill) && (-deltaError < nn->targetDeltaError))
			break;
	}

	nn->finalError = error;
	nn->finalDeltaError = deltaError;

	return (it == nn->maxIteration);
}

void CLAnnPrintResults(CLAnn * nn)
{
	CLMatrixUpdateValuesFromMem(nn->outputs, nn->queue);

	for (CLUInt i = 0; i < nn->nInputs; ++i) {
		printf("Inputs[%2d]|", i);
	}
	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf("Target[%2d]|", i);
	}
	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf("Output[%2d]|", i);
	}
	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf(" Error%%[%2d] |", i);
	}
	printf("\n");

	CLFloat * errorPerc = malloc(sizeof(CLFloat) * nn->nTargets);

	for (CLUInt p = 0; p < nn->nPatterns; ++p) {

		for (CLUInt i = 0; i < nn->nInputs; ++i) {
			printf("%+10g|", nn->inputs->values[p * nn->nInputs + i]);
		}

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			errorPerc[o] = nn->targets->values[p * nn->nTargets + o];
			printf("%+10g|", errorPerc[o]);
		}

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			CLFloat value = nn->outputs->values[p * nn->nTargets + o];
			errorPerc[o] -= value;
			printf("%+10g|", value);
		}

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			CLFloat errorPercValue = fabs(errorPerc[o]) * 100;
			printf( (errorPercValue > 30.0f ? "%10g♥️|" : "%10g💚|"), errorPercValue);
		}
		printf("\n");
	}
}

void CLAnnRelease(CLAnn * nn)
{
	CLMatrixRelease(nn->cholesky);
	CLMatrixRelease(nn->d);
	CLMatrixRelease(nn->delta);
	CLMatrixRelease(nn->hessian);
	CLMatrixRelease(nn->jacobian);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixRelease(nn->hActivations[i]);
		CLMatrixRelease(nn->weightsForLayer[i]);
	}
	CLMatrixRelease(nn->weightsForLayer[nn->nHiddenLayers]);

	CLMatrixRelease(nn->outputs);
	CLMatrixRelease(nn->weightsTemp);
	CLMatrixRelease(nn->weights);
	CLMatrixRelease(nn->targets);
	CLMatrixRelease(nn->inputs);
	CLReleaseMemObject(nn->chiSquaredError, "chiSquaredError");


	nn->cholesky = NULL;
	nn->d = NULL;
	nn->delta = NULL;
	nn->hessian = NULL;
	nn->jacobian = NULL;
	nn->hActivations = NULL;
	nn->outputs = NULL;
	nn->weights = NULL;
	nn->weightsTemp = NULL;
	nn->weightsForLayer = NULL;
	nn->targets = NULL;
	nn->inputs = NULL;

	CLReleaseKernel(nn->kernelActivation[CLActivationSigmoid], kActivationSigmoid);
	CLReleaseKernel(nn->kernelActivation[CLActivationTansig], kActivationTansig);
	CLReleaseKernel(nn->kernelActivation[CLActivationRadbas], kActivationRadbas);


	CLReleaseKernel(nn->kernelChiSquared, kChiSquared);
	CLReleaseKernel(nn->kernelChiSquaredReduce, kChiSquaredReduce);
	CLReleaseKernel(nn->kernelJacobian, kJacobian);
	CLReleaseKernel(nn->kernelDelta, kDelta);
	CLReleaseKernel(nn->kernelCholeskyDecomposition, kCholeskyDecomposition);

	CLReleaseProgram(nn->program, "program");
	CLReleaseQueue(nn->queue, "queue");
	CLReleaseContext(nn->context, "context");
	CLReleaseDevice(nn->device, "device");

	nn->program = NULL;
	nn->queue = NULL;
	nn->context = NULL;
	nn->device = NULL;

	free(nn);
	nn = NULL;

	clblasTeardown();
}