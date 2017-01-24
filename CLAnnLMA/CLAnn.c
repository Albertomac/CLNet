//
//  CLAnn.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/6/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLAnn.h"
#include <math.h>
#include </usr/local/include/clBLAS.h>
#include "CLBenchmark.h"

#define kClean "clean"
#define kActivation "activation"
#define kChiSquared "chiSquared"
#define kChiSquaredReduce "chiSquaredReduce"
#define kJacobian "jacobian"
#define kDelta "delta"
#define kCholeskyDecomposition "choleskyDecomposition"

CLUInt BLOCK_SIZE = 32;

void CLAnnInit(CLAnn * nn, CLFloat learningRate, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddenLayers, CLUInt nNeuronsPerLayer, CLUInt nTargets, CLStringConst name)
{
	nn->name = malloc(sizeof(CLChar) * 1024);
	strcpy(nn->name, name);
	nn->nPatterns = nPatterns;
	nn->nInputs = nInputs;
	nn->nHiddenLayers = nHiddenLayers;
	nn->nNeuronsPerLayer = nNeuronsPerLayer;
	nn->nTargets = nTargets;

	printf("/** %s **/\n"
		   "nPatters: %d\n"
		   "%d x (%d)^(%d) x %d\n"
		   "/** %s **/\n", nn->name, nn->nPatterns, nn->nInputs, nn->nNeuronsPerLayer, nn->nHiddenLayers, nn->nTargets, nn->name);

	nn->inputs = malloc(sizeof(CLMatrix));
	nn->targets = malloc(sizeof(CLMatrix));

	nn->weights = malloc(sizeof(CLMatrix));
	nn->weightsTemp = malloc(sizeof(CLMatrix));
	nn->outputs = malloc(sizeof(CLMatrix));

	nn->hActivations = malloc(sizeof(CLMatrix));
	nn->jacobian = malloc(sizeof(CLMatrix));
	nn->hessian = malloc(sizeof(CLMatrix));
	nn->delta = malloc(sizeof(CLMatrix));
	nn->cholesky = malloc(sizeof(CLMatrix));

	nn->learningRate = learningRate;

	nn->verbose = CLTrue;
	nn->maxIteration = 10000;
	nn->initialLambda = 0.0001;
	nn->upFactor = 10.0f;
	nn->downFactor = 10.0f;
	nn->targetDeltaError = 1e-12f;
	nn->finalError = 0.0f;
	nn->finalDeltaError = 0.0f;

	CLUInt nWeights = nn->nInputs * nn->nNeuronsPerLayer + (nn->nHiddenLayers - 1) * nn->nNeuronsPerLayer * nn->nNeuronsPerLayer + nn->nNeuronsPerLayer * nn->nTargets;

	CLMatrixInit(nn->inputs, nn->nPatterns, nn->nInputs, "inputs");
	CLMatrixInit(nn->targets, nn->nPatterns, nn->nTargets, "targets");

	CLMatrixInit(nn->weights, 1, nWeights, "weights");
	CLMatrixInit(nn->weightsTemp, 1, nWeights, "weightsTemp");
	CLMatrixInit(nn->outputs, nn->nPatterns, nn->nTargets, "outputs");

	nn->hActivations = malloc(sizeof(CLMatrix) * nn->nHiddenLayers);

	CLString layerName = malloc(sizeof(CLChar) * 32);
	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		snprintf(layerName, 31, "hActivation[%d]", i);
		nn->hActivations[i] = malloc(sizeof(CLMatrix));
		CLMatrixInit(nn->hActivations[i], nn->nPatterns, nn->nNeuronsPerLayer, layerName);
	}

	CLMatrixInit(nn->jacobian, nn->nPatterns * nn->nTargets, nWeights, "jacobian");
	CLMatrixInit(nn->hessian, nWeights, nWeights, "hessian");
	CLMatrixInit(nn->delta, 1, nWeights, "delta");
	CLMatrixInit(nn->cholesky, nWeights, nWeights, "cholesky");

	CLMatrixPrintStats(nn->inputs);
	CLMatrixPrintStats(nn->targets);
	CLMatrixPrintStats(nn->weights);
	CLMatrixPrintStats(nn->weightsTemp);
	CLMatrixPrintStats(nn->outputs);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixPrintStats(nn->hActivations[i]);
	}

	CLMatrixPrintStats(nn->jacobian);
	CLMatrixPrintStats(nn->hessian);
	CLMatrixPrintStats(nn->delta);
	CLMatrixPrintStats(nn->cholesky);


	//TODO: Controllare se è giusta
	CLSize totalSize = nn->inputs->size + nn->targets->size + nn->weights->size + nn->weightsTemp->size + nn->outputs->size + nn->jacobian->size + nn->hessian->size + nn->delta->size + nn->cholesky->size;
	printf("Memory usage (aprox) : %0.2f MB\n", (float)totalSize / 1e6);
}

void CLAnnUpdateWithRandomWeights(CLAnn * nn)
{
	CLMatrixFillRandom(nn->weights);
}

void CLAnnSetupTrainingFor(CLAnn * nn, CLPlatform platform, CLDevice device, int activationFunction)
{
	nn->platform = platform;
	nn->device = device;
	nn->context = CLCreateContext(platform, device);
	nn->queue = CLCreateQueue(nn->context, nn->device);
//	nn->program = CLCreateProgram(nn->context, nn->device, "Kernels.ocl");

	switch (activationFunction) {
		case ACTIVATION_LINEAR:
			//#define activationFunction(e) e
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) e");
			break;
		case ACTIVATION_SIGMOID:
			//#define activationFunction(e) (1.0f / (1.0f + exp(-e)))
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) (1.0f / (1.0f + exp(-e)))");
			break;
		case ACTIVATION_TANSIG:
			//#define activationFunction(e) (2.0f/(1.0f + exp(-2.0f * e))-1.0f)
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) (2.0f/(1.0f + exp(-2.0f * e))-1.0f)");
			break;

		default:
			break;
	}

	nn->kernelClean = CLCreateKernel(nn->program, kClean);
	nn->kernelActivation = CLCreateKernel(nn->program, kActivation);
	nn->kernelChiSquared = CLCreateKernel(nn->program, kChiSquared);
	nn->kernelChiSquaredReduce = CLCreateKernel(nn->program, kChiSquaredReduce);
	nn->kernelJacobian = CLCreateKernel(nn->program, kJacobian);
	nn->kernelDelta = CLCreateKernel(nn->program, kDelta);
	nn->kernelCholeskyDecomposition = CLCreateKernel(nn->program, kCholeskyDecomposition);

	clblasSetup();

	CLMatrixCreateMemHostVar(nn->inputs, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMemHostVar(nn->targets, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->weightsTemp, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMem(nn->outputs, nn->context, CL_MEM_READ_ONLY);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixCreateMem(nn->hActivations[i], nn->context, CL_MEM_READ_WRITE);
	}

	CLMatrixCreateMem(nn->jacobian, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->hessian, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->delta, nn->context, CL_MEM_READ_WRITE);
	CLMatrixCreateMem(nn->cholesky, nn->context, CL_MEM_READ_WRITE);

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
	CLAnnCleanMatrix(nn, matrixResult);

	const CLFloat alphaBeta = 1.0f;
	CLSize m = matrixA->rows;
	CLSize n = matrixB->columns;
	CLSize k = matrixA->columns;

	CLEvent event;
	clblasStatus status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alphaBeta,
									  matrixA->mem, 0, k,
									  matrixB->mem, 0, n, alphaBeta,
									  matrixResult->mem, 0, n, 1,
									  &nn->queue, 0, NULL, &event);

	CLErrorCheck(status, "clBlasSgemm", nameEvent, CHECK_EXIT);

	CLWaitForEvent(&event, nameEvent);
	CLReleaseEvent(event, nameEvent);
}

void CLAnnActivation(CLAnn * nn, CLMatrix * matrix, CLStringConst nameEvent)
{
	CLEvent event;
	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(matrix->rows, lws[0]), CLGetOptimalGlobalWorkItemsSize(matrix->columns, lws[1])};

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(matrix->mem), &matrix->mem, matrix->name);
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(CLUInt), &matrix->rows, "rows");
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(CLUInt), &matrix->columns, "columns");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelActivation, workDim, NULL, gws, lws, 0, NULL, &event, nameEvent);
	CLWaitForEvent(&event, nameEvent);
}

void CLAnnForward(CLAnn * nn, CLUInt updateWeightsFromHost, CLUInt printOutputs)
{
	if (updateWeightsFromHost == CLTrue) {
		CLMatrixReleaseMem(nn->weights);
		CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_WRITE);
	}

	CLMatrix * hWeights = malloc(sizeof(CLMatrix));
	CLMatrixInit(hWeights, nn->nInputs, nn->nNeuronsPerLayer, "hWeights");
	hWeights->mem = CLCreateSubBuffer(nn->weights->mem, CL_MEM_READ_ONLY, 0, hWeights->size, hWeights->name);

	CLAnnMatrixMultiply(nn, nn->inputs, hWeights, nn->hActivations[0], "inputsXhWeights[0]");
	CLAnnActivation(nn, nn->hActivations[0], nn->hActivations[0]->name);

	//MULTILAYER

	CLSize offset = hWeights->size;
	hWeights->rows = nn->nNeuronsPerLayer;
	hWeights->columns = nn->nNeuronsPerLayer;
	hWeights->elements = hWeights->rows * hWeights->columns;
	hWeights->size = sizeof(CLFloat) * hWeights->elements;

	for (CLUInt i = 1; i < nn->nHiddenLayers; ++i) {

		snprintf(hWeights->name, 32, "hWeights[%d]", i);
		hWeights->mem = CLCreateSubBuffer(nn->weights->mem, CL_MEM_READ_ONLY, offset, hWeights->size, hWeights->name);

		CLAnnMatrixMultiply(nn, nn->hActivations[i-1], hWeights, nn->hActivations[i], nn->hActivations[i]->name);
		CLAnnActivation(nn, nn->hActivations[i], nn->hActivations[i]->name);
		offset += hWeights->size;
	}

	//END MULTILAYER

	//Outputs = hiddenActivations x outputsWeights
	CLMatrix * oWeights = malloc(sizeof(CLMatrix));
	CLMatrixInit(oWeights, nn->nNeuronsPerLayer, nn->nTargets, "oWeights");
	oWeights->mem = CLCreateSubBuffer(nn->weights->mem, CL_MEM_READ_ONLY, offset, oWeights->size, oWeights->name);

	CLAnnMatrixMultiply(nn, nn->hActivations[nn->nHiddenLayers - 1], oWeights, nn->outputs, nn->outputs->name);


#if DEBUG_LOG
	CLMatrixPrint(nn->inputs, CLMatrixNoTrans);
//	CLMatrixPrint(nn->weights, CLMatrixNoTrans);

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

	//Releases
	CLMatrixRelease(hWeights);
	CLMatrixRelease(oWeights);
}

CLFloat CLAnnChiSquared(CLAnn * nn)
{
	if (nn->outputs->mem == NULL) {
		fprintf(stderr, "Call CLAnnForward() before!");
		exit(-1);
	}

	//ChiSquared
	CLEvent eventChiSquared;

	CLUInt workDim = 1;
	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->targets->elements, lws[0])};
	CLSize nwg = divUpSize(gws[0], lws[0]);

	CLMem chiSquaredError = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(CLFloat) * nwg, "chiSquaredError");

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(nn->outputs->mem), &nn->outputs->mem, nn->outputs->name);
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(nn->targets->mem), &nn->targets->mem, nn->targets->name);
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(CLFloat) * BLOCK_SIZE, NULL, "localSums");
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(chiSquaredError), &chiSquaredError, "chiSquaredError");
	CLSetKernelArg(nn->kernelChiSquared, nArg++, sizeof(CLUInt), &nn->targets->elements, "elements");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelChiSquared, workDim, NULL, gws, lws, 0, NULL, &eventChiSquared, kChiSquared);
	CLWaitForEvent(&eventChiSquared, "eventChiSquared");

#if BENCHMARK
	CLSize loads = nn->outputs->elements + nn->targets->elements;
	CLSize stores = nwg;
	CLSize elements = loads + stores;
	CLSize dataSize = nn->outputs->size + nn->targets->size + nwg * sizeof(CLFloat);
	CLSize operations = 3 * elements;
	CLBenchmarkLog(eventChiSquared, eventChiSquared, loads, stores, elements, dataSize, operations, kChiSquared);
#endif

	//ChiSquaredReduce
	CLEvent eventChiSquaredReduce;
	gws[0] = nwg;
	lws[0] = nwg;
	nArg = 0;
	CLSetKernelArg(nn->kernelChiSquaredReduce, nArg++, sizeof(chiSquaredError), &chiSquaredError, "partialSums");
	CLSetKernelArg(nn->kernelChiSquaredReduce, nArg++, sizeof(CLFloat) * lws[0], NULL, "localSums");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelChiSquaredReduce, workDim, NULL, gws, lws, 1, &eventChiSquared, &eventChiSquaredReduce, kChiSquaredReduce);
	CLWaitForEvent(&eventChiSquaredReduce, "eventChiSquaredReduce");

#if BENCHMARK
	loads = nwg;
	stores = 1;
	elements = nwg;
	dataSize = sizeof(CLFloat) * (nwg + 1);
	operations = 3 * elements;
	CLBenchmarkLog(eventChiSquaredReduce, eventChiSquaredReduce, loads, stores, elements, dataSize, operations, kChiSquaredReduce);
#endif

	CLFloat error = *((CLFloat *)CLEnqueueReadBuffer(nn->queue, chiSquaredError, sizeof(CLFloat), "chiSquaredError"));
	debugLog("ChiSquared: %f\n", error);

	//Releases
	CLReleaseEvent(eventChiSquared, "eventChiSquared");
	CLReleaseEvent(eventChiSquaredReduce, "eventChiSquaredReduce");
	CLReleaseMemObject(chiSquaredError, "chiSquaredError");

	return error;
}

void CLAnnJacobian(CLAnn * nn)
{
	CLAnnCleanMatrix(nn, nn->jacobian);

	CLUInt offset = 0;
	CLUInt slope = nn->nNeuronsPerLayer;
	CLUInt yTimes = nn->nTargets;

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->inputs->mem), &nn->inputs->mem, nn->inputs->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->jacobian->columns, "jacobianColumns");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->rows, "rowsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->columns, "columnsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

	CLEvent eventJacobian;
	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->inputs->columns, lws[0]), CLGetOptimalGlobalWorkItemsSize(nn->inputs->rows, lws[1])};

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobian, "jacobian");
	CLWaitForEvent(&eventJacobian, "eventJacobian");

#if BENCHMARK
	CLSize loads = nn->inputs->elements;
	CLSize stores = nn->nNeuronsPerLayer * nn->jacobian->rows;
	CLSize elements = nn->inputs->elements + (nn->nNeuronsPerLayer * nn->jacobian->rows);
	CLSize dataSize = sizeof(CLFloat) * (loads + stores);
	CLSize operations = 1;
	CLBenchmarkLog(eventJacobian, eventJacobian, loads, stores, elements, dataSize, operations, "jacobian");
#endif

	offset = nn->nInputs * nn->nNeuronsPerLayer;
	slope = (nn->nHiddenLayers == 1) ? 1 : (nn->nHiddenLayers - 1) * nn->nNeuronsPerLayer;

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLEvent eventJacobianX;
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

		CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobianX, "jacobianX");
		CLWaitForEvent(&eventJacobianX, "eventJacobianX");
		offset += nn->nNeuronsPerLayer * slope;
	}
#if BENCHMARK
	loads = nn->hActivations->elements;
	stores = nn->nTargets * nn->jacobian->rows;
	elements = nn->hActivations->elements + (nn->nTargets * nn->jacobian->rows);
	dataSize = sizeof(CLFloat) * (loads + stores);
	operations = 1;
	CLBenchmarkLog(eventJacobian[1], eventJacobian[1], loads, stores, elements, dataSize, operations, "jacobian[1]");
#endif

#if DEBUG_LOG
	debugLog("offset: %d\nslope: %d\n", offset, slope);
	CLMatrixUpdateValuesFromMem(nn->jacobian, nn->queue);
	CLMatrixPrint(nn->jacobian, CLMatrixNoTrans);
#endif

	//Releases
	CLReleaseEvent(eventJacobian, "eventJacobian");
}

//Hessian = J^T J
void CLAnnHessian(CLAnn * nn)
{

	CLAnnCleanMatrix(nn, nn->hessian);

	CLEvent eventHessian;
	CLSize m = nn->jacobian->columns;
	CLSize n = nn->jacobian->columns;
	CLSize k = nn->jacobian->rows;
	CLFloat alphaBeta = 1.0f;

	clblasStatus status = clblasSgemm(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k, alphaBeta,
									  nn->jacobian->mem, 0, m,
									  nn->jacobian->mem, 0, n, alphaBeta,
									  nn->hessian->mem, 0, n,
									  1, &nn->queue, 0, NULL, &eventHessian);

	if (status != CL_SUCCESS) {
		debugLog("hessian errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventHessian, "eventHessian");

#if BENCHMARK
	CLSize loads = nn->jacobian->size * 2;
	CLSize stores = nn->hessian->size;
	CLSize elements = loads + stores;
	CLSize dataSize = 2 * nn->jacobian->size + nn->hessian->size;
	CLSize operations = 2 * n * m * n;
	CLBenchmarkLog(eventHessian, eventHessian, loads, stores, elements, dataSize, operations, "hessian");
#endif

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->hessian, nn->queue);
	CLMatrixPrint(nn->hessian, CLMatrixNoTrans);
#endif

	//Releases
	CLReleaseEvent(eventHessian, "eventHessian");}

void CLAnnCholeskyDecomposition(CLAnn * nn, CLFloat mult)
{
	if (nn->outputs->mem == NULL) {
		fprintf(stderr, "Call CLAnnForward() before!");
		return;
	}
	if (nn->jacobian->mem == NULL) {
		fprintf(stderr, "Call CLAnnJacobian() before!");
		return;
	}
	if (nn->hessian->mem == NULL) {
		fprintf(stderr, "Call CLAnnHessian() before!");
		return;
	}

	if (nn->delta->mem == NULL)
		CLMatrixCreateMemHostVar(nn->delta, nn->context, CL_MEM_READ_WRITE);

//	CLMatrixReleaseMem(nn->cholesky);
//	CLMatrixCreateMemHostVar(nn->cholesky, nn->context, CL_MEM_READ_WRITE);
	CLAnnCleanMatrix(nn, nn->cholesky);

	//Delta
	CLEvent eventDelta;
	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->delta->mem), &nn->delta->mem, nn->delta->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->targets->mem), &nn->targets->mem, nn->targets->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->outputs->mem), &nn->outputs->mem, nn->outputs->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(CLUInt), &nn->targets->elements, "ny");
	CLSetKernelArg(nn->kernelDelta, nArg++, sizeof(CLUInt), &nn->weights->elements, "npar");

	CLUInt workDim = 1;
	CLSize lws[] = {BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->delta->columns, BLOCK_SIZE)};

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelDelta, workDim, NULL, gws, lws, 0, NULL, &eventDelta, kDelta);
	CLWaitForEvent(&eventDelta, "eventDelta");

#if BENCHMARK
	CLSize loads = nn->targets->elements + nn->outputs->elements + nn->jacobian->elements;
	CLSize stores = nn->delta->elements;
	CLSize elements = nn->targets->elements + nn->outputs->elements + nn->jacobian->elements + nn->delta->elements;
	CLSize dataSize = nn->targets->size + nn->outputs->size + nn->jacobian->size + nn->delta->size;
	CLSize operations = 3 * nn->jacobian->elements;
	CLBenchmarkLog(eventDelta, eventDelta, loads, stores, elements, dataSize, operations, "delta");
#endif
	
#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif

	//CholeskyDecomposition
	CLMem ill = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(CLUInt), "ill");

	CLEvent eventCholeskyDecomposition;

	nArg = 0;
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->cholesky->mem), &nn->cholesky->mem, nn->cholesky->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(nn->hessian->mem), &nn->hessian->mem, nn->hessian->name);
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLUInt), &nn->weights->elements, "npar");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(CLFloat), &mult, "alpha");
	CLSetKernelArg(nn->kernelCholeskyDecomposition, nArg++, sizeof(ill), &ill, "ill");

	workDim = 1;
	gws[0] = CLGetOptimalGlobalWorkItemsSize(nn->weights->elements, BLOCK_SIZE);
	lws[0] = BLOCK_SIZE;

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelCholeskyDecomposition, workDim, NULL, gws, lws, 0, NULL, &eventCholeskyDecomposition, kCholeskyDecomposition);
	CLWaitForEvent(&eventCholeskyDecomposition, "eventCholeskyDecomposition");

#if BENCHMARK
	//TODO: NON SO CHE PESCI PIJARE!
//	loads = nn->hessian->elements;
//	stores = nn->cholesky->elements;
//	elements = ;
//	dataSize = ;
//	operations = ;
//	CLBenchmarkLog(eventCholeskyDecomposition, eventCholeskyDecomposition, loads, stores, elements, dataSize, operations, kCholeskyDecomposition);
#endif

	nn->ill = *((CLUInt *)CLEnqueueReadBuffer(nn->queue, ill, sizeof(CLUInt), "ill"));

#if DEBUG_LOG
	printf("Ill: %d\n", nn->ill);
	CLMatrixUpdateValuesFromMem(nn->cholesky, nn->queue);
	CLMatrixPrint(nn->cholesky, CLMatrixNoTrans);
#endif

	//Releases
	CLReleaseEvent(eventDelta, "eventDelta");
	CLReleaseEvent(eventCholeskyDecomposition, "eventCholeskyDecomposition");
	CLReleaseMemObject(ill, "ill");
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

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif

	CLWaitForEvent(&eventSolveTriangular, eventName);
	CLReleaseEvent(eventSolveTriangular, eventName);
}

void CLAnnCholeskySolve(CLAnn * nn)
{
	CLAnnSolveTriangular(nn, nn->cholesky, nn->delta, clblasTrans, "choleskySolve[0]");
	CLAnnSolveTriangular(nn, nn->cholesky, nn->delta, clblasNoTrans, "choleskySolve[1]");

//	clblasStatus status;
//	CLEvent eventCholeskySolve[2];
//	status = clblasStrsv(clblasRowMajor, clblasUpper, clblasTrans, clblasNonUnit,
//						 nn->cholesky->rows, nn->cholesky->mem, 0,
//						 nn->cholesky->rows, nn->delta->mem, 0, 1,
//						 1, &nn->queue, 0, NULL, &eventCholeskySolve[0]);
//	if (status != CL_SUCCESS) {
//		debugLog("CholeskySolve[0] errorCode: %d", status);
//		exit(status);
//	}
//	CLWaitForEvent(&eventCholeskySolve[0], "eventCholeskySolve[0]");
//
//	status = clblasStrsv(clblasRowMajor, clblasUpper, clblasNoTrans, clblasNonUnit,
//						 nn->cholesky->rows, nn->cholesky->mem, 0,
//						 nn->cholesky->rows, nn->delta->mem, 0, 1,
//						 1, &nn->queue, 0, NULL, &eventCholeskySolve[1]);
//	if (status != CL_SUCCESS) {
//		debugLog("CholeskySolve[1] errorCode: %d", status);
//		exit(status);
//	}
//	CLWaitForEvent(&eventCholeskySolve[1], "eventCholeskySolve[1]");

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif

	//Releases
//	CLReleaseEvent(eventCholeskySolve[0], "eventCholeskySolve[0]");
//	CLReleaseEvent(eventCholeskySolve[1], "eventCholeskySolve[1]");
}

//void CLAnnUpdateWeightsTemp(CLAnn * nn)
//{
//	if (nn->delta->mem == NULL) {
//		fprintf(stderr, "Call CLAnnCholeskySolve() before!");
//		return;
//	}
//
//	clblasStatus status;
//	CLEvent eventUpdateWeightsTemp;
//	status = clblasSaxpy(nn->weightsTemp->elements, 1.0f, nn->delta->mem, 0, 1, nn->weightsTemp->mem, 0, 1, 1, &nn->queue, 0, NULL, &eventUpdateWeightsTemp);
//	if (status != CL_SUCCESS) {
//		debugLog("UpdateWeightsTemp errorCode: %d", status);
//		exit(status);
//	}
//	CLWaitForEvent(&eventUpdateWeightsTemp, "eventUpdateWeightsTemp");
//
//#if BENCHMARK
//	CLSize loads = nn->weightsTemp->elements + nn->delta->elements;
//	CLSize stores = nn->weightsTemp->elements;
//	CLSize elements = nn->weightsTemp->elements + nn->delta->elements;
//	CLSize dataSize = nn->weightsTemp->size + nn->delta->size;
//	CLSize operations = nn->weightsTemp->elements;
//	CLBenchmarkLog(eventUpdateWeights, eventUpdateWeights, loads, stores, elements, dataSize, operations, "updateWeightsTemp");
//#endif
//
//#if DEBUG_LOG
//	CLMatrixUpdateValuesFromMem(nn->weightsTemp, nn->queue);
//	CLMatrixPrint(nn->weightsTemp, CLMatrixNoTrans);
//#endif
//	//Releases
//	CLReleaseEvent(eventUpdateWeightsTemp, "eventUpdateWeightsTemp");
//}


void CLAnnUpdateWeights(CLAnn * nn)
{
	if (nn->delta->mem == NULL) {
		fprintf(stderr, "Call CLAnnCholeskySolve() before!");
		return;
	}

	CLMatrixReleaseMem(nn->weights);
	CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_WRITE);

	clblasStatus status;
	CLEvent eventUpdateWeights;
	status = clblasSaxpy(nn->weights->elements, nn->learningRate, nn->delta->mem, 0, 1, nn->weights->mem, 0, 1, 1, &nn->queue, 0, NULL, &eventUpdateWeights);
	if (status != CL_SUCCESS) {
		debugLog("UpdateWeights errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventUpdateWeights, "eventUpdateWeights");

#if BENCHMARK
	CLSize loads = nn->weights->elements + nn->delta->elements;
	CLSize stores = nn->weights->elements;
	CLSize elements = nn->weights->elements + nn->delta->elements;
	CLSize dataSize = nn->weights->size + nn->delta->size;
	CLSize operations = nn->weights->elements;
	CLBenchmarkLog(eventUpdateWeights, eventUpdateWeights, loads, stores, elements, dataSize, operations, "updateWeights");
#endif

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->weights, nn->queue);
	CLMatrixPrint(nn->weights, CLMatrixNoTrans);
#endif
	//Releases
	CLReleaseEvent(eventUpdateWeights, "eventUpdateWeights");
}

//void CLAnnUpdateLocalWeights(CLAnn * nn) {
//
//	clblasStatus status;
//	CLEvent eventConfirmWeights;
//	status = clblasScopy(nn->weightsTemp->elements, nn->weightsTemp->mem, 0, 1.0f, nn->weights->mem, 0, 1, 1, &nn->queue, 0, NULL, &eventConfirmWeights);
//	if (status != CL_SUCCESS) {
//		debugLog("confirmWeights errorCode: %d", status);
//		exit(status);
//	}
//	CLWaitForEvent(&eventConfirmWeights, "eventConfirmWeights");
//}

void CLAnnUpdateLocalWeights(CLAnn * nn) {
	CLMatrixUpdateValuesFromMem(nn->weights, nn->queue);
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

	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf(" Target[%2d] |", i);
	}
	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf(" Output[%2d] |", i);
	}
	for (CLUInt i = 0; i < nn->nTargets; ++i) {
		printf(" Error%%[%2d] |", i);
	}
	printf("\n");

	CLFloat * errorPerc = malloc(sizeof(CLFloat) * nn->nTargets);

	for (CLUInt p = 0; p < nn->nPatterns; ++p) {

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			errorPerc[o] = nn->targets->values[p * nn->nTargets + o];
			printf("%12g|", errorPerc[o]);
		}

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			CLFloat value = nn->outputs->values[p * nn->nTargets + o];
			errorPerc[o] -= value;
			printf("%12g|", value);
		}

		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			printf("%12g|", fabs(errorPerc[o]) * 100);
		}
		printf("\n");
	}
}

void CLAnnRelease(CLAnn * nn)
{
	CLMatrixRelease(nn->cholesky);
	CLMatrixRelease(nn->delta);
	CLMatrixRelease(nn->hessian);
	CLMatrixRelease(nn->jacobian);

	for (CLUInt i = 0; i < nn->nHiddenLayers; ++i) {
		CLMatrixRelease(nn->hActivations[i]);
	}
	CLMatrixRelease(nn->outputs);
	CLMatrixRelease(nn->weights);
//	CLMatrixRelease(nn->weightsTemp);
	CLMatrixRelease(nn->targets);
	CLMatrixRelease(nn->inputs);

	nn->cholesky = NULL;
	nn->delta = NULL;
	nn->hessian = NULL;
	nn->jacobian = NULL;
	nn->hActivations = NULL;
	nn->outputs = NULL;
	nn->weights = NULL;
	nn->targets = NULL;
	nn->inputs = NULL;

	CLReleaseKernel(nn->kernelActivation, kActivation);
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