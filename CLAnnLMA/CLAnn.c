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

#define kActivation "activation"
#define kChiSquared "chiSquared"
#define kChiSquaredReduce "chiSquaredReduce"
#define kJacobian "jacobian"
#define kDelta "delta"
#define kCholeskyDecomposition "choleskyDecomposition"

CLUInt BLOCK_SIZE = 32;

void CLAnnInit(CLAnn * nn, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddens, CLUInt nTargets, CLStringConst name)
{
	nn->name = malloc(sizeof(CLChar) * 1024);
	strcpy(nn->name, name);
	nn->nPatterns = nPatterns;
	nn->nInputs = nInputs;
	nn->nHiddens = nHiddens;
	nn->nTargets = nTargets;

	printf("/** %s **/\n"
		   "nPatters: %d\n"
		   "%d x %d x %d\n"
		   "/** %s **/\n", nn->name, nn->nPatterns, nn->nInputs, nn->nHiddens, nn->nTargets, nn->name);

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

	nn->verbose = CLTrue;
	nn->maxIteration = 10000;
	nn->initialLambda = 0.0001;
	nn->upFactor = 10.0f;
	nn->downFactor = 10.0f;
	nn->targetDeltaError = 1e-12f;
	nn->finalError = 0.0f;
	nn->finalDeltaError = 0.0f;

	CLUInt nWeights = nn->nInputs * nn->nHiddens + nn->nHiddens * nn->nTargets;

	CLMatrixInit(nn->inputs, nn->nPatterns, nn->nInputs, "inputs");
	CLMatrixInit(nn->targets, nn->nPatterns, nn->nTargets, "targets");

	CLMatrixInit(nn->weights, 1, nWeights, "weights");
	CLMatrixInit(nn->weightsTemp, 1, nWeights, "weightsTemp");
	CLMatrixInit(nn->outputs, nn->nPatterns, nn->nTargets, "outputs");

	CLMatrixInit(nn->hActivations, nn->nPatterns, nn->nHiddens, "hActivations");
	CLMatrixInit(nn->jacobian, nn->nPatterns * nn->nTargets, nWeights, "jacobian");
	CLMatrixInit(nn->hessian, nWeights, nWeights, "hessian");
	CLMatrixInit(nn->delta, 1, nWeights, "delta");
	CLMatrixInit(nn->cholesky, nWeights, nWeights, "cholesky");

	CLMatrixPrintStats(nn->inputs);
	CLMatrixPrintStats(nn->targets);
	CLMatrixPrintStats(nn->weights);
	CLMatrixPrintStats(nn->weightsTemp);
	CLMatrixPrintStats(nn->outputs);
	CLMatrixPrintStats(nn->hActivations);
	CLMatrixPrintStats(nn->jacobian);
	CLMatrixPrintStats(nn->hessian);
	CLMatrixPrintStats(nn->delta);
	CLMatrixPrintStats(nn->cholesky);


	CLSize totalSize = nn->inputs->size + nn->targets->size + nn->weights->size + nn->weightsTemp->size + nn->outputs->size + nn->hActivations->size + nn->jacobian->size + nn->hessian->size + nn->delta->size + nn->cholesky->size;
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

	//#define activationFunction(e) e
	//#define activationFunction(e) (1.0f / (1.0f + exp(-e)))
	//#define activationFunction(e) (2.0f/(1.0f + exp(-2.0f * e))-1.0f)
	switch (activationFunction) {
		case ACTIVATION_LINEAR:
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) e");
			break;
		case ACTIVATION_SIGMOID:
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) (1.0f / (1.0f + exp(-e)))");
			break;
		case ACTIVATION_TANSIG:
			nn->program = CLCreateProgramWithMacro(nn->context, nn->device, "Kernels.ocl", "#define activationFunction(e) (2.0f/(1.0f + exp(-2.0f * e))-1.0f)");
			break;

		default:
			break;
	}

	nn->kernelActivation = CLCreateKernel(nn->program, kActivation);
	nn->kernelChiSquared = CLCreateKernel(nn->program, kChiSquared);
	nn->kernelChiSquaredReduce = CLCreateKernel(nn->program, kChiSquaredReduce);
	nn->kernelJacobian = CLCreateKernel(nn->program, kJacobian);
	nn->kernelDelta = CLCreateKernel(nn->program, kDelta);
	nn->kernelCholeskyDecomposition = CLCreateKernel(nn->program, kCholeskyDecomposition);

	clblasSetup();

	CLMatrixCreateMemHostVar(nn->inputs, nn->context, CL_MEM_READ_ONLY);
	CLMatrixCreateMemHostVar(nn->targets, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->weightsTemp, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->outputs, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->hActivations, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->jacobian, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->hessian, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->delta, nn->context, CL_MEM_READ_ONLY);
//	CLMatrixCreateMemHostVar(nn->cholesky, nn->context, CL_MEM_READ_ONLY);

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

void CLAnnNormalizePatterns(CLAnn * nn)
{
//TODO: da sistemare
	clblasStatus status;
	CLEvent eventNormalizeInputs;

	CLMem normValue = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, sizeof(CLFloat), "normValue");
	CLMem scratchBuff = CLCreateBuffer(nn->context, CL_MEM_READ_WRITE, nn->inputs->size * 2, "scratchBuff");
	status = clblasSnrm2(nn->inputs->elements, normValue, 0, nn->inputs->mem, 0, 1, scratchBuff, 1, &nn->queue, 0, NULL, &eventNormalizeInputs);
	if (status != CL_SUCCESS) {
		debugLog("normalizePatterns errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventNormalizeInputs, "eventNormalize");

	CLEvent eventNormalizeTargets;
	status = clblasSnrm2(nn->targets->elements, normValue, 0, nn->targets->mem, 0, 1, scratchBuff, 1, &nn->queue, 0, NULL, &eventNormalizeTargets);
	if (status != CL_SUCCESS) {
		debugLog("normalizePatterns errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventNormalizeTargets, "eventNormalizeTargets");

	CLReleaseEvent(eventNormalizeInputs, "eventNormalize");
	CLReleaseEvent(eventNormalizeTargets, "eventNormalizeTargets");
	CLReleaseMemObject(normValue, "normValue");
	CLReleaseMemObject(scratchBuff, "scratchBuff");


	CLMatrixUpdateValuesFromMem(nn->inputs, nn->queue);
	CLMatrixUpdateValuesFromMem(nn->targets, nn->queue);

	CLMatrixPrint(nn->inputs, CLMatrixNoTrans);
	CLMatrixPrint(nn->targets, CLMatrixNoTrans);
}

void CLAnnForward(CLAnn * nn, CLUInt updateWeightsFromHost, CLUInt printOutputs)
{
	if (updateWeightsFromHost == CLTrue) {
		CLMatrixReleaseMem(nn->weights);
		CLMatrixCreateMemHostVar(nn->weights, nn->context, CL_MEM_READ_WRITE);
	}

	//Needs to recreate Mem because of clblasSgemm (i think it doesn't replace with new values but it does a sum with old values)
	CLMatrixReleaseMem(nn->outputs);
	CLMatrixCreateMemHostVar(nn->outputs, nn->context, CL_MEM_READ_WRITE);
	CLMatrixReleaseMem(nn->hActivations);
	CLMatrixCreateMemHostVar(nn->hActivations, nn->context, CL_MEM_READ_WRITE);

	CLMatrix * hWeights = malloc(sizeof(CLMatrix));
	CLMatrixInit(hWeights, nn->nInputs, nn->nHiddens, "hWeights");
	hWeights->mem = CLCreateSubBuffer(nn->weights->mem, CL_MEM_READ_ONLY, 0, hWeights->size, hWeights->name);

	//Stuff
	clblasStatus status;
	CLUInt nArg;

	//Inputs x hiddenWeights
	CLEvent eventResultIxHW;
	CLSize m = nn->hActivations->rows;
	CLSize n = nn->hActivations->columns;
	CLSize k = nn->inputs->columns;
	CLFloat alphaBeta = 1.0f;

	status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alphaBeta,
						 nn->inputs->mem, 0, k,
						 hWeights->mem, 0, n, alphaBeta,
						 nn->hActivations->mem, 0, n, 1,
						 &nn->queue, 0, NULL, &eventResultIxHW);
	if (status != CL_SUCCESS) {
		debugLog("resultIxHW errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventResultIxHW, "eventResultIxHW");

#if BENCHMARK
	CLSize loads = nn->inputs->elements + hWeights->elements;
	CLSize stores = nn->hActivations->elements;
	CLSize elements = loads + stores;
	CLSize dataSize = nn->inputs->size + hWeights->size + nn->hActivations->size;
	CLSize operations = 2 * m * n * k;
	CLBenchmarkLog(eventResultIxHW, eventResultIxHW, loads, stores, elements, dataSize, operations, "resultIxHW");
#endif

	//Activation hidden layer
	CLEvent eventActivation;
	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->hActivations->rows, lws[0]), CLGetOptimalGlobalWorkItemsSize(nn->hActivations->columns, lws[1])};

	nArg = 0;
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(nn->hActivations->mem), &nn->hActivations->mem, nn->hActivations->name);
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(CLUInt), &nn->hActivations->rows, "rows");
	CLSetKernelArg(nn->kernelActivation, nArg++, sizeof(CLUInt), &nn->hActivations->columns, "columns");

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelActivation, workDim, NULL, gws, lws, 0, NULL, &eventActivation, kActivation);
	CLWaitForEvent(&eventActivation, kActivation);

#if BENCHMARK
	loads = nn->hActivations->elements;
	stores = loads;
	elements = nn->hActivations->elements;
	dataSize = nn->hActivations->size;
	operations = 5 * elements;
	CLBenchmarkLog(eventActivation, eventActivation, loads, stores, elements, dataSize, operations, kActivation);
#endif

#if DEBUG_LOG
//	CLMatrixUpdateValuesFromMem(nn->hActivations, nn->queue);
//	CLMatrixPrint(nn->hActivations, CLMatrixNoTrans);
#endif

	//Outputs = hiddenActivations x outputsWeights
	CLMatrix * oWeights = malloc(sizeof(CLMatrix));
	CLMatrixInit(oWeights, nn->nHiddens, nn->nTargets, "oWeights");
	oWeights->mem = CLCreateSubBuffer(nn->weights->mem, CL_MEM_READ_ONLY, hWeights->size, oWeights->size, oWeights->name);

	CLEvent eventOutputs;
	m = nn->outputs->rows;
	n = nn->outputs->columns;
	k = nn->nHiddens;

	status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k, alphaBeta,
						 nn->hActivations->mem, 0, k,
						 oWeights->mem, 0, n, alphaBeta,
						 nn->outputs->mem, 0, n, 1,
						 &nn->queue, 0, NULL, &eventOutputs);
	if (status != CL_SUCCESS) {
		debugLog("resultHWxOW errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventOutputs, "eventOutputs");

#if BENCHMARK
	loads = nn->hActivations->elements + oWeights->elements;
	stores = nn->outputs->elements;
	elements = loads + stores;
	dataSize = nn->hActivations->size + oWeights->size + nn->outputs->size;
	operations = 2 * m * n * k;
	CLBenchmarkLog(eventActivation, eventActivation, loads, stores, elements, dataSize, operations, "resultHWxOW");
#endif

	if (printOutputs == CLTrue) {
		CLMatrixUpdateValuesFromMem(nn->outputs, nn->queue);
		CLMatrixPrint(nn->outputs, CLMatrixNoTrans);
	}

	//Releases
	CLReleaseEvent(eventResultIxHW, "eventResultIxHW");
	CLReleaseEvent(eventActivation, kActivation);
	CLReleaseEvent(eventOutputs, "eventOutputs");
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

	//Releases
	CLReleaseEvent(eventChiSquared, "eventChiSquared");
	CLReleaseEvent(eventChiSquaredReduce, "eventChiSquaredReduce");
	CLReleaseMemObject(chiSquaredError, "chiSquaredError");

	return error;
}

void CLAnnJacobian(CLAnn * nn)
{
	if (nn->hActivations->mem == NULL) {
		fprintf(stderr, "Call CLAnnForward() before!");
		return;
	}

	CLMatrixReleaseMem(nn->jacobian);
	CLMatrixCreateMemHostVar(nn->jacobian, nn->context, CL_MEM_READ_WRITE);

	CLUInt offset = 0;

	CLUInt nArg = 0;
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->inputs->mem), &nn->inputs->mem, nn->inputs->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->jacobian->columns, "jacobianColumns");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->rows, "rowsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->inputs->columns, "columnsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->nHiddens, "slope");

	CLEvent eventJacobian[2];
	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE, BLOCK_SIZE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(nn->inputs->columns, lws[0]), CLGetOptimalGlobalWorkItemsSize(nn->inputs->rows, lws[1])};

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobian[0], "jacobian[0]");
	CLWaitForEvent(&eventJacobian[0], "eventJacobian[0]");

#if BENCHMARK
	CLSize loads = nn->inputs->elements;
	CLSize stores = nn->nHiddens * nn->jacobian->rows;
	CLSize elements = nn->inputs->elements + (nn->nHiddens * nn->jacobian->rows);
	CLSize dataSize = sizeof(CLFloat) * (loads + stores);
	CLSize operations = 1;
	CLBenchmarkLog(eventJacobian[0], eventJacobian[0], loads, stores, elements, dataSize, operations, "jacobian[0]");
#endif

	offset = nn->nInputs * nn->nHiddens;

	nArg = 0;
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->jacobian->mem), &nn->jacobian->mem, nn->jacobian->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(nn->hActivations->mem), &nn->hActivations->mem, nn->hActivations->name);
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->jacobian->columns, "jacobianColumns");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->hActivations->rows, "rowsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->hActivations->columns, "columnsI");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
	CLSetKernelArg(nn->kernelJacobian, nArg++, sizeof(CLUInt), &nn->nTargets, "slope");

	gws[0] = CLGetOptimalGlobalWorkItemsSize(nn->hActivations->columns, lws[0]);
	gws[1] = CLGetOptimalGlobalWorkItemsSize(nn->hActivations->rows, lws[1]);

	CLEnqueueNDRangeKernel(nn->queue, nn->kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobian[1], "jacobian[1]");
	CLWaitForEvent(&eventJacobian[1], "eventJacobian[1]");

#if BENCHMARK
	loads = nn->hActivations->elements;
	stores = nn->nTargets * nn->jacobian->rows;
	elements = nn->hActivations->elements + (nn->nTargets * nn->jacobian->rows);
	dataSize = sizeof(CLFloat) * (loads + stores);
	operations = 1;
	CLBenchmarkLog(eventJacobian[1], eventJacobian[1], loads, stores, elements, dataSize, operations, "jacobian[1]");
#endif

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->jacobian, nn->queue);
	CLMatrixPrint(nn->jacobian, CLMatrixNoTrans);
#endif

	//Releases
	CLReleaseEvent(eventJacobian[0], "eventJacobian[0]");
	CLReleaseEvent(eventJacobian[1], "eventJacobian[1]");
}

//Hessian = J^T J
void CLAnnHessian(CLAnn * nn)
{
	if (nn->jacobian->mem == NULL) {
		fprintf(stderr, "Call CLAnnJacobian() before!");
		return;
	}

	CLMatrixReleaseMem(nn->hessian);
	CLMatrixCreateMemHostVar(nn->hessian, nn->context, CL_MEM_READ_WRITE);

	clblasStatus status;

	CLEvent eventHessian;
	CLSize m = nn->jacobian->columns;
	CLSize n = nn->jacobian->columns;
	CLSize k = nn->jacobian->rows;
	CLFloat alphaBeta = 1.0f;

	status = clblasSgemm(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k, alphaBeta,
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
	CLReleaseEvent(eventHessian, "eventHessian");
}

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

	if (nn->delta->mem == NULL) CLMatrixCreateMemHostVar(nn->delta, nn->context, CL_MEM_READ_WRITE);

	CLMatrixReleaseMem(nn->cholesky);
	CLMatrixCreateMemHostVar(nn->cholesky, nn->context, CL_MEM_READ_WRITE);

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

void CLAnnCholeskySolve(CLAnn * nn)
{
	if (nn->cholesky->mem == NULL || nn->delta->mem == NULL) {
		fprintf(stderr, "Call CLAnnCholeskyDecomposition() before!");
		return;
	}

	clblasStatus status;
	CLEvent eventCholeskySolve[2];
	status = clblasStrsv(clblasRowMajor, clblasUpper, clblasTrans, clblasNonUnit,
						 nn->cholesky->rows, nn->cholesky->mem, 0,
						 nn->cholesky->rows, nn->delta->mem, 0, 1,
						 1, &nn->queue, 0, NULL, &eventCholeskySolve[0]);
	if (status != CL_SUCCESS) {
		debugLog("CholeskySolve[0] errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventCholeskySolve[0], "eventCholeskySolve[0]");

	status = clblasStrsv(clblasRowMajor, clblasUpper, clblasNoTrans, clblasNonUnit,
						 nn->cholesky->rows, nn->cholesky->mem, 0,
						 nn->cholesky->rows, nn->delta->mem, 0, 1,
						 1, &nn->queue, 0, NULL, &eventCholeskySolve[1]);
	if (status != CL_SUCCESS) {
		debugLog("CholeskySolve[1] errorCode: %d", status);
		exit(status);
	}
	CLWaitForEvent(&eventCholeskySolve[1], "eventCholeskySolve[1]");

#if DEBUG_LOG
	CLMatrixUpdateValuesFromMem(nn->delta, nn->queue);
	CLMatrixPrint(nn->delta, CLMatrixNoTrans);
#endif

	//Releases
	CLReleaseEvent(eventCholeskySolve[0], "eventCholeskySolve[0]");
	CLReleaseEvent(eventCholeskySolve[1], "eventCholeskySolve[1]");
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
	status = clblasSaxpy(nn->weights->elements, 1.0f, nn->delta->mem, 0, 1, nn->weights->mem, 0, 1, 1, &nn->queue, 0, NULL, &eventUpdateWeights);
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

CLUInt CLAnnTraining(CLAnn * nn, CLBool normalizePatterns) {

	CLUInt it;
	CLFloat mult;
	CLFloat lambda = nn->initialLambda;
	CLUInt ill;
	CLFloat error = -1.0f;
	CLFloat newError = -1.0f;
	CLFloat deltaError = -1.0f;

	if (normalizePatterns == CLTrue) {
		CLAnnNormalizePatterns(nn);
	}

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

	for (CLUInt p = 0; p < nn->nPatterns; ++p) {
		for (CLUInt o = 0; o < nn->nTargets; ++o) {
			CLFloat target = nn->targets->values[p * nn->nTargets + o];
			CLFloat output = nn->outputs->values[p * nn->nTargets + o];
			CLFloat diff = target - output;
			CLFloat perc = (fabs(target) < CLFTollerance) ? NAN : (diff / target * 100.0f);

			for (CLUInt i = 0; i < nn->nTargets; ++i) {
				printf("%12g|", target);
			}
			for (CLUInt i = 0; i < nn->nTargets; ++i) {
				printf("%12g|", output);
			}
			for (CLUInt i = 0; i < nn->nTargets; ++i) {
				printf("%12g|", perc);
			}
			printf("\n");
		}
	}
}

void CLAnnRelease(CLAnn * nn)
{
	CLMatrixRelease(nn->cholesky);
	CLMatrixRelease(nn->delta);
	CLMatrixRelease(nn->hessian);
	CLMatrixRelease(nn->jacobian);
	CLMatrixRelease(nn->hActivations);
	CLMatrixRelease(nn->outputs);
	CLMatrixRelease(nn->weights);
	CLMatrixRelease(nn->weightsTemp);
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