//
//  CLNet.c
//  CLAnnLMA
//
//  Created by Albertomac on 1/26/17.
//  Copyright © 2017 Albertomac. All rights reserved.
//

#include "CLNet.h"
#include "CLRandom.h"
#include "CLBenchmark.h"
#include <math.h>

#ifdef __APPLE__
#include </usr/local/include/clBLAS.h>
#else
#include <clBLAS.h>
#endif

#define BUFFER_STRING 64

#define BLOCK_SIZE_MEMSET 64
#define BLOCK_SIZE_FUNCTION 32
#define BLOCK_SIZE_CHI_SQUARED 64

#define BLOCK_SIZE_JACOBIAN_DIAGONAL 32
#define BLOCK_SIZE_JACOBIAN_MULTIPLY 32
#define BLOCK_SIZE_JACOBIAN 32

#define BLOCK_SIZE_DELTA 64
#define BLOCK_SIZE_HESSIAN_UPDATE 32
#define BLOCK_SIZE_CHOLESKY_DECOMPOSITION 1024

#define NUMBER_OF_EVENTS 16

#pragma mark CLDeviceContext

#define nCommandQueue 16

void CLDeviceContextInit(CLDeviceContext * devContext, CLPlatform platform, CLDevice device)
{
	//platform
	devContext->platform = platform;

	//device
	devContext->device = device;

	//context
	devContext->context = CLCreateContext(devContext->platform, devContext->device);

	//queue
	devContext->queue = calloc(nCommandQueue, sizeof(CLQueue));
	devContext->queue[0] = CLCreateQueue(devContext->context, devContext->device);
	for (CLUInt i = 1; i < nCommandQueue; ++i) {
		devContext->queue[i] = devContext->queue[0];
		clRetainCommandQueue(devContext->queue[0]);
	}

	//program
	devContext->program = CLCreateProgramWithMacro(devContext->context, devContext->device, "Kernels.ocl", (CLNetPrecisionDouble == CLTrue ? "#define CLNetPrecisionDouble 1" : "#define CLNetPrecisionDouble 0"));

	//kernelMemSet
	devContext->kernelMemSet = CLCreateKernel(devContext->program, kMemSet);

	//kernelsActivation
	devContext->kernelsFunction = calloc(nFunctions, sizeof(CLKernel));
	devContext->kernelsFunction[CLFunctionLinear] = CLCreateKernel(devContext->program, kLinear);
	devContext->kernelsFunction[CLFunctionSigmoid] = CLCreateKernel(devContext->program, kSigmoid);
	devContext->kernelsFunction[CLFunctionTansig] = CLCreateKernel(devContext->program, kTansig);
	devContext->kernelsFunction[CLFunctionRadbas] = CLCreateKernel(devContext->program, kRadbas);

	//kernelChiSquared
	devContext->kernelChiSquared = CLCreateKernel(devContext->program, kChiSquared);

	//kernelChiSquaredReduce
	devContext->kernelChiSquaredReduce = CLCreateKernel(devContext->program, kChiSquaredReduce);

	//kernelJacobianDiagonal
	devContext->kernelJacobianDiagonal = CLCreateKernel(devContext->program, kJacobianDiagonal);

	//kernelJacobianMultiply
	devContext->kernelJacobianMultiply = CLCreateKernel(devContext->program, kJacobianMultiply);

	//kernelDelta
	devContext->kernelDelta = CLCreateKernel(devContext->program, kDelta);

	//kernelUpdateDiagonal
	devContext->kernelUpdateDiagonal = CLCreateKernel(devContext->program, kUpdateDiagonal);

	//kernelCholeskyDecomposition
	devContext->kernelCholeskyDecomposition = CLCreateKernel(devContext->program, kCholeskyDecomposition);
}

void CLDeviceContextCleanUp(CLDeviceContext * devContext)
{
	CLReleaseKernel(devContext->kernelMemSet, kMemSet);
	CLReleaseKernel(devContext->kernelsFunction[CLFunctionLinear], kLinear);
	CLReleaseKernel(devContext->kernelsFunction[CLFunctionSigmoid], kSigmoid);
	CLReleaseKernel(devContext->kernelsFunction[CLFunctionTansig], kTansig);
	CLReleaseKernel(devContext->kernelsFunction[CLFunctionRadbas], kRadbas);
	CLReleaseKernel(devContext->kernelChiSquared, kChiSquared);
	CLReleaseKernel(devContext->kernelChiSquaredReduce, kChiSquaredReduce);
	CLReleaseKernel(devContext->kernelJacobianDiagonal, kJacobianDiagonal);
	CLReleaseKernel(devContext->kernelJacobianMultiply, kJacobianMultiply);
	CLReleaseKernel(devContext->kernelDelta, kDelta);
	CLReleaseKernel(devContext->kernelUpdateDiagonal, kUpdateDiagonal);
	CLReleaseKernel(devContext->kernelCholeskyDecomposition, kCholeskyDecomposition);

	CLReleaseProgram(devContext->program, "Kernels.ocl");

	for (CLUInt i = 0; i < nCommandQueue; ++i) {
		CLReleaseQueue(devContext->queue[0], "queue");
	}
	CLReleaseContext(devContext->context, "context");
	CLReleaseDevice(devContext->device, "device");
}


#pragma mark CLNet

void printMatrixToFile(CLDeviceContext * devContext, CLMatrix * matrix, CLStringConst path)
{
	FILE * f = fopen(path, "w+");
	CLMatrixUpdateValuesFromMem(matrix, devContext->queue[0]);

	for (CLUInt i = 0; i < matrix->rows; ++i) {
		for (CLUInt j = 0; j < matrix->columns; ++j) {
			fprintf(f, "%.16g\t", matrix->values[i * matrix->columns + j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
}

void printMatrix(CLDeviceContext * devContext, CLMatrix * matrix)
{
	CLMatrixUpdateValuesFromMem(matrix, devContext->queue[0]);
	CLMatrixPrint(matrix, CLMatrixNoTrans);
//
//	CLString path = calloc(BUFFER_STRING, sizeof(CLChar));
//	snprintf(path, BUFFER_STRING - 1, ramDisk"/%s.txt", matrix->name);
//	printMatrixToFile(devContext, matrix, path);
//
//	free(path);
//	path = NULL;
}

void swapRow(CLNetDataType * matrix, CLUInt rows, CLUInt columns, CLUInt fromRow, CLUInt toRow)
{
	for (CLUInt i = 0; i < columns; ++i) {
		CLNetDataType tmp = matrix[fromRow * columns + i];
		matrix[fromRow * columns + i] = matrix[toRow * columns + i];
		matrix[toRow * columns + i] = tmp;
	}
}

void shufflePatterns(CLNetDataType * p, CLNetDataType * t, CLUInt nPatterns, CLUInt nInputs, CLUInt nTargets)
{
	if (nPatterns > 1) {
		for (CLUInt i = nPatterns - 1; i > 0; --i) {
			CLUInt j = CLRandomValue() * (i + 1);
			swapRow(p, nPatterns, nInputs, i, j);
			swapRow(t, nPatterns, nTargets, i, j);
		}
	}
}

CLUInt getReferenceCountEvent(CLEvent event)
{
	CLUInt count;
	CLInt error = clGetEventInfo(event, CL_EVENT_REFERENCE_COUNT, sizeof(CLUInt), &count, NULL);
	CLErrorCheck(error, "clGetEventInfo", "", CHECK_NOT_EXIT);
	return count;
}

void CLNetMemSet(CLNet * net, CLDeviceContext * devContext, CLMem mem, CLUInt elements, CLNetDataType value, CLStringConst name)
{
	CLEvent eventMemSet;
	CLSize lws[] = {BLOCK_SIZE_MEMSET};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(elements, lws[0])};

	CLUInt nArg = 0;
	CLKernel kernelMemSet = devContext->kernelMemSet;
	CLSetKernelArg(kernelMemSet, nArg++, sizeof(mem), &mem, name);
	CLSetKernelArg(kernelMemSet, nArg++, sizeof(CLUInt), &elements, "elements");
	CLSetKernelArg(kernelMemSet, nArg++, sizeof(CLNetDataType), &value, "value");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelMemSet, 1, 0, gws, lws, 0, NULL, &eventMemSet, kMemSet);
	CLReleaseEvent(eventMemSet, kMemSet);
}

void CLNetMemSetMatrix(CLNet * net, CLDeviceContext * devContext, CLMatrix * matrix, CLNetDataType value)
{
	CLNetMemSet(net, devContext, matrix->mem, matrix->elements, value, matrix->name);
}

void printStat(CLDouble flops, CLDouble time, CLStringConst name)
{
	printf("%s:\t\t\t%g GB/ms\t%g ms\n", name, sizeof(CLNetDataType) * flops / time, time * 1e-6);
}

void CLNetInit(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * patterns,
			   CLUInt nLayers, CLUInt * neuronsPerLayer, CLFunction * activationFunctionPerLayer,
			   CLUInt nTargets, CLNetDataType * targets,
			   CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns, CLBool bias)
{
	if (neuronsPerLayer[nLayers - 1]  != nTargets) {
		fprintf(stderr, "nTargets must be the same of last value of neuronsPerLayer\n");
		exit(-1);
	}

	if (patterns == NULL) {
		fprintf(stderr, "patterns must be not NULL\n");
		exit(-2);
	}

	if (neuronsPerLayer == NULL) {
		fprintf(stderr, "neuronsPerLayer must be not NULL\n");
		exit(-3);
	}

	if (activationFunctionPerLayer == NULL) {
		fprintf(stderr, "activationFunctionPerLayer must be not NULL\n");
		exit(-4);
	}

	if (targets == NULL) {
		fprintf(stderr, "targets must be not NULL\n");
		exit(-5);
	}

	//bias
	net->bias = bias;

	//nPatterns
	net->nPatterns = nPatterns;

	//nInputs
	net->nInputs = nInputs + (bias == CLTrue ? 1 : 0);

	//patterns
	net->p = calloc(net->nPatterns * net->nInputs, sizeof(*patterns));
	memcpy(net->p, patterns, sizeof(CLNetDataType) * net->nPatterns * net->nInputs);

	net->p = calloc(net->nPatterns * net->nInputs, sizeof(*patterns));
	if (net->bias == CLTrue) {
		for (CLUInt i = 0; i < net->nPatterns; ++i) {
			for (CLUInt j = 0; j < net->nInputs; j++) {
				net->p[i * net->nInputs + j] = (j < net->nInputs - 1) ? patterns[i * (net->nInputs - 1) + j] : 1;
			}
		}
	} else {
		memcpy(net->p, patterns, sizeof(CLNetDataType) * net->nPatterns * net->nInputs);
	}

	//nLayers (hidden + output)
	net->nLayers = nLayers;

	//nNeuronsPerHiddenLayer
	net->neuronsPerLayer = calloc(net->nLayers, sizeof(CLUInt));
	memcpy(net->neuronsPerLayer, neuronsPerLayer, sizeof(CLUInt) * net->nLayers);

	//activationFunctionPerLayer
	net->activationFunctionPerLayer = calloc(net->nLayers, sizeof(CLFunction));
	memcpy(net->activationFunctionPerLayer, activationFunctionPerLayer, sizeof(CLFunction) * net->nLayers);

	//nWeights
	net->nWeights = net->nInputs * net->neuronsPerLayer[0];
	for (CLUInt i = 1; i < net->nLayers; ++i) {
		net->nWeights += net->neuronsPerLayer[i - 1] * net->neuronsPerLayer[i];
	}

	//weights
	net->w = calloc(net->nWeights, sizeof(CLNetDataType));

	//nTargets
	net->nTargets = nTargets;

	//targets
	net->t = calloc(net->nPatterns * net->nTargets, sizeof(CLNetDataType));
	memcpy(net->t, targets, sizeof(CLNetDataType) * net->nPatterns * net->nTargets);

	//name
	net->name = calloc(BUFFER_STRING, sizeof(CLChar));
	snprintf(net->name, BUFFER_STRING - 1, "%s", name);

	//shufflePatterns
	if (shufflePattners == CLTrue) {
		shufflePatterns(net->p, net->t, net->nPatterns, net->nInputs, net->nTargets);
	}

	//nTestPatterns
	net->nTestPatterns = nTestPatterns;

	//nTrainingPatterns
	net->nTrainingPatterns = net->nPatterns - net->nTestPatterns;


	//Allocation of CLMatrix
	//testPatterns
	net->testPatterns = calloc(1, sizeof(CLMatrix));

	net->trainingPatterns = calloc(1, sizeof(CLMatrix));

	//weights
	net->weights = calloc(1, sizeof(CLMatrix));

	//weightsTemp
	net->weightsTemp = calloc(1, sizeof(CLMatrix));

	//weightsPerLayer
	net->weightsPerLayer = calloc(net->nLayers, sizeof(CLMatrix));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		net->weightsPerLayer[i] = calloc(1, sizeof(CLMatrix));
	}

	//valuesPerLayer
	net->valuesPerLayer = calloc(net->nLayers, sizeof(CLMatrix));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		net->valuesPerLayer[i] = calloc(1, sizeof(CLMatrix));
	}

	//derivativesPerLayer
	net->derivativesPerLayer = calloc(net->nLayers, sizeof(CLMatrix));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		net->derivativesPerLayer[i] = calloc(1, sizeof(CLMatrix));
	}

	//activationPerLayer
	net->activationPerLayer = calloc(net->nLayers, sizeof(CLMatrix));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		net->activationPerLayer[i] = calloc(1, sizeof(CLMatrix));
	}

	//outputs
	net->outputs = net->activationPerLayer[net->nLayers - 1];

	//testTargets
	net->testTargets = calloc(1, sizeof(CLMatrix));

	//trainingTargets
	net->trainingTargets = calloc(1, sizeof(CLMatrix));

	//chiSquaredError
	net->chiSquaredError = calloc(1, sizeof(CLMatrix));

	//jacobianPerLayer
	net->jacobianPerLayer = calloc(net->nLayers, sizeof(CLMatrix));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		net->jacobianPerLayer[i] = calloc(1, sizeof(CLMatrix));
	}

	//hessian
	net->hessian = calloc(1, sizeof(CLMatrix));

	//d
	net->d = calloc(1, sizeof(CLMatrix));

	//delta
	net->delta = calloc(1, sizeof(CLMatrix));

	//cholesky
	net->cholesky = calloc(1, sizeof(CLMatrix));

	//choleskySums
	net->choleskySums = calloc(1, sizeof(CLMatrix));

	//Levenberg-Marquartd stuff
	net->ill = CLFalse;
	net->verbose = CLTrue;
	net->maxIterations = 10000;
	net->initialLambda = 0.001;
	net->upFactor = 10.0;
	net->downFactor = 10.0;
	net->targetDeltaError = 1e-12f;
	net->finalError = 0.0;
	net->finalDeltaError = 0.0;

	//benchmark
	net->benchmark = CLFalse;
}

void CLNetInitWithFile(CLNet * net, CLStringConst fileName)
{
	//TODO: da fare dopo aver completato il progetto ed aver visto che funziona correttamente
}

CLUInt getLastEventIndex(CLEvent * events, CLUInt nEvents)
{
	for (CLUInt i = 1; i < nEvents; ++i) {
		if (events[i] == NULL) {
			return i - 1;
		}
	}
	return 0;
}

void CLNetMatrixMultiply(CLNet * net, CLDeviceContext * devContext,
						 CLMatrix * matrixA,
						 CLMatrix * matrixB,
						 CLMatrix * matrixResult)
{
	CLSize m = matrixA->rows;
	CLSize n = matrixB->columns;
	CLSize k = matrixA->columns;

	CLEvent * events = calloc(10, sizeof(CLEvent));

	clblasStatus status = GEMM(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k,
							   1, matrixA->mem, matrixA->offsetMem, k,
							   matrixB->mem, matrixB->offsetMem, n,
							   0, matrixResult->mem, matrixResult->offsetMem, n,
							   nCommandQueue, devContext->queue, 0, NULL, events);

	CLErrorCheck(status, "GEMM", "", CHECK_EXIT);

	CLUInt indexEvent = getLastEventIndex(events, NUMBER_OF_EVENTS);

#pragma mark BENCHMARK_MATRIX_MULTIPLY
	if (net->benchmark == CLTrue) {

		CLWaitForEvent(&events[indexEvent], "eventMatrixMultiply");

		CLSize loads = matrixA->elements + matrixB->elements;
		CLSize stores = matrixResult->elements;
		CLSize elements = matrixA->elements + matrixB->elements + matrixResult->elements;
		CLSize dataSize = elements * sizeof(CLNetDataType);
		CLSize operations = 2 * m * n * k;
		CLBenchmarkLog(events[0], events[indexEvent], loads, stores, elements, dataSize, operations, "matrixMultiply");
	}

	CLReleaseEvents(indexEvent + 1, events);
	free(events);
}


void CLNetComputeLayer(CLNet * net, CLDeviceContext * devContext,
					   CLMatrix * derivativeLayer, CLMatrix * activatedLayer,
					   CLMatrix * valuesLayer, CLFunction function)
{
	CLEvent event;
	CLSize lws[] = {BLOCK_SIZE_FUNCTION};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(valuesLayer->elements, lws[0])};

	CLUInt nArg = 0;
	CLKernel kernelActivation = devContext->kernelsFunction[function];
	CLSetKernelArg(kernelActivation, nArg++, sizeof(derivativeLayer->mem), &derivativeLayer->mem, derivativeLayer->name);
	CLSetKernelArg(kernelActivation, nArg++, sizeof(activatedLayer->mem), &activatedLayer->mem, activatedLayer->name);
	CLSetKernelArg(kernelActivation, nArg++, sizeof(valuesLayer->mem), &valuesLayer->mem, valuesLayer->name);
	CLSetKernelArg(kernelActivation, nArg++, sizeof(valuesLayer->elements), &valuesLayer->elements, "elements");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelActivation, 1, NULL, gws, lws, 0, NULL, &event, valuesLayer->name);

#pragma mark BENCHMARK_COMPUTE_LAYER
	if (net->benchmark == CLTrue) {
		CLWaitForEvent(&event, "event");

		CLSize loads = valuesLayer->elements;
		CLSize stores = valuesLayer->elements * 2;
		CLSize elements = valuesLayer->elements * 3;
		CLSize dataSize = elements * sizeof(CLNetDataType);
		CLSize operations = 1;
		switch (function) {
			case CLFunctionLinear:
				operations = stores;
				break;
			case CLFunctionSigmoid:
				operations = valuesLayer->elements * 10;
				break;
			case CLFunctionTansig:
				operations = valuesLayer->elements * 7;
				break;
			case CLFunctionRadbas:
				operations = valuesLayer->elements * 9;
				break;
			default:
				break;
		}
		CLStringConst label = "ComputeLayer";
		CLBenchmarkLog(event, event, loads, stores, elements, dataSize, operations, label);
	}

	CLReleaseEvent(event, "eventMatrixMultiply");
}

void CLNetForward(CLNet * net, CLDeviceContext * devContext)
{
	CLNetMatrixMultiply(net, devContext,
						net->trainingPatterns,
						net->weightsPerLayer[0],
						net->valuesPerLayer[0]);
	CLNetComputeLayer(net, devContext, net->derivativesPerLayer[0], net->activationPerLayer[0], net->valuesPerLayer[0], net->activationFunctionPerLayer[0]);

	for (CLUInt i = 1; i < net->nLayers; ++i) {
		CLNetMatrixMultiply(net, devContext,
							net->activationPerLayer[i - 1],
							net->weightsPerLayer[i],
							net->valuesPerLayer[i]);
		CLNetComputeLayer(net, devContext, net->derivativesPerLayer[i], net->activationPerLayer[i], net->valuesPerLayer[i], net->activationFunctionPerLayer[i]);
	}
}

void CLNetChiSquared(CLNet * net, CLDeviceContext * devContext)
{
	//ChiSquared
	CLEvent eventChiSquared;
	CLSize lws[] = {BLOCK_SIZE_CHI_SQUARED};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->outputs->elements, lws[0])};
	CLSize nwg = divUpSize(gws[0], lws[0]);

	CLUInt nArg = 0;
	CLKernel kernelChiSquared = devContext->kernelChiSquared;
	CLSetKernelArg(kernelChiSquared, nArg++, sizeof(net->outputs->mem), &net->outputs->mem, net->outputs->name);
	CLSetKernelArg(kernelChiSquared, nArg++, sizeof(net->trainingTargets->mem), &net->trainingTargets->mem, net->trainingTargets->name);
	CLSetKernelArg(kernelChiSquared, nArg++, sizeof(CLNetDataType) * BLOCK_SIZE_CHI_SQUARED, NULL, "localSums");
	CLSetKernelArg(kernelChiSquared, nArg++, sizeof(net->chiSquaredError->mem), &net->chiSquaredError->mem, "chiSquaredError");
	CLSetKernelArg(kernelChiSquared, nArg++, sizeof(CLUInt), &net->outputs->elements, "elements");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelChiSquared, 1, NULL, gws, lws, 0, NULL, &eventChiSquared, kChiSquared);


	//ChiSquaredReduce
	CLEvent eventChiSquaredReduce;
	gws[0] = nwg;
	lws[0] = nwg;
	nArg = 0;
	CLKernel kernelChiSquaredReduce = devContext->kernelChiSquaredReduce;
	CLSetKernelArg(kernelChiSquaredReduce, nArg++, sizeof(net->chiSquaredError->mem), &net->chiSquaredError->mem, "partialSums");
	CLSetKernelArg(kernelChiSquaredReduce, nArg++, sizeof(CLNetDataType) * lws[0], NULL, "localSums");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelChiSquaredReduce, 1, NULL, gws, lws, 0, NULL, &eventChiSquaredReduce, kChiSquaredReduce);


	//ReadError
	CLEvent eventReadErrorValue;
	CLInt error = clEnqueueReadBuffer(devContext->queue[0], net->chiSquaredError->mem, CLTrue, 0, sizeof(CLNetDataType), &net->errorChiSquared, 0, NULL, &eventReadErrorValue);
	CLErrorCheck(error, "clEnqueueReadBuffer", "read errorValue", CHECK_EXIT);

#pragma mark BENCHMARK_CHI_SQUARED
	if (net->benchmark == CLTrue) {
//
//		CLDouble time = 0;
//		CLDouble flops = 0;
//
//		time = timeBetweenEventsNS(eventChiSquared, eventChiSquared);
//		flops = 3 * net->outputs->elements;
//		printStat(flops, time, "chiSquared");
//
//		time = timeBetweenEventsNS(eventChiSquaredReduce, eventChiSquaredReduce);
//		flops = 3 * nwg;
//		printStat(flops, time, "chiSquaredReduce");
	}

	//Releases
	CLReleaseEvent(eventChiSquared, "eventChiSquared");
	CLReleaseEvent(eventChiSquaredReduce, "eventChiSquaredReduce");
	CLReleaseEvent(eventReadErrorValue, "eventReadErrorValue");
}

void CLNetJacobianDiagonal(CLNet * net, CLDeviceContext * devContext, CLMatrix * jacobian, CLUInt offset, CLMatrix * values, CLUInt dimDiag, CLEvent * event)
{
	CLSize lws[] = {BLOCK_SIZE_JACOBIAN_DIAGONAL,
					BLOCK_SIZE_JACOBIAN_DIAGONAL};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(jacobian->columns, lws[0]),
					CLGetOptimalGlobalWorkItemsSize(jacobian->rows, lws[1])};
	
	CLUInt nArgs = 0;
	CLKernel kernelJacobianDiagonal = devContext->kernelJacobianDiagonal;
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(jacobian->mem), &jacobian->mem, jacobian->name);
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(offset), &offset, "offset");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(jacobian->rows), &jacobian->rows, "rowsJacobian");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(jacobian->columns), &jacobian->columns, "columnsJacobian");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(values->mem), &values->mem, "values");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(values->rows), &values->rows, "rowsValues");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(values->columns), &values->columns, "columnsValues");
	CLSetKernelArg(kernelJacobianDiagonal, nArgs++, sizeof(dimDiag), &dimDiag, "nValues");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelJacobianDiagonal, 2, NULL, gws, lws, 0, NULL, event, "jacobianDiagonal");

#pragma BENCHMARK_JACOBIAN_DIAGONAL
	if (net->benchmark == CLTrue) {
	}
}

#define COMPUTE_JACOBIAN_MULTIPLY_LINEAR 0

void CLNetJacobianMultiply(CLNet * net, CLDeviceContext * devContext, CLMatrix * jacobian, CLMatrix * values, CLEvent * event)
{

#if COMPUTE_JACOBIAN_MULTIPLY_LINEAR
	CLSize lws[] = {BLOCK_SIZE_JACOBIAN_MULTIPLY};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(jacobian->elements, lws[0])};

	CLUInt nArgs = 0;
	CLKernel kernelJacobianMultiply = devContext->kernelJacobianMultiply;
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->mem), &jacobian->mem, jacobian->name);
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->elements), &jacobian->elements, "elementsJacobian");
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->columns), &jacobian->columns, "columnsJacobian");
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(values->mem), &values->mem, "values");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelJacobianMultiply, 1, NULL, gws, lws, 0, NULL, event, "jacobianMultiply");

#pragma BENCHMARK_JACOBIAN_MULTIPLY
	if (net->benchmark == CLTrue) {
	}
#else
	CLSize lws[] = {BLOCK_SIZE_JACOBIAN_MULTIPLY,
					BLOCK_SIZE_JACOBIAN_MULTIPLY};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(jacobian->columns, lws[0]),
					CLGetOptimalGlobalWorkItemsSize(jacobian->rows, lws[1])};

	CLUInt nArgs = 0;
	CLKernel kernelJacobianMultiply = devContext->kernelJacobianMultiply;
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->mem), &jacobian->mem, jacobian->name);
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->rows), &jacobian->rows, "rowsJacobian");
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(jacobian->columns), &jacobian->columns, "columnsJacobian");
	CLSetKernelArg(kernelJacobianMultiply, nArgs++, sizeof(values->mem), &values->mem, "values");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelJacobianMultiply, 2, NULL, gws, lws, 0, NULL, event, "jacobianMultiply");

#pragma BENCHMARK_JACOBIAN_MULTIPLY
	if (net->benchmark == CLTrue) {
	}
#endif
}

//TODO: creare 2 o più code per poter eseguire i kernel contemporaneamente
void CLNetJacobian(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent * eventJacobianDiagonal = calloc(net->nLayers, sizeof(CLEvent));
	CLEvent * eventJacobianMultiply = calloc(net->nLayers, sizeof(CLEvent));

	CLNetJacobianDiagonal(net, devContext, net->jacobianPerLayer[0], 0,
						  net->trainingPatterns, /*net->weightsPerLayer[0]->elements / net->nInputs*/ net->neuronsPerLayer[0],
						  &eventJacobianDiagonal[0]);
	CLNetJacobianMultiply(net, devContext, net->jacobianPerLayer[0], net->derivativesPerLayer[0], &eventJacobianMultiply[0]);

	for (CLUInt l = 1; l < net->nLayers; ++l) {

		CLSize m = net->weightsPerLayer[l]->columns;
		CLSize n = net->jacobianPerLayer[l - 1]->columns;
		CLSize k = net->weightsPerLayer[l]->rows;
		CLSize lda = net->weightsPerLayer[l]->columns;
		CLSize ldb = net->jacobianPerLayer[l - 1]->columns;
		CLSize ldc = net->jacobianPerLayer[l]->columns;

		for (CLUInt i = 0; i < net->nTrainingPatterns; ++i) {
			CLEvent * eventsGEMM = calloc(NUMBER_OF_EVENTS, sizeof(CLEvent));
			clblasStatus status = GEMM(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k,
									   1, net->weightsPerLayer[l]->mem, net->weightsPerLayer[l]->offsetMem, lda,
									   net->jacobianPerLayer[l - 1]->mem, i * ldb * net->neuronsPerLayer[l - 1], ldb,
									   0, net->jacobianPerLayer[l]->mem, i * ldc * net->neuronsPerLayer[l], ldc,
									   nCommandQueue, devContext->queue, 0, NULL, eventsGEMM);
			CLErrorCheck(status, "GEMM", "GEMM weights' x jacobianPerLayer[0]", CHECK_EXIT);

			CLUInt indexEvent = getLastEventIndex(eventsGEMM, NUMBER_OF_EVENTS);
			CLReleaseEvents(indexEvent + 1, eventsGEMM);

			free(eventsGEMM);
			eventsGEMM = NULL;
		}

		CLNetJacobianDiagonal(net, devContext, net->jacobianPerLayer[l], net->jacobianPerLayer[l - 1]->columns,
							  net->activationPerLayer[l - 1], /*net->weightsPerLayer[l]->elements / net->neuronsPerLayer[l - 1]*/ net->neuronsPerLayer[l],
							  &eventJacobianDiagonal[l]);
		CLNetJacobianMultiply(net, devContext, net->jacobianPerLayer[l], net->derivativesPerLayer[l], &eventJacobianMultiply[l]);
	}

#pragma mark BENCHMARK_JACOBIAN
	if (net->benchmark == CLTrue) {
	}

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		CLReleaseEvent(eventJacobianDiagonal[i], "eventJacobianDiagonal");
		CLReleaseEvent(eventJacobianMultiply[i], "eventJacobianMultiply");
	}

	free(eventJacobianDiagonal);
	eventJacobianDiagonal = NULL;
	free(eventJacobianMultiply);
	eventJacobianMultiply = NULL;
}

void CLNetHessian(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent * eventsHessian = calloc(NUMBER_OF_EVENTS, sizeof(CLEvent));
	CLSize m = net->jacobian->columns;
	CLSize n = net->jacobian->columns;
	CLSize k = net->jacobian->rows;

	clblasStatus status = GEMM(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k,
							   1, net->jacobian->mem, 0, m,
							   net->jacobian->mem, 0, n,
							   0, net->hessian->mem, 0, n,
							   nCommandQueue, devContext->queue, 0, NULL, eventsHessian);
	CLErrorCheck(status, "GEMM", "hessian", CHECK_EXIT);

	CLUInt indexEvent = getLastEventIndex(eventsHessian, NUMBER_OF_EVENTS);

#pragma mark BENCHMARK_HESSIAN
	if (net->benchmark == CLTrue) {
		CLWaitForEvent(&eventsHessian[indexEvent], "eventHessian");

		CLSize loads = 2 * net->jacobian->elements;
		CLSize stores = net->hessian->elements;
		CLSize elements = 2 * net->jacobian->elements + net->hessian->elements;
		CLSize dataSize = elements * sizeof(CLNetDataType);
		CLSize operations = 2 * m * n * k;
		CLBenchmarkLog(eventsHessian[0], eventsHessian[indexEvent], loads, stores, elements, dataSize, operations, "hessian");
	}

	CLReleaseEvents(indexEvent + 1, eventsHessian);
	free(eventsHessian);
	eventsHessian = NULL;
}

void CLNetCalculateD(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventAXPY;
	clblasStatus status = AXPY(net->trainingTargets->elements, -1, net->trainingTargets->mem, 0, 1, net->outputs->mem, 0, 1, 1, devContext->queue, 0, NULL, &eventAXPY);
	CLErrorCheck(status, "AXPY", "outputs = outputs - trainingTargetes", CHECK_EXIT);

	CLEvent eventGEMV;
	status = GEMV(clblasRowMajor, clblasTrans, net->jacobian->rows, net->jacobian->columns, -1, net->jacobian->mem, 0, net->jacobian->columns,
				  net->outputs->mem, 0, 1, 0, net->d->mem, 0, 1,
				  1, devContext->queue, 0, NULL, &eventGEMV);
	CLErrorCheck(status, "GEMV", "-J^T * (O - T)", CHECK_EXIT);

	CLReleaseEvent(eventAXPY, "eventAXPY");
	CLReleaseEvent(eventGEMV, "eventGEMV");

#pragma mark BENCHMARK_CALCULATE_D
	if (net->benchmark == CLTrue) {
	}
}

void CLNetUpdateHessianDiagonal(CLNet * net, CLDeviceContext * devContext, CLNetDataType mult)
{
	CLEvent eventUpdateDiagonal;

	CLSize lws[] = {BLOCK_SIZE_HESSIAN_UPDATE};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->hessian->rows, lws[0])};

	CLUInt nArgs = 0;
	CLKernel kernelUpdateDiagonal = devContext->kernelUpdateDiagonal;
	CLSetKernelArg(kernelUpdateDiagonal, nArgs++, sizeof(net->hessian->mem), &net->hessian->mem, net->hessian->name);
	CLSetKernelArg(kernelUpdateDiagonal, nArgs++, sizeof(net->hessian->rows), &net->hessian->rows, "dim");
	CLSetKernelArg(kernelUpdateDiagonal, nArgs++, sizeof(mult), &mult, "mult");

	CLEnqueueNDRangeKernel(devContext->queue[0], kernelUpdateDiagonal, 1, NULL, gws, lws, 0, NULL, &eventUpdateDiagonal, "updateHessianDiagonal");

#pragma BENCHMARK_UPDATE_HESSIAN_DIAGONAL
	if (net->benchmark == CLTrue) {
	}

	CLReleaseEvent(eventUpdateDiagonal, "eventUpdateDiagonal");
}

void CLNetCholeskyDecomposition(CLNet * net, CLDeviceContext * devContext)
{
	CLNetMemSetMatrix(net, devContext, net->choleskySums, 0);

	//Set ill to zero
	net->ill = 0;
	CLEvent eventWriteIll;
	clEnqueueWriteBuffer(devContext->queue[0], net->illMem, CLTrue, 0, sizeof(CLUInt), &net->ill, 0, NULL, &eventWriteIll);
	CLReleaseEvent(eventWriteIll, "eventWriteIll");

	//Cholesky
	CLSize lws[] = {BLOCK_SIZE_CHOLESKY_DECOMPOSITION};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->nWeights, lws[0])};

	CLUInt nArg = 0;
	CLKernel kernelCholeskyDecomposition = devContext->kernelCholeskyDecomposition;
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->cholesky->mem), &net->cholesky->mem, net->cholesky->name);
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->hessian->mem), &net->hessian->mem, net->hessian->name);
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->choleskySums->mem), &net->choleskySums->mem, net->choleskySums->name);
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(CLUInt), &net->nWeights, "npar");
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->illMem), &net->illMem, "ill");

	CLEvent * eventCholeskyDecomposition = calloc(net->cholesky->rows, sizeof(CLEvent));
	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {

		CLSetKernelArg(kernelCholeskyDecomposition, nArg, sizeof(CLUInt), &i, "row");
		CLEnqueueNDRangeKernel(devContext->queue[0], kernelCholeskyDecomposition, 1, NULL, gws, lws, 0, NULL, eventCholeskyDecomposition+i, kCholeskyDecomposition);
	}

	//Read Ill
	CLEvent eventReadIll;
	CLUInt error = clEnqueueReadBuffer(devContext->queue[0], net->illMem, CLTrue, 0, sizeof(CLUInt), &net->ill, 0, NULL, &eventReadIll);
	CLErrorCheck(error, "clEnqueueReadBuffer", "read errorValue", CHECK_EXIT);


#pragma mark BENCHMARK_CHOLESKY_DECOMPOSITION
	if (net->benchmark == CLTrue) {
//		CLDouble time = timeBetweenEventsNS(eventCholeskyDecomposition[0], eventCholeskyDecomposition[net->cholesky->rows - 1]);
//		CLDouble flops = 1;
//
////		printStat(flops, time, "choleskyDecomposition");
	}

	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {
		CLReleaseEvent(eventCholeskyDecomposition[i], "eventCholeskyDecomposition");
	}

	CLReleaseEvent(eventReadIll, "eventReadErrorValue");

	free(eventCholeskyDecomposition);
	eventCholeskyDecomposition = NULL;
}

void CLNetSolveTriangular(CLNet * net, CLDeviceContext * devContext, CLMatrix * cholesky, CLMatrix * delta, clblasTranspose uplo)
{
	CLEvent * events = calloc(NUMBER_OF_EVENTS, sizeof(CLEvent));
	clblasStatus status = TRSV(clblasRowMajor, clblasUpper, uplo, clblasNonUnit,
							   cholesky->rows, cholesky->mem, 0,
							   cholesky->rows, delta->mem, 0, 1,
							   nCommandQueue, devContext->queue, 0, NULL, events);

	CLErrorCheck(status, "TRSV", "", CHECK_EXIT);

	CLUInt indexEvent = getLastEventIndex(events, NUMBER_OF_EVENTS);
	CLReleaseEvents(indexEvent + 1, events);
	free(events);
	events = NULL;
}

void CLNetCholeskySolve(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventCopyDelta;
	CLInt status = clEnqueueCopyBuffer(devContext->queue[0], net->d->mem, net->delta->mem, 0, 0, net->d->size, 0, NULL, &eventCopyDelta);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "copyDelta", CHECK_EXIT);

	CLNetSolveTriangular(net, devContext, net->cholesky, net->delta, clblasTrans);
	CLNetSolveTriangular(net, devContext, net->cholesky, net->delta, clblasNoTrans);

	CLReleaseEvent(eventCopyDelta, "eventCopyDelta");
}

void CLNetReloadWeights(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventReloadWeights;
	CLInt status = clEnqueueCopyBuffer(devContext->queue[0], net->weightsTemp->mem, net->weights->mem, 0, 0, net->weightsTemp->size, 0, NULL, &eventReloadWeights);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);
	CLReleaseEvent(eventReloadWeights, "eventReloadWeights");
}

void CLNetUpdateWeightsTemp(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeightsTemp;
	CLInt status = clEnqueueCopyBuffer(devContext->queue[0], net->weights->mem, net->weightsTemp->mem, 0, 0, net->weights->size, 0, NULL, &eventUpdateWeightsTemp);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weights -> weightsTemp", CHECK_EXIT);
	CLReleaseEvent(eventUpdateWeightsTemp, "eventUpdateWeightsTemp");
}

void CLNetUpdateWeightsWithDelta(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeights;
	clblasStatus status = AXPY(net->weights->elements, 1, net->delta->mem, 0, 1, net->weights->mem, 0, 1, 1, devContext->queue, 0, NULL, &eventUpdateWeights);
	CLErrorCheck(status, "AXPY", "updateWeightsWithDelta", CHECK_EXIT);
	CLReleaseEvent(eventUpdateWeights, "eventUpdateWeights");
}

void CLNetInitializeWeights(CLNet * net, CLDeviceContext * devContext)
{
	CLMatrixFillRandomWithMult(net->weights, 1, -0.5);
	CLEvent eventInitializeWeights;
	CLInt status = clEnqueueWriteBuffer(devContext->queue[0], net->weights->mem, CLTrue, 0, net->weights->size, net->weights->values, 0, NULL, &eventInitializeWeights);
	CLErrorCheck(status, "clEnqueueWriteBuffer", "initializeWeights", CHECK_EXIT);
	CLReleaseEvent(eventInitializeWeights, "eventInitializeWeights");

	CLNetUpdateWeightsTemp(net, devContext);
}

void CLNetTrainLMA(CLNet * net, CLDeviceContext * devContext)
{
	CLNetInitializeWeights(net, devContext);

	net->benchmark = CLTrue;

	if (net->benchmark) {
		CLBenchmarkSetup("/Volumes/RamDisk/benchmark.csv");
	}

	CLNetDataType mult;
	CLNetDataType lambda = net->initialLambda;
	CLNetDataType error = -1.0f;
	CLNetDataType newError = -1.0f;
	CLNetDataType deltaError = -1.0f;
	CLBool ill;

	CLNetForward(net, devContext);									//Forward per calcolare l'errore iniziale
	CLNetChiSquared(net, devContext);								//Calcolo dell'errore iniziale
	error = net->errorChiSquared;

	for (CLUInt i = 0; i < net->maxIterations; ++i) {

		CLNetReloadWeights(net, devContext);
		CLNetForward(net, devContext);								//Forward all'inizio dell'iterazione per ricalcolare le matrici Jacobian e Hessian
		CLNetJacobian(net, devContext);								//Calcolo matrice Jacobian
		CLNetHessian(net, devContext);								//Calcolo matrice Hessian
		CLNetCalculateD(net, devContext);							//Calcolo array D

		mult = 1 + lambda;											//Aggiornamento moltiplicatore
		ill = 1;													//ill settato a 1 per entrare almeno una volta nel while
		//Non si può sostituire con il do/while per via della seconda condizione
		while (ill && (i < net->maxIterations)) {

			CLNetUpdateHessianDiagonal(net, devContext, mult);		//Aggiorno la diagonale dell'hessian
			CLNetCholeskyDecomposition(net, devContext);			//Calcolo della decomposizione di Cholesky
			ill = net->ill;

			if (!ill) {
				CLNetCholeskySolve(net, devContext);				//Risoluzione di Cholesky per il calcolo dei delta dei pesi
				CLNetReloadWeights(net, devContext);
				CLNetUpdateWeightsWithDelta(net, devContext);		//Aggiornamento dei pesi con i delta calcolati nello step precedente

				CLNetForward(net, devContext);						//Forward per ricalcolare l'errore
				CLNetChiSquared(net, devContext);
				newError = net->errorChiSquared;					//Calcolo del nuovo errore
				deltaError = newError - error;						//Calcolo del delta error
				ill = (deltaError > 0);								//Aggiornamento di ill a 0 se il delta error è negativo
			}

			printf("it = %4d,   lambda = %10g,   err = %10g,   derr = %10g    choleskyIll = %d\n", i, lambda, error, deltaError, net->ill);

			if (ill) {												//Se ill è ancora 1, vengono aggiornati i moltiplicatori
				mult = (1 + lambda * net->upFactor)/(1 + lambda);
				lambda *= net->upFactor;
				i++;
			}

			if (isnan(newError) || lambda > 1e7 || lambda < 1e-20) {
				CLNetInitializeWeights(net, devContext);
				lambda = net->initialLambda;
				i = 0;
				printf("\n\n WEIGHTS RE-INITIALIZED\n\n");
				CLNetForward(net, devContext);									//Forward per calcolare l'errore iniziale
				CLNetChiSquared(net, devContext);								//Calcolo dell'errore iniziale
				error = net->errorChiSquared;
				
				break;
			}
		}
		CLNetUpdateWeightsTemp(net, devContext);					//I nuovi pesi vengono salvati

		error = newError;
		lambda /= net->downFactor;

		if ((!ill) && (-deltaError < net->targetDeltaError)) break;
	}

	net->finalError = error;
	net->finalDeltaError = deltaError;
}

void CLNetPrintForward(CLNet * net, CLDeviceContext * devContext)
{
	CLMatrixUpdateValuesFromMem(net->outputs, devContext->queue[0]);

	if (net->nInputs < 3){
		for (CLUInt i = 0; i < net->nInputs - (net->bias == CLTrue ? 1 : 0); ++i) {
			printf("  Inputs[%2d]  |", i);
		}
	}

	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf("  Target[%2d]  |", i);
	}
	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf("  Output[%2d]  |", i);
	}
	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf("   Error%%[%2d]  |", i);
	}
		printf("\n");

#define PRINT_VALUE " %12g"

	for (CLUInt p = 0; p < net->nTrainingPatterns; ++p) {

		if (net->nInputs < 3) {
			for (CLUInt i = 0; i < net->nInputs - (net->bias == CLTrue ? 1 : 0); ++i) {
				printf(PRINT_VALUE " |", net->trainingPatterns->values[p * net->nInputs + i]);
			}
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			CLNetDataType value = net->trainingTargets->values[p * net->nTargets + o];
			printf(PRINT_VALUE " |", value);
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			CLNetDataType value = net->outputs->values[p * net->nTargets + o];
			printf(PRINT_VALUE " |", value);
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			CLNetDataType valueTarget = net->trainingTargets->values[p * net->nTargets + o];
			CLNetDataType valueOutput = net->outputs->values[p * net->nTargets + o];

			CLNetDataType errorPercValue = (valueTarget == 0) ? valueOutput * 100 : fabs(valueTarget - valueOutput) / valueTarget * 100;
			printf( (fabs(errorPercValue) > 20.0f ? PRINT_VALUE "♥️|" : PRINT_VALUE"💚|"), errorPercValue);
		}
		printf("\n");
	}
}


void CLNetTrainWithDeviceContext(CLNet * net, CLDeviceContext * devContext)
{
	net->partialJacobianFilled = CLFalse;

	//testPatterns
	CLMatrixInit(net->testPatterns, net->nTestPatterns, net->nInputs, "testPatterns");
	memcpy(net->testPatterns->values, net->p, net->testPatterns->size);
	CLMatrixCreateMemHostVar(net->testPatterns, devContext->context, CL_MEM_READ_ONLY);

	//trainingPatterns
	CLMatrixInit(net->trainingPatterns, net->nTrainingPatterns, net->nInputs, "patterns");
	memcpy(net->trainingPatterns->values, net->p + net->testPatterns->elements, net->trainingPatterns->size);
	CLMatrixCreateMemHostVar(net->trainingPatterns, devContext->context, CL_MEM_READ_ONLY);

	//weights
	CLMatrixInit(net->weights, 1, net->nWeights, "weights");
	CLMatrixUpdateValues(net->weights, net->w);
	CLMatrixCreateMemHostVar(net->weights, devContext->context, CL_MEM_READ_WRITE);

	//weightsTemp
	CLMatrixInit(net->weightsTemp, 1, net->nWeights, "weightsTemp");
	CLMatrixUpdateValues(net->weightsTemp, net->w);
	CLMatrixCreateMemHostVar(net->weightsTemp, devContext->context, CL_MEM_READ_WRITE);

	//weightsPerLayer
	CLString weightsPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	snprintf(weightsPerLayerName, BUFFER_STRING - 1, "weightsPerLayer[%d]", 0);
	CLMatrixInit(net->weightsPerLayer[0], net->nInputs, net->neuronsPerLayer[0], weightsPerLayerName);

	CLSize offset = 0;
	net->weightsPerLayer[0]->offsetMem = offset;
	net->weightsPerLayer[0]->mem = net->weights->mem;
	clRetainMemObject(net->weights->mem);
	offset += net->weightsPerLayer[0]->elements;

	for (CLUInt i = 1; i < net->nLayers; ++i) {
		snprintf(weightsPerLayerName, BUFFER_STRING - 1, "weightsPerLayer[%d]", i);
		CLMatrixInit(net->weightsPerLayer[i], net->neuronsPerLayer[i - 1], net->neuronsPerLayer[i], weightsPerLayerName);

		net->weightsPerLayer[i]->offsetMem = offset;
		net->weightsPerLayer[i]->mem = net->weights->mem;
		clRetainMemObject(net->weights->mem);
		offset += net->weightsPerLayer[i]->elements;
	}
	free(weightsPerLayerName);
	weightsPerLayerName = NULL;

	//valuesPerLayer
	CLString valuesPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(valuesPerLayerName, BUFFER_STRING - 1, "valuesPerLayer[%d]", i);
		CLMatrixInit(net->valuesPerLayer[i], net->nTrainingPatterns, net->neuronsPerLayer[i], valuesPerLayerName);
		CLMatrixCreateMemHostVar(net->valuesPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(valuesPerLayerName);
	valuesPerLayerName = NULL;

	//derivativesPerLayer
	CLString derivativesPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(derivativesPerLayerName, BUFFER_STRING - 1, "derivativesPerLayer[%d]", i);
		CLMatrixInit(net->derivativesPerLayer[i], net->nTrainingPatterns, net->neuronsPerLayer[i], derivativesPerLayerName);
		CLMatrixCreateMemHostVar(net->derivativesPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(derivativesPerLayerName);
	derivativesPerLayerName = NULL;

	//activationPerLayer
	CLString activationPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(activationPerLayerName, BUFFER_STRING - 1, "activationPerLayer[%d]", i);
		CLMatrixInit(net->activationPerLayer[i], net->nTrainingPatterns, net->neuronsPerLayer[i], activationPerLayerName);
		CLMatrixCreateMemHostVar(net->activationPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(activationPerLayerName);
	activationPerLayerName = NULL;

	//chiSquaredError
	CLSize chiSquaredErrorColumns = divUpSize(CLGetOptimalGlobalWorkItemsSize(net->outputs->elements, BLOCK_SIZE_CHI_SQUARED), BLOCK_SIZE_CHI_SQUARED);
	CLMatrixInit(net->chiSquaredError, 1, (CLUInt)chiSquaredErrorColumns, "chiSquaredError");
	CLMatrixCreateMemHostVar(net->chiSquaredError, devContext->context, CL_MEM_READ_WRITE);

	//testTargets
	CLMatrixInit(net->testTargets, net->nTestPatterns, net->nTargets, "testTargets");
	memcpy(net->testTargets->values, net->t, net->testTargets->size);
	CLMatrixCreateMemHostVar(net->testTargets, devContext->context, CL_MEM_READ_ONLY);

	//trainingPatterns
	CLMatrixInit(net->trainingTargets, net->nTrainingPatterns, net->nTargets, "patterns");
	memcpy(net->trainingTargets->values, net->t + net->testTargets->elements, net->trainingTargets->size);
	CLMatrixCreateMemHostVar(net->trainingTargets, devContext->context, CL_MEM_READ_ONLY);

	//jacobianPerLayer
	CLString jacobianPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	snprintf(jacobianPerLayerName, BUFFER_STRING - 1, "jacobianPerLayer[%d]", 0);
	CLUInt jacobianWeights = 0;

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(jacobianPerLayerName, BUFFER_STRING - 1, "jacobianPerLayer[%d]", i);
		jacobianWeights += net->weightsPerLayer[i]->elements;
		CLMatrixInit(net->jacobianPerLayer[i], net->neuronsPerLayer[i] * net->nTrainingPatterns, jacobianWeights, jacobianPerLayerName);
		CLMatrixCreateMemHostVar(net->jacobianPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(weightsPerLayerName);
	weightsPerLayerName = NULL;

	//jacobian
	net->jacobian = net->jacobianPerLayer[net->nLayers - 1];

	//hessian
	CLMatrixInit(net->hessian, net->nWeights, net->nWeights, "hessian");
	CLMatrixCreateMemHostVar(net->hessian, devContext->context, CL_MEM_READ_WRITE);

	//d
	CLMatrixInit(net->d, 1, net->nWeights, "d");
	CLMatrixCreateMemHostVar(net->d, devContext->context, CL_MEM_READ_WRITE);

	//delta
	CLMatrixInit(net->delta, 1, net->nWeights, "delta");
	CLMatrixCreateMemHostVar(net->delta, devContext->context, CL_MEM_READ_WRITE);

	//cholesky
	CLMatrixInit(net->cholesky, net->nWeights, net->nWeights, "cholesky");
	CLMatrixCreateMemHostVar(net->cholesky, devContext->context, CL_MEM_READ_WRITE);

	//choleskySums
	CLMatrixInit(net->choleskySums, 1, net->nWeights, "choleskySums");
	CLMatrixCreateMemHostVar(net->choleskySums, devContext->context, CL_MEM_READ_WRITE);

	//ill & illMem
	net->ill = CLFalse;
	net->illMem = CLCreateBuffer(devContext->context, CL_MEM_READ_WRITE, sizeof(CLNetDataType), "illMem");

	clblasSetup();

	CLNetTrainLMA(net, devContext);

	CLNetPrintForward(net, devContext);
}

void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * inputs)
{

}

void CLNetCleanUp(CLNet * net)
{
	clblasTeardown();

	free(net->name);
	free(net->neuronsPerLayer);
	free(net->activationFunctionPerLayer);
	free(net->p);
	free(net->w);
	free(net->t);

	CLMatrixRelease(net->testPatterns);
	CLMatrixRelease(net->trainingPatterns);
	CLMatrixRelease(net->weights);
	CLMatrixRelease(net->weightsTemp);

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		CLMatrixRelease(net->weightsPerLayer[i]);
		CLMatrixRelease(net->valuesPerLayer[i]);
		CLMatrixRelease(net->activationPerLayer[i]);

		CLMatrixRelease(net->jacobianPerLayer[i]);
	}

	net->outputs = NULL; //Remove pointer to last CLMatrix in activationPerLayer
	net->jacobian = NULL; //Remove pointer to last CLMatrix in jacobianPerLayer

	CLMatrixRelease(net->testTargets);
	CLMatrixRelease(net->trainingTargets);

	CLMatrixRelease(net->chiSquaredError);
	CLMatrixRelease(net->hessian);
	CLMatrixRelease(net->d);
	CLMatrixRelease(net->delta);
	CLMatrixRelease(net->cholesky);
	CLMatrixRelease(net->choleskySums);

	CLReleaseMemObject(net->illMem, "illMem");

	free(net);
}