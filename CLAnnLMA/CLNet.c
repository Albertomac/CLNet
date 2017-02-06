//
//  CLNet.c
//  CLAnnLMA
//
//  Created by Albertomac on 1/26/17.
//  Copyright © 2017 Albertomac. All rights reserved.
//

#include "CLNet.h"
#include "CLRandom.h"
#include <math.h>

#ifdef __APPLE__
#include </usr/local/include/clBLAS.h>
#else
#include <clBLAS.h>
#endif

#define BUFFER_STRING 64

#define BLOCK_SIZE_MEMSET 64
#define BLOCK_SIZE_ACTIVATION 32
#define BLOCK_SIZE_CHI_SQUARED 64
#define BLOCK_SIZE_JACOBIAN 32
#define BLOCK_SIZE_DELTA 64
#define BLOCK_SIZE_HESSIAN_UPDATE 32
#define BLOCK_SIZE_CHOLESKY_DECOMPOSITION 32


#pragma mark CLDeviceContext

void CLDeviceContextInit(CLDeviceContext * devContext, CLPlatform platform, CLDevice device)
{
	//platform
	devContext->platform = platform;

	//device
	devContext->device = device;

	//context
	devContext->context = CLCreateContext(devContext->platform, devContext->device);

	//queue
	devContext->queue = CLCreateQueue(devContext->context, devContext->device);

	//program
	devContext->program = CLCreateProgramWithMacro(devContext->context, devContext->device, "Kernels.ocl", (CLNetPrecisionDouble == CLTrue ? "#define CLNetDataType double" : "#define CLNetDataType float"));

	//kernelMemSet
	devContext->kernelMemSet = CLCreateKernel(devContext->program, kMemSet);

	//kernelsActivation
	devContext->kernelsActivation = calloc(nActivationFunctions, sizeof(CLKernel));
	devContext->kernelsActivation[CLActivationSigmoid] = CLCreateKernel(devContext->program, kActivationSigmoid);
	devContext->kernelsActivation[CLActivationTansig] = CLCreateKernel(devContext->program, kActivationTansig);
	devContext->kernelsActivation[CLActivationRadbas] = CLCreateKernel(devContext->program, kActivationRadbas);

	//kernelChiSquared
	devContext->kernelChiSquared = CLCreateKernel(devContext->program, kChiSquared);

	//kernelChiSquaredReduce
	devContext->kernelChiSquaredReduce = CLCreateKernel(devContext->program, kChiSquaredReduce);

	//kernelJacobian
	devContext->kernelJacobian = CLCreateKernel(devContext->program, kJacobian);

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
	CLReleaseKernel(devContext->kernelsActivation[CLActivationSigmoid], kActivationSigmoid);
	CLReleaseKernel(devContext->kernelsActivation[CLActivationTansig], kActivationTansig);
	CLReleaseKernel(devContext->kernelsActivation[CLActivationRadbas], kActivationRadbas);
	CLReleaseKernel(devContext->kernelChiSquared, kChiSquared);
	CLReleaseKernel(devContext->kernelChiSquaredReduce, kChiSquaredReduce);
	CLReleaseKernel(devContext->kernelJacobian, kJacobian);
	CLReleaseKernel(devContext->kernelDelta, kDelta);
	CLReleaseKernel(devContext->kernelUpdateDiagonal, kUpdateDiagonal);
	CLReleaseKernel(devContext->kernelCholeskyDecomposition, kCholeskyDecomposition);

	CLReleaseProgram(devContext->program, "Kernels.ocl");
	CLReleaseQueue(devContext->queue, "queue");
	CLReleaseContext(devContext->context, "context");
	CLReleaseDevice(devContext->device, "device");
}


#pragma mark CLNet

void printMatrixToFile(CLDeviceContext * devContext, CLMatrix * matrix, CLStringConst path)
{
	FILE * f = fopen(path, "w+");
	CLMatrixUpdateValuesFromMem(matrix, devContext->queue);

	for (CLUInt i = 0; i < matrix->rows; ++i) {
		for (CLUInt j = 0; j < matrix->columns; ++j) {
			fprintf(f, "%.16g\t", matrix->values[i * matrix->columns + j]);
		}
		fprintf(f, "\n");
	}
	fclose(f);
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

	CLEnqueueNDRangeKernel(devContext->queue, kernelMemSet, 1, 0, gws, lws, 0, NULL, &eventMemSet, kMemSet);
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
			   CLUInt nLayers, CLUInt * neuronsPerLayer, CLActivation * activationFunctionPerLayer,
			   CLUInt nTargets, CLNetDataType * targets,
			   CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns, CLUInt nBias)
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
	net->bias = nBias;

	//nPatterns
	net->nPatterns = nPatterns;

	//nInputs
//	net->nInputs = nInputs;
	net->nInputs = nInputs + net->bias;

	//patterns
//	net->p = calloc(net->nPatterns * net->nInputs, sizeof(*patterns));
//	memcpy(net->p, patterns, sizeof(CLNetDataType) * net->nPatterns * net->nInputs);
	net->p = calloc(net->nPatterns * net->nInputs, sizeof(*patterns));

	if (net->bias > 0) {
		for (CLUInt i = 0; i < net->nPatterns; ++i) {
			for (CLUInt j = 0; j < net->nInputs; j++) {
				net->p[i * net->nInputs + j] = (j < net->nInputs - net->bias) ? patterns[i * (net->nInputs - net->bias) + j] : 1;
			}
		}
	} else {
		memcpy(net->p, patterns, sizeof(CLNetDataType) * net->nPatterns * net->nInputs);
	}

	//nHiddenLayers
	net->nLayers = nLayers;

	//nNeuronsPerHiddenLayer
	net->neuronsPerLayer = calloc(net->nLayers, sizeof(CLUInt));
	memcpy(net->neuronsPerLayer, neuronsPerLayer, sizeof(CLUInt) * net->nLayers);

	//MARK:BIAS
	for (CLUInt i = 0; i < net->nLayers - 1; ++i) {
		net->neuronsPerLayer[i] += net->bias;
	}

	//activationFunctionPerLayer
	net->activationFunctionPerLayer = calloc(net->nLayers, sizeof(CLActivation));
	memcpy(net->activationFunctionPerLayer, activationFunctionPerLayer, sizeof(CLActivation) * net->nLayers);

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

	//jacobian
	net->jacobian = calloc(1, sizeof(CLMatrix));

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
	net->initialLambda = 0.0001;
	net->upFactor = 10.0;
	net->downFactor = 10.0;
	net->targetDeltaError = 1e-18f;
	net->finalError = 0.0;
	net->finalDeltaError = 0.0;

	//benchmark
	net->benchmark = CLFalse;
}

void CLNetInitWithFile(CLNet * net, CLStringConst fileName)
{
	//TODO: da fare dopo aver completato il progetto ed aver visto che funziona correttamente
}

void CLNetMatrixMultiply(CLNet * net, CLDeviceContext * devContext,
						 CLMatrix * matrixA, clblasTranspose transposeA,
						 CLMatrix * matrixB, clblasTranspose transposeB,
						 CLMatrix * matrixResult, CLEvent * event)
{
	//TODO: sistemare la funzione per far moltiplicare le matrici transposte
//	CLSize m = (transposeA == clblasNoTrans) ? matrixA->rows    : matrixA->columns;
//	CLSize n = (transposeB == clblasNoTrans) ? matrixB->columns : matrixB->rows;
//	CLSize k = (transposeA == clblasNoTrans) ? matrixA->columns : matrixA->rows;

	CLSize m = matrixA->rows;
	CLSize n = matrixB->columns;
	CLSize k = matrixA->columns;

//	printf("GEMM %s(%zu, %zu) X %s(%zu, %zu) = %s(%zu, %zu)\n", matrixA->name, m, k, matrixB->name, k, n, matrixResult->name, m, n);

//	clblasStatus status = GEMM(clblasRowMajor, transposeA, transposeB, m, n, k,
//							   1, matrixA->mem, matrixA->offsetMem, (transposeA == clblasNoTrans ? k : matrixA->rows),
//							   matrixB->mem, matrixB->offsetMem, (transposeB == clblasNoTrans ? n : matrixB->rows),
//							   0, matrixResult->mem, matrixResult->offsetMem, (transposeB == clblasNoTrans) ? n : matrixB->rows,
//							   1, &devContext->queue, 0, NULL, event);

	clblasStatus status = GEMM(clblasRowMajor, transposeA, transposeB, m, n, k,
							   1, matrixA->mem, matrixA->offsetMem, k,
							   matrixB->mem, matrixB->offsetMem, n,
							   0, matrixResult->mem, matrixResult->offsetMem, n,
							   1, &devContext->queue, 0, NULL, event);

	CLErrorCheck(status, "GEMM", "", CHECK_EXIT);

	CLWaitForEvent(event, "GEMM event");

#pragma mark BENCHMARK_MATRIX_MULTIPLY
	if (net->benchmark == CLTrue) {
		CLDouble time = timeBetweenEventsNS(*event, *event);
		CLDouble flops = 2 * m * n * k;

		printStat(flops, time, "matrixMultiply");
	}
}

void CLNetActivationLayer(CLNet * net, CLDeviceContext * devContext, CLMatrix * layer, CLActivation activationFunction, CLEvent * event)
{
	if (activationFunction == CLActivationLinear) {
		return;
	}

	CLSize lws[] = {BLOCK_SIZE_ACTIVATION,
					BLOCK_SIZE_ACTIVATION};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(layer->rows, lws[0]),
					CLGetOptimalGlobalWorkItemsSize(layer->columns, lws[1])};

	CLUInt nArg = 0;
	CLKernel kernelActivation = devContext->kernelsActivation[activationFunction];
	CLSetKernelArg(kernelActivation, nArg++, sizeof(layer->mem), &layer->mem, layer->name);
	CLSetKernelArg(kernelActivation, nArg++, sizeof(layer->rows), &layer->rows, "rows");
	CLSetKernelArg(kernelActivation, nArg++, sizeof(layer->columns), &layer->columns, "columns");

	CLEnqueueNDRangeKernel(devContext->queue, kernelActivation, 2, NULL, gws, lws, 0, NULL, event, layer->name);
	CLWaitForEvent(event, "eventActivation");
}

void TESTForward(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->activationPerLayer[0], "/Volumes/RamDisk/TESTForward/H0.txt");
	printMatrixToFile(devContext, net->activationPerLayer[1], "/Volumes/RamDisk/TESTForward/H1.txt");
	printMatrixToFile(devContext, net->activationPerLayer[2], "/Volumes/RamDisk/TESTForward/H2.txt");
}

void CLNetForward(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent * eventsMultiply = calloc(net->nLayers, sizeof(CLEvent));
	CLEvent * eventsActivation = calloc(net->nLayers, sizeof(CLEvent));

	CLNetMatrixMultiply(net, devContext,
						net->trainingPatterns, clblasNoTrans,
						net->weightsPerLayer[0], clblasNoTrans,
						net->activationPerLayer[0], &eventsMultiply[0]);

	CLNetActivationLayer(net, devContext, net->activationPerLayer[0], net->activationFunctionPerLayer[0], &eventsActivation[0]);

	//Hidden Layers
	for (CLUInt i = 1; i < net->nLayers; ++i) {
		CLNetMatrixMultiply(net, devContext,
							net->activationPerLayer[i - 1], clblasNoTrans,
							net->weightsPerLayer[i], clblasNoTrans,
							net->activationPerLayer[i], &eventsMultiply[i]);

		CLNetActivationLayer(net, devContext, net->activationPerLayer[i], net->activationFunctionPerLayer[i], &eventsActivation[i]);
	}

#pragma mark BENCHMARK_FORWARD
	if (net->benchmark == CLTrue) {
		CLDouble time = 0;
		CLDouble flops = 0;

		for (CLUInt i = 0; i < net->nLayers; ++i) {

			if (net->activationFunctionPerLayer[i] != CLActivationLinear) {

				time = timeBetweenEventsNS(eventsActivation[i], eventsActivation[i]);
				flops = net->activationPerLayer[i]->elements;

				switch (net->activationFunctionPerLayer[i]) {
					//1 / (1 + exp(-x))
					//1		 1  4  1   flops
					case CLActivationSigmoid:
						flops *= 7;
						break;

					//2 / (1 + exp(-2 * x)) - 1
					//  1    1  4  1  1       1
					case CLActivationTansig:
						flops *= 9;
						break;

					//exp(- x * x)
					// 4  1   1
					case CLActivationRadbas:
						flops *= 5;
						break;

					CLActivationLinear:
					default:
						break;
				}
				printStat(flops, time, "activation[]");
			}
		}
	}

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		CLReleaseEvent(eventsMultiply[i], "eventMultiply");

		if (net->activationFunctionPerLayer[i] != CLActivationLinear)
			CLReleaseEvent(eventsActivation[i], "eventActivation");
	}

	free(eventsMultiply);
	free(eventsActivation);
	eventsMultiply = NULL;
	eventsActivation = NULL;
}

void TESTErrorChiSquared(CLNet * net, CLDeviceContext * devContext)
{
	printf("errorChiSquared (expected, real): (990902.087739," CLNetDataTypeScanf ")\n", net->errorChiSquared);
}

void CLNetChiSquared(CLNet * net, CLDeviceContext * devContext)
{
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

	CLEnqueueNDRangeKernel(devContext->queue, kernelChiSquared, 1, NULL, gws, lws, 0, NULL, &eventChiSquared, kChiSquared);

	//ChiSquaredReduce
	CLEvent eventChiSquaredReduce;
	gws[0] = nwg;
	lws[0] = nwg;
	nArg = 0;
	CLKernel kernelChiSquaredReduce = devContext->kernelChiSquaredReduce;
	CLSetKernelArg(kernelChiSquaredReduce, nArg++, sizeof(net->chiSquaredError->mem), &net->chiSquaredError->mem, "partialSums");
	CLSetKernelArg(kernelChiSquaredReduce, nArg++, sizeof(CLNetDataType) * lws[0], NULL, "localSums");

	CLEnqueueNDRangeKernel(devContext->queue, kernelChiSquaredReduce, 1, NULL, gws, lws, 0, NULL, &eventChiSquaredReduce, kChiSquaredReduce);

	CLEvent eventReadErrorValue;
	CLNetDataType * errorValue = calloc(1, sizeof(CLNetDataType));
	CLInt error = clEnqueueReadBuffer(devContext->queue, net->chiSquaredError->mem, CLTrue, 0, sizeof(CLNetDataType), errorValue, 0, NULL, &eventReadErrorValue);
	CLErrorCheck(error, "clEnqueueReadBuffer", "read errorValue", CHECK_EXIT);
	CLWaitForEvent(&eventReadErrorValue, "eventReadErrorValue");
	CLReleaseEvent(eventReadErrorValue, "eventReadErrorValue");
	net->errorChiSquared = errorValue[0];
	free(errorValue);

	//
	CLWaitForEvent(&eventChiSquared, "eventChiSquared");
	CLWaitForEvent(&eventChiSquaredReduce, "eventChiSquaredReduce");

#pragma mark BENCHMARK_CHI_SQUARED
	if (net->benchmark == CLTrue) {

		CLDouble time = 0;
		CLDouble flops = 0;

		time = timeBetweenEventsNS(eventChiSquared, eventChiSquared);
		flops = 3 * net->outputs->elements;
		printStat(flops, time, "chiSquared");

		time = timeBetweenEventsNS(eventChiSquaredReduce, eventChiSquaredReduce);
		flops = 3 * nwg;
		printStat(flops, time, "chiSquaredReduce");
	}

	//Releases
	CLReleaseEvent(eventChiSquared, "eventChiSquared");
	CLReleaseEvent(eventChiSquaredReduce, "eventChiSquaredReduce");
}

void TESTJacobian(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->jacobian, "/Volumes/RamDisk/TESTForward/jMac.txt");
}

void CLNetJacobian(CLNet * net, CLDeviceContext * devContext)
{
	CLUInt offset = 0;
	CLUInt slope = net->neuronsPerLayer[0];
	CLUInt yTimes = net->nTargets;
	CLUInt nArg = 0;

	CLUInt workDim = 2;
	CLSize lws[] = {BLOCK_SIZE_JACOBIAN, BLOCK_SIZE_JACOBIAN};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->trainingPatterns->columns, lws[0]), CLGetOptimalGlobalWorkItemsSize(net->trainingPatterns->rows, lws[1])};

	CLKernel kernelJacobian = devContext->kernelJacobian;

	if (net->partialJacobianFilled == CLFalse) {

		CLEvent eventJacobianFirst;

		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->jacobian->mem), &net->jacobian->mem, net->jacobian->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->trainingPatterns->mem), &net->trainingPatterns->mem, net->trainingPatterns->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->jacobian->columns, "jacobianColumns");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->trainingPatterns->rows, "rowsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->trainingPatterns->columns, "columnsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

		CLEnqueueNDRangeKernel(devContext->queue, kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventJacobianFirst, "jacobianFirst");
		CLReleaseEvent(eventJacobianFirst, "jacobianFirst");

		net->partialJacobianFilled = CLTrue;
	}


	CLEvent * eventsJacobian = calloc(net->nLayers - 1, sizeof(CLEvent));

	offset = net->trainingPatterns->columns * slope;

	for (CLUInt i = 0; i < net->nLayers - 1; ++i) {

		slope = net->neuronsPerLayer[i + 1];

		nArg = 0;
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->jacobian->mem), &net->jacobian->mem, net->jacobian->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->activationPerLayer[i]->mem), &net->activationPerLayer[i]->mem, net->activationPerLayer[i]->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->jacobian->columns, "jacobianColumns");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->activationPerLayer[i]->rows, "rowsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->activationPerLayer[i]->columns, "columnsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

		gws[0] = CLGetOptimalGlobalWorkItemsSize(net->activationPerLayer[i]->columns, lws[0]);
		gws[1] = CLGetOptimalGlobalWorkItemsSize(net->activationPerLayer[i]->rows, lws[1]);

		CLEnqueueNDRangeKernel(devContext->queue, kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventsJacobian[i], "eventsJacobian");
		offset += net->neuronsPerLayer[i] * slope;
	}

	//
	for (CLUInt i = 0; i < net->nLayers - 1; ++i) {
		CLWaitForEvent(&eventsJacobian[i], "eventsJacobian");
	}

	for (CLUInt i = 0; i < net->nLayers - 1; ++i) {
		CLReleaseEvent(eventsJacobian[i], "eventsJacobian");
	}

	free(eventsJacobian);
	eventsJacobian = NULL;
}

void TESTHessian(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->hessian, "/Volumes/RamDisk/TESTForward/hMac.txt");
}

void CLNetHessian(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventHessian;
	CLSize m = net->jacobian->columns;
	CLSize n = net->jacobian->columns;
	CLSize k = net->jacobian->rows;

	clblasStatus status = GEMM(clblasRowMajor, clblasTrans, clblasNoTrans, m, n, k,
							   1, net->jacobian->mem, 0, m,
							   net->jacobian->mem, 0, n,
							   0, net->hessian->mem, 0, n,
							   1, &devContext->queue, 0, NULL, &eventHessian);

	CLErrorCheck(status, "GEMM", "hessian", CHECK_EXIT);

//	CLNetMatrixMultiply(net, devContext,
//						net->jacobian, clblasTrans,
//						net->jacobian, clblasNoTrans,
//						net->hessian, &eventHessian);
	CLWaitForEvent(&eventHessian, "eventHessian");

#pragma mark BENCHMARK_HESSIAN
	if (net->benchmark == CLTrue) {
		CLDouble time = timeBetweenEventsNS(eventHessian, eventHessian);
		CLDouble flops = 2 * m * n * k;

		printStat(flops, time, "hessian");
	}

	CLReleaseEvent(eventHessian, "eventHessian");
}

void TESTCalculateD(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->d, "/Volumes/RamDisk/TESTForward/dMac.txt");
}

void CLNetCalculateD(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventD;
	CLUInt nArg = 0;
	CLKernel kernelDelta = devContext->kernelDelta;
	CLSetKernelArg(kernelDelta, nArg++, sizeof(net->d->mem), &net->d->mem, net->d->name);
	CLSetKernelArg(kernelDelta, nArg++, sizeof(net->trainingTargets->mem), &net->trainingTargets->mem, net->trainingTargets->name);
	CLSetKernelArg(kernelDelta, nArg++, sizeof(net->outputs->mem), &net->outputs->mem, net->outputs->name);
	CLSetKernelArg(kernelDelta, nArg++, sizeof(net->jacobian->mem), &net->jacobian->mem, net->jacobian->name);
	CLSetKernelArg(kernelDelta, nArg++, sizeof(CLUInt), &net->trainingTargets->elements, "ny");
	CLSetKernelArg(kernelDelta, nArg++, sizeof(CLUInt), &net->nWeights, "npar");

	CLSize lws[] = {BLOCK_SIZE_DELTA};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->d->columns, BLOCK_SIZE_DELTA)};

	CLEnqueueNDRangeKernel(devContext->queue, kernelDelta, 1, NULL, gws, lws, 0, NULL, &eventD, kDelta);

	//
	CLWaitForEvent(&eventD, "eventD");

#pragma mark BENCHMARK_CALCULATE_D
	if (net->benchmark == CLTrue) {
		CLDouble time = timeBetweenEventsNS(eventD, eventD);
		CLDouble flops = 3 * (net->nWeights * net->nPatterns);

		printStat(flops, time, "calculateD");
	}

	CLReleaseEvent(eventD, "eventDelta");
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

	CLEnqueueNDRangeKernel(devContext->queue, kernelUpdateDiagonal, 1, NULL, gws, lws, 0, NULL, &eventUpdateDiagonal, "updateHessianDiagonal");
	CLWaitForEvent(&eventUpdateDiagonal, "eventUpdateDiagonal");

#pragma BENCHMARK_UPDATE_HESSIAN_DIAGONAL
	//TODO: inserire il benchmark

	CLReleaseEvent(eventUpdateDiagonal, "eventUpdateDiagonal");
}

void TESTCholeskyDecomposition(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->cholesky, "/Volumes/RamDisk/TESTForward/cholMac.txt");
	printf("ill: %d\n", net->ill);
}

void CLNetCholeskyDecomposition(CLNet * net, CLDeviceContext * devContext)
{
	CLNetMemSetMatrix(net, devContext, net->choleskySums, 0);

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

		CLEnqueueNDRangeKernel(devContext->queue, kernelCholeskyDecomposition, 1, NULL, gws, lws, 0, NULL, eventCholeskyDecomposition+i, kCholeskyDecomposition);
	}

	CLEvent eventReadIll;
	CLNetDataType * illResult = calloc(1, sizeof(CLNetDataType));
	CLInt error = clEnqueueReadBuffer(devContext->queue, net->illMem, CLTrue, 0, sizeof(CLNetDataType), illResult, 0, NULL, &eventReadIll);
	CLErrorCheck(error, "clEnqueueReadBuffer", "read errorValue", CHECK_EXIT);
	CLWaitForEvent(&eventReadIll, "eventReadErrorValue");
	CLReleaseEvent(eventReadIll, "eventReadErrorValue");
	net->ill = illResult[0];
	free(illResult);
	illResult = NULL;

	//
	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {
		CLWaitForEvent(&eventCholeskyDecomposition[i], "eventCholeskyDecomposition");
	}

	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {
		CLReleaseEvent(eventCholeskyDecomposition[i], "eventCholeskyDecomposition");
	}

	free(eventCholeskyDecomposition);
	eventCholeskyDecomposition = NULL;
}


void CLNetSolveTriangular(CLNet * net, CLDeviceContext * devContext, CLMatrix * cholesky, CLMatrix * delta, clblasTranspose uplo, CLEvent * eventSolve)
{
	clblasStatus status = TRSV(clblasRowMajor, clblasUpper, uplo, clblasNonUnit,
							   cholesky->rows, cholesky->mem, 0,
							   cholesky->rows, delta->mem, 0, 1,
							   1, &devContext->queue, 0, NULL, eventSolve);

	CLErrorCheck(status, "TRSV", "", CHECK_EXIT);
}

void CLNetCholeskySolve(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventCopyDelta;
	CLInt status = clEnqueueCopyBuffer(devContext->queue, net->d->mem, net->delta->mem, 0, 0, net->d->size, 0, NULL, &eventCopyDelta);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "copyDelta", CHECK_EXIT);

	CLEvent eventsSolve[2];
	CLNetSolveTriangular(net, devContext, net->cholesky, net->delta, clblasTrans, &eventsSolve[0]);
	CLNetSolveTriangular(net, devContext, net->cholesky, net->delta, clblasNoTrans, &eventsSolve[1]);

	//
	CLWaitForEvent(&eventCopyDelta, "eventCopyDelta");
	CLWaitForEvent(&eventsSolve[0], "eventSolve[0]");
	CLWaitForEvent(&eventsSolve[1], "eventSolve[1]");

	CLReleaseEvent(eventCopyDelta, "eventCopyDelta");
	CLReleaseEvent(eventsSolve[0], "eventSolve[0]");
	CLReleaseEvent(eventsSolve[1], "eventSolve[1]");
}

void CLNetReloadWeights(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventReloadWeights;
	CLInt status = clEnqueueCopyBuffer(devContext->queue, net->weightsTemp->mem, net->weights->mem, 0, 0, net->weightsTemp->size, 0, NULL, &eventReloadWeights);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);

	CLWaitForEvent(&eventReloadWeights, "eventReloadWeights");

	CLReleaseEvent(eventReloadWeights, "eventReloadWeights");
}

void CLNetUpdateWeightsTemp(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeightsTemp;
	CLInt status = clEnqueueCopyBuffer(devContext->queue, net->weights->mem, net->weightsTemp->mem, 0, 0, net->weights->size, 0, NULL, &eventUpdateWeightsTemp);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weights -> weightsTemp", CHECK_EXIT);

	CLWaitForEvent(&eventUpdateWeightsTemp, "eventReloadWeights");

	CLReleaseEvent(eventUpdateWeightsTemp, "eventUpdateWeightsTemp");
}

void CLNetUpdateWeightsWithDelta(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeights;
	clblasStatus status = AXPY(net->weights->elements, 1, net->delta->mem, 0, 1, net->weights->mem, 0, 1, 1, &devContext->queue, 0, NULL, &eventUpdateWeights);
	CLErrorCheck(status, "AXPY", "updateWeightsWithDelta", CHECK_EXIT);

	CLWaitForEvent(&eventUpdateWeights, "eventReloadWeights");

	CLReleaseEvent(eventUpdateWeights, "eventUpdateWeights");
}

void CLNetTrainLMA(CLNet * net, CLDeviceContext * devContext)
{
	CLNetDataType mult;
	CLNetDataType lambda = net->initialLambda;
	CLNetDataType error = -1.0f;
	CLNetDataType newError = -1.0f;
	CLNetDataType deltaError = -1.0f;
	CLBool ill;

	CLNetForward(net, devContext);									//Forward per calcolare l'errore iniziale
	CLNetChiSquared(net, devContext);								//Calcolo dell'errore iniziale
	error = net->errorChiSquared;

//	printMatrixToFile(devContext, net->trainingPatterns, "/Volumes/RamDisk/patterns.txt");
//	printMatrixToFile(devContext, net->activationPerLayer[0], "/Volumes/RamDisk/H0.txt");
//	printMatrixToFile(devContext, net->activationPerLayer[1], "/Volumes/RamDisk/H1.txt");
//	printMatrixToFile(devContext, net->activationPerLayer[2], "/Volumes/RamDisk/H2.txt");
//	printMatrixToFile(devContext, net->weights, "/Volumes/RamDisk/weights.txt");
//	printMatrixToFile(devContext, net->trainingTargets, "/Volumes/RamDisk/targets.txt");

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

			CLNetUpdateHessianDiagonal(net, devContext, mult);	//Aggiorno la diagonale dell'hessian
			CLNetCholeskyDecomposition(net, devContext);		//Calcolo della decomposizione di Cholesky
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

			printf("it = %4d,   lambda = %10g,   err = %10g,   derr = %10g\n", i, lambda, error, deltaError);

			if (isnan(newError) || lambda > 1e20) return;

			if (ill) {												//Se ill è ancora 1, vengono aggiornati i moltiplicatori
				mult = (1 + lambda * net->upFactor)/(1 + lambda);
				lambda *= net->upFactor;
				i++;
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
	CLMatrixUpdateValuesFromMem(net->outputs, devContext->queue);

	if (net->nInputs < 3){
		for (CLUInt i = 0; i < net->nInputs - net->bias; ++i) {
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
			for (CLUInt i = 0; i < net->nInputs - net->bias; ++i) {
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

	//activationPerLayer
	CLString activationPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(activationPerLayerName, BUFFER_STRING - 1, "activationPerLayer[%d]", i);
		CLMatrixInit(net->activationPerLayer[i], net->nTrainingPatterns, net->neuronsPerLayer[i], activationPerLayerName);
		CLMatrixCreateMem(net->activationPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(activationPerLayerName);
	activationPerLayerName = NULL;

	//chiSquaredError
	CLSize chiSquaredErrorColumns = divUpSize(CLGetOptimalGlobalWorkItemsSize(net->outputs->elements, BLOCK_SIZE_CHI_SQUARED), BLOCK_SIZE_CHI_SQUARED);
	CLMatrixInit(net->chiSquaredError, 1, (CLUInt)chiSquaredErrorColumns, "chiSquaredError");
	CLMatrixCreateMem(net->chiSquaredError, devContext->context, CL_MEM_READ_WRITE);

	//testTargets
	CLMatrixInit(net->testTargets, net->nTestPatterns, net->nTargets, "testTargets");
	memcpy(net->testTargets->values, net->t, net->testTargets->size);
	CLMatrixCreateMemHostVar(net->testTargets, devContext->context, CL_MEM_READ_ONLY);

	//trainingPatterns
	CLMatrixInit(net->trainingTargets, net->nTrainingPatterns, net->nTargets, "patterns");
	memcpy(net->trainingTargets->values, net->t + net->testTargets->elements, net->trainingTargets->size);
	CLMatrixCreateMemHostVar(net->trainingTargets, devContext->context, CL_MEM_READ_ONLY);

	//jacobian
	CLMatrixInit(net->jacobian, net->nTrainingPatterns * net->nTargets, net->nWeights, "jacobian");
	CLMatrixCreateMem(net->jacobian, devContext->context, CL_MEM_READ_WRITE);

	//hessian
	CLMatrixInit(net->hessian, net->nWeights, net->nWeights, "hessian");
	CLMatrixCreateMem(net->hessian, devContext->context, CL_MEM_READ_WRITE);

	//d
	CLMatrixInit(net->d, 1, net->nWeights, "d");
	CLMatrixCreateMem(net->d, devContext->context, CL_MEM_READ_WRITE);

	//delta
	CLMatrixInit(net->delta, 1, net->nWeights, "delta");
	CLMatrixCreateMem(net->delta, devContext->context, CL_MEM_READ_WRITE);

	//cholesky
	CLMatrixInit(net->cholesky, net->nWeights, net->nWeights, "cholesky");
	CLMatrixCreateMem(net->cholesky, devContext->context, CL_MEM_READ_WRITE);

	//choleskySums
	CLMatrixInit(net->choleskySums, 1, net->nWeights, "choleskySums");
	CLMatrixCreateMem(net->choleskySums, devContext->context, CL_MEM_READ_WRITE);

	//ill & illMem
	net->ill = CLFalse;
	net->illMem = CLCreateBuffer(devContext->context, CL_MEM_READ_WRITE, sizeof(CLNetDataType), "illMem");

	clblasSetup();

//	CLNetForward(net, devContext);
//	TESTForward(net, devContext);
//
//	CLNetChiSquared(net, devContext);
//	TESTErrorChiSquared(net, devContext);
//
//	CLNetJacobian(net, devContext);
//	TESTJacobian(net, devContext);
//
//	CLNetHessian(net, devContext);
//	TESTHessian(net, devContext);
//
//	CLNetCalculateD(net, devContext);
//	TESTCalculateD(net, devContext);
//
//	CLNetCholeskyDecomposition(net, devContext, 1);
//	TESTCholeskyDecomposition(net, devContext);
	CLNetTrainLMA(net, devContext);

	CLNetPrintForward(net, devContext);

//	printMatrixToFile(devContext, net->trainingPatterns, "/Volumes/RamDisk/patterns.txt");
//	printMatrixToFile(devContext, net->trainingTargets, "/Volumes/RamDisk/targets.txt");
//	printMatrixToFile(devContext, net->outputs, "/Volumes/RamDisk/outputs.txt");
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
		CLMatrixRelease(net->activationPerLayer[i]);
	}

	net->outputs = NULL; //Remove pointer to last CLMatrix in activationPerLayer

	CLMatrixRelease(net->testTargets);
	CLMatrixRelease(net->trainingTargets);

	CLMatrixRelease(net->chiSquaredError);
	CLMatrixRelease(net->jacobian);
	CLMatrixRelease(net->hessian);
	CLMatrixRelease(net->d);
	CLMatrixRelease(net->delta);
	CLMatrixRelease(net->cholesky);
	CLMatrixRelease(net->choleskySums);

	CLReleaseMemObject(net->illMem, "illMem");

	free(net);
}