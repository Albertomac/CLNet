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

#define BLOCK_SIZE_ACTIVATION 32
#define BLOCK_SIZE_CHI_SQUARED 64
#define BLOCK_SIZE_JACOBIAN 32
#define BLOCK_SIZE_DELTA 64
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

	//kernelCholeskyDecomposition
	devContext->kernelCholeskyDecomposition = CLCreateKernel(devContext->program, kCholeskyDecomposition);
}

void CLDeviceContextRelease(CLDeviceContext * devContext)
{
	CLReleaseKernel(devContext->kernelMemSet, kMemSet);
	CLReleaseKernel(devContext->kernelsActivation[CLActivationSigmoid], kActivationSigmoid);
	CLReleaseKernel(devContext->kernelsActivation[CLActivationTansig], kActivationTansig);
	CLReleaseKernel(devContext->kernelsActivation[CLActivationRadbas], kActivationRadbas);
	CLReleaseKernel(devContext->kernelChiSquared, kChiSquared);
	CLReleaseKernel(devContext->kernelChiSquaredReduce, kChiSquaredReduce);
	CLReleaseKernel(devContext->kernelJacobian, kJacobian);
	CLReleaseKernel(devContext->kernelDelta, kDelta);
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

void CLNetInit(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * patterns,
			   CLUInt nLayers, CLUInt * neuronsPerLayer, CLActivation * activationFunctionPerLayer,
			   CLUInt nTargets, CLNetDataType * targets,
			   CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns)
{
	if (neuronsPerLayer[nLayers - 1]  != nTargets) {
		printf("nTargets must be the same of last value of neuronsPerLayer");
		exit(-1);
	}

	//nPatterns
	net->nPatterns = nPatterns;

	//nInputs
	net->nInputs = nInputs;

	//patterns
	net->p = calloc(net->nPatterns * net->nInputs, sizeof(*patterns));
	memcpy(net->p, patterns, sizeof(CLNetDataType) * net->nPatterns * net->nInputs);

	//nHiddenLayers
	net->nLayers = nLayers;

	//nNeuronsPerHiddenLayer
	net->neuronsPerLayer = calloc(net->nLayers, sizeof(*neuronsPerLayer));
	memcpy(net->neuronsPerLayer, neuronsPerLayer, sizeof(CLUInt) * net->nLayers);

	//activationFunctionPerLayer
	net->activationFunctionPerLayer = calloc(net->nLayers, sizeof(*activationFunctionPerLayer));
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

	//trainingPatterns TODO: verificare che i calloc abbiano bisogno solo del sizeof del puntatore di CLMatrix
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

	//Levenberg-Marquartd stuff
	net->ill = CLFalse;
	net->verbose = CLTrue;
	net->maxIterations = 10000;
	net->initialLambda = 0.0001;
	net->upFactor = 10.0f;
	net->downFactor = 10.0f;
	net->targetDeltaError = 1e-12f;
	net->finalError = 0.0f;
	net->finalDeltaError = 0.0f;
}

void CLNetInitWithFile(CLNet * net, CLStringConst fileName)
{
	//TODO: da fare dopo aver completato il progetto ed aver visto che funziona correttamente
}

void CLNetMatrixMultiply(CLDeviceContext * devContext, CLMatrix * matrixA, CLMatrix * matrixB, CLMatrix * matrixResult, CLEvent * event)
{
	CLSize m = matrixA->rows;
	CLSize n = matrixB->columns;
	CLSize k = matrixA->columns;


	clblasStatus status = GEMM(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k,
							   1, matrixA->mem, matrixA->offsetMem, k,
							   matrixB->mem, matrixB->offsetMem, n,
							   0, matrixResult->mem, matrixResult->offsetMem, n,
							   1, &devContext->queue, 0, NULL, event);

	CLErrorCheck(status, "GEMM", "", CHECK_EXIT);
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

	CLNetMatrixMultiply(devContext, net->trainingPatterns, net->weightsPerLayer[0], net->activationPerLayer[0], &eventsMultiply[0]);
	CLNetActivationLayer(net, devContext, net->activationPerLayer[0], net->activationFunctionPerLayer[0], &eventsActivation[0]);

	//Hidden Layers
	for (CLUInt i = 1; i < net->nLayers; ++i) {
		CLNetMatrixMultiply(devContext, net->activationPerLayer[i - 1], net->weightsPerLayer[i], net->activationPerLayer[i], &eventsMultiply[i]);
		CLNetActivationLayer(net, devContext, net->activationPerLayer[i], net->activationFunctionPerLayer[i], &eventsActivation[i]);
	}

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		CLReleaseEvent(eventsMultiply[i], "eventMultiply");
		CLReleaseEvent(eventsActivation[i], "eventActivation");
	}
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

	CLNetDataType * errorValue = CLEnqueueReadBuffer(devContext->queue, net->chiSquaredError->mem, sizeof(CLNetDataType), "errorChiSquared");
	net->errorChiSquared = errorValue[0];
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

	CLEvent * eventsJacobian = calloc(net->nLayers, sizeof(CLEvent));
	CLKernel kernelJacobian = devContext->kernelJacobian;

	if (net->partialJacobianFilled == CLFalse) {

		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->jacobian->mem), &net->jacobian->mem, net->jacobian->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(net->trainingPatterns->mem), &net->trainingPatterns->mem, net->trainingPatterns->name);
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->jacobian->columns, "jacobianColumns");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->trainingPatterns->rows, "rowsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &net->trainingPatterns->columns, "columnsI");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &offset, "offset");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &slope, "slope");
		CLSetKernelArg(kernelJacobian, nArg++, sizeof(CLUInt), &yTimes, "yTimes");

		CLEnqueueNDRangeKernel(devContext->queue, kernelJacobian, workDim, NULL, gws, lws, 0, NULL, &eventsJacobian[0], "jacobian");

		net->partialJacobianFilled = CLTrue;
	}

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

	for (CLUInt i = 0; i < net->nLayers; ++i) {
		CLReleaseEvent(eventsJacobian[i], "eventsJacobian");
	}
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
	CLSetKernelArg(kernelDelta, nArg++, sizeof(CLUInt), &net->weights->elements, "npar");

	CLSize lws[] = {BLOCK_SIZE_DELTA};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->d->columns, BLOCK_SIZE_DELTA)};

	CLEnqueueNDRangeKernel(devContext->queue, kernelDelta, 1, NULL, gws, lws, 0, NULL, &eventD, kDelta);

	CLReleaseEvent(eventD, "eventDelta");
}

void TESTCholeskyDecomposition(CLNet * net, CLDeviceContext * devContext)
{
	printMatrixToFile(devContext, net->cholesky, "/Volumes/RamDisk/TESTForward/cholMac.txt");
	printf("ill: %d\n", net->ill);
}

void CLNetCholeskyDecomposition(CLNet * net, CLDeviceContext * devContext, CLNetDataType mult)
{
	CLSize lws[] = {BLOCK_SIZE_CHOLESKY_DECOMPOSITION};
	CLSize gws[] = {CLGetOptimalGlobalWorkItemsSize(net->nWeights, lws[0])};

	//TODO: da inserire nella struct CLNet per evitare di ricrearlo ogni volta
	CLMem sums = CLCreateBuffer(devContext->context, CL_MEM_READ_WRITE, sizeof(CLNetDataType) * net->cholesky->columns, "sums");

	CLUInt nArg = 0;
	CLKernel kernelCholeskyDecomposition = devContext->kernelCholeskyDecomposition;
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->cholesky->mem), &net->cholesky->mem, net->cholesky->name);
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(CLNetDataType), &mult, "alpha");
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->hessian->mem), &net->hessian->mem, net->hessian->name);
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(sums), &sums, "sums");
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(CLUInt), &net->nWeights, "npar");
	CLSetKernelArg(kernelCholeskyDecomposition, nArg++, sizeof(net->illMem), &net->illMem, "ill");


	CLEvent * eventCholeskyDecomposition = malloc(sizeof(CLEvent) * net->cholesky->rows);
	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {

		CLSetKernelArg(kernelCholeskyDecomposition, nArg, sizeof(CLUInt), &i, "row");

		CLEnqueueNDRangeKernel(devContext->queue, kernelCholeskyDecomposition, 1, NULL, gws, lws, 0, NULL, eventCholeskyDecomposition+i, kCholeskyDecomposition);
	}

	CLUInt * illResult = CLEnqueueReadBuffer(devContext->queue, net->illMem, sizeof(CLUInt), "illMem");
	net->ill = illResult[0];

	for (CLUInt i = 0; i < net->cholesky->rows; ++i) {
		CLReleaseEvent(eventCholeskyDecomposition[i], "eventCholeskyDecomposition");
	}

	CLReleaseMemObject(sums, "sums");
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
}

void CLNetReloadWeights(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventReloadWeights;
	CLInt status = clEnqueueCopyBuffer(devContext->queue, net->weightsTemp->mem, net->weights->mem, 0, 0, net->weightsTemp->size, 0, NULL, &eventReloadWeights);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weightsTemp -> weights", CHECK_EXIT);
}

void CLNetUpdateWeightsTemp(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeightsTemp;
	CLInt status = clEnqueueCopyBuffer(devContext->queue, net->weights->mem, net->weightsTemp->mem, 0, 0, net->weights->size, 0, NULL, &eventUpdateWeightsTemp);
	CLErrorCheck(status, "clEnqueueCopyBuffer", "weights -> weightsTemp", CHECK_EXIT);
}

void CLNetUpdateWeightsWithDelta(CLNet * net, CLDeviceContext * devContext)
{
	CLEvent eventUpdateWeights;
	clblasStatus status = AXPY(net->weights->elements, 1, net->delta->mem, 0, 1, net->weights->mem, 0, 1, 1, &devContext->queue, 0, NULL, &eventUpdateWeights);
	CLErrorCheck(status, "AXPY", "updateWeightsWithDelta", CHECK_EXIT);
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

	for (CLUInt i = 0; i < net->maxIterations; ++i) {

		CLNetForward(net, devContext);								//Forward all'inizio dell'iterazione per ricalcolare le matrici Jacobian e Hessian
		CLNetJacobian(net, devContext);								//Calcolo matrice Jacobian
		CLNetHessian(net, devContext);								//Calcolo matrice Hessian
		CLNetCalculateD(net, devContext);							//Calcolo array D

		mult = 1 + lambda;											//Aggiornamento moltiplicatore
		ill = 1;													//ill settato a 1 per entrare almeno una volta nel while
		//Non si può sostituire con il do/while per via della seconda condizione
		while (ill && (i < net->maxIterations)) {

			CLNetCholeskyDecomposition(net, devContext, mult);		//Calcolo della decomposizione di Cholesky e la diagonale di hessian viene moltiplicata per mult
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

			if (isnan(newError) || lambda > 1e10) return;

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

	for (CLUInt i = 0; i < net->nInputs; ++i) {
		printf("Inputs[%2d]|", i);
	}
	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf("Target[%2d]|", i);
	}
	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf("Output[%2d]|", i);
	}
	for (CLUInt i = 0; i < net->nTargets; ++i) {
		printf(" Error%%[%2d] |", i);
	}
		printf("\n");

	CLFloat * errorPerc = malloc(sizeof(CLFloat) * net->nTargets);

	for (CLUInt p = 0; p < net->nPatterns; ++p) {

		for (CLUInt i = 0; i < net->nInputs; ++i) {
			printf("%10g|", net->trainingPatterns->values[p * net->nInputs + i]);
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			errorPerc[o] = net->trainingTargets->values[p * net->nTargets + o];
			printf("%10g|", errorPerc[o]);
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			CLFloat value = net->outputs->values[p * net->nTargets + o];
			errorPerc[o] -= value;
			printf("%10g|", value);
		}

		for (CLUInt o = 0; o < net->nTargets; ++o) {
			CLFloat errorPercValue = fabs(errorPerc[o]) * 100;
			printf( (errorPercValue > 30.0f ? "%10g♥️|" : "%10g💚|"), errorPercValue);
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
		offset += net->weightsPerLayer[i - 1]->elements;
	}
	free(weightsPerLayerName);

	//activationPerLayer
	CLString activationPerLayerName = calloc(BUFFER_STRING, sizeof(CLChar));
	for (CLUInt i = 0; i < net->nLayers; ++i) {
		snprintf(activationPerLayerName, BUFFER_STRING - 1, "activationPerLayer[%d]", i);
		CLMatrixInit(net->activationPerLayer[i], net->nTrainingPatterns, net->neuronsPerLayer[i], activationPerLayerName);
		CLMatrixCreateMem(net->activationPerLayer[i], devContext->context, CL_MEM_READ_WRITE);
	}
	free(activationPerLayerName);

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
}

void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * inputs)
{

}

void CLNetRelease(CLNet * net)
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

	CLReleaseMemObject(net->illMem, "illMem");

	free(net);
	net = NULL;
}