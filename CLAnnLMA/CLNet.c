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

	//trainingPatterns
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

	//targets
	net->targets = calloc(1, sizeof(CLMatrix));

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

	clblasStatus status;

	if (CLNetPrecisionDouble == CLTrue) {

		status = clblasDgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k,
							 1, matrixA->mem, matrixA->offsetMem, k,
							 matrixB->mem, matrixB->offsetMem, n,
							 0, matrixResult->mem, matrixResult->offsetMem, n,
							 1, &devContext->queue, 0, NULL, event);
	} else {
		status = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, m, n, k,
							 1, matrixA->mem, matrixA->offsetMem, k,
							 matrixB->mem, matrixB->offsetMem, n,
							 0, matrixResult->mem, matrixResult->offsetMem, n,
							 1, &devContext->queue, 0, NULL, event);
	}

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

	CLWaitForEvent(event, layer->name);
}

//void TESTNetActivation(CLNet * net, CLDeviceContext * devContext, CLMatrix * layer, CLActivation activationFunction)
//{
//	CLNetDataType * values = calloc(layer->elements, sizeof(CLNetDataType));
//	memcpy(values, layer->values, layer->size);
//
//	CLMatrixUpdateValuesFromMem(layer, devContext->queue);
//
//	for (CLUInt i = 0; i < layer->rows; ++i) {
//		for (CLUInt j = 0; j < layer->columns; ++j) {
//
//			CLNetDataType value = values[i * layer->columns + j];
//
//			switch (activationFunction) {
//				case CLActivationSigmoid:
//					value = 1 / (1 + exp( -value));
//					break;
//				case CLActivationTansig:
//					value = 2 / (1 + exp(-2 * value)) - 1;
//					break;
//				case CLActivationRadbas:
//					value = exp(-(value * value));
//					break;
//				case CLActivationLinear:
//				default:
//					break;
//			}
//
//			values[i * layer->columns + j] = value;
//		}
//	}
//
//	for (CLUInt i = 0; i < layer->rows; ++i) {
//		for (CLUInt j = 0; j < layer->columns; ++j) {
//
//			CLUInt index = i * layer->columns + j;
//			CLNetDataType layerValue = layer->values[index];
//			CLNetDataType value = values[index];
//			if (fabs(layerValue - value) > CLFTollerance) {
//				fprintf(stderr, "ACTIVATION ERROR: (layerValue, value)[%d][%d] = (%g, %g)\n", i, j, layerValue, value);
//			}
//		}
//	}
//}
//
//
//void TESTNetForward(CLNet * net, CLDeviceContext * devContext)
//{
//	CLMatrixMultiply(net->trainingPatterns, net->weightsPerLayer[0], net->activationPerLayer[0], CLMatrixNoTrans, CLMatrixNoTrans);
//	TESTNetActivation(net, devContext, net->activationPerLayer[0], net->activationFunctionPerLayer[0]);
//
//	for (CLUInt i = 1; i < net->nLayers; ++i) {
//		CLMatrixMultiply(net->activationPerLayer[i - 1], net->weightsPerLayer[i], net->activationPerLayer[i], CLMatrixNoTrans, CLMatrixNoTrans);
//		TESTNetActivation(net, devContext, net->activationPerLayer[i], net->activationFunctionPerLayer[i]);
//	}
//}

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


	FILE * fH0 = fopen("/Volumes/RamDisk/H0.txt", "w+");
	CLMatrix * H0 = net->activationPerLayer[0];

	CLMatrixUpdateValuesFromMem(H0, devContext->queue);

	for (CLUInt i = 0; i < H0->elements; ++i) {
		fprintf(fH0, "%.16g\t", H0->values[i]);
	}
	fclose(fH0);


	FILE * fH1 = fopen("/Volumes/RamDisk/H1.txt", "w+");
	CLMatrix * h1 = net->activationPerLayer[1];

	CLMatrixUpdateValuesFromMem(h1, devContext->queue);

	for (CLUInt i = 0; i < h1->elements; ++i) {
		fprintf(fH1, "%.16g\t", h1->values[i]);
	}
	fclose(fH1);



	FILE * fH2 = fopen("/Volumes/RamDisk/H2.txt", "w+");
	CLMatrix * H2 = net->activationPerLayer[2];

	CLMatrixUpdateValuesFromMem(H2, devContext->queue);

	for (CLUInt i = 0; i < H2->elements; ++i) {
		fprintf(fH2, "%.16g\t", H2->values[i]);
	}
	fclose(fH2);

}

void CLNetTrainWithDeviceContext(CLNet * net, CLDeviceContext * devContext)
{
	net->partialJacobianFilled = CLFalse;

	//testPatterns
	CLMatrixInit(net->testPatterns, net->nTestPatterns, net->nInputs, "testPatterns");
	memcpy(net->testPatterns->values, net->p, net->testPatterns->size);
	CLMatrixCreateMemHostVar(net->testPatterns, devContext->context, CL_MEM_READ_ONLY);

	//trainingPatterns
	CLMatrixInit(net->trainingPatterns, net->nPatterns - net->nTestPatterns, net->nInputs, "patterns");
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

	clblasSetup();

	CLNetForward(net, devContext);
//	TESTNetForward(net, devContext);

//	FILE * fForward = fopen("/Volumes/RamDisk/forward.txt", "w+");
//	CLMatrix * forwardMatrix = net->activationPerLayer[net->nLayers - 1];
//
//	CLMatrixUpdateValuesFromMem(forwardMatrix, devContext->queue);
//
//	for (CLUInt i = 0; i < forwardMatrix->elements; ++i) {
//		fprintf(fForward, "%.16g\t", forwardMatrix->values[i]);
//	}
//	fclose(fForward);
}

void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * inputs)
{

}

void CLNetRelease(CLNet * net)
{
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

	CLMatrixRelease(net->targets);

	CLMatrixRelease(net->jacobian);
	CLMatrixRelease(net->hessian);
	CLMatrixRelease(net->d);
	CLMatrixRelease(net->delta);
	CLMatrixRelease(net->cholesky);

	CLReleaseMemObject(net->illMem, "illMem");

	free(net);
	net = NULL;
}