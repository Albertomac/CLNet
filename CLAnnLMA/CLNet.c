////
////  CLNet.c
////  CLAnnLMA
////
////  Created by Albertomac on 1/26/17.
////  Copyright © 2017 Albertomac. All rights reserved.
////
//
//#include "CLNet.h"
//
//#include "CLRandom.h"
//
//#define BUFFER_STRING 64
//
//void swapRow(CLFloat * matrix, CLUInt rows, CLUInt columns, CLUInt fromRow, CLUInt toRow)
//{
//	for (CLUInt i = 0; i < columns; ++i) {
//		CLFloat tmp = matrix[fromRow * columns + i];
//		matrix[fromRow * columns + i] = matrix[toRow * columns + i];
//		matrix[toRow * columns + i] = tmp;
//	}
//}
//
//void shufflePatterns(CLFloat * inputs, CLFloat * targets, CLUInt nPatterns, CLUInt nInputs, CLUInt nTargets)
//{
//	if (nPatterns > 1) {
//		for (CLUInt i = nPatterns - 1; i > 0; --i) {
//			CLUInt j = CLRandomValue() * (i + 1);
//			swapRow(inputs, nPatterns, nInputs, i, j);
//			swapRow(targets, nPatterns, nTargets, i, j);
//		}
//	}
//}
//
//void CLNetInit(CLNet * net, CLUInt nTotalPatterns, CLUInt nInputs, CLFloat * inputs, CLUInt nHiddenLayers, CLUInt * nNeuronsPerHiddenLayer, CLActivation * activationFunctionPerLayer, CLFloat * weights, CLUInt nTargets, CLFloat * targets, CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns)
//{
//	net->nNeuronsPerHiddenLayer = malloc(sizeof(CLUInt) * nHiddenLayers);
//	net->activationFunctionPerLayer = malloc(sizeof(CLActivation) * nHiddenLayers);
//	net->name = malloc(sizeof(CLChar) * BUFFER_STRING);
//
//
//
//	net->nTotalPatterns = nTotalPatterns;
//	net->nTestPatterns = nTestPatterns;
//	net->nTrainingPatterns = net->nTotalPatterns - net->nTestPatterns;
//
//	net->nInputs = nInputs;
//	net->nHiddenLayers = nHiddenLayers;
//	memcpy(net->nNeuronsPerHiddenLayer, nNeuronsPerHiddenLayer, sizeof(CLUInt) * net->nHiddenLayers);
//	memcpy(net->activationFunctionPerLayer, activationFunctionPerLayer, sizeof(CLActivation) * net->nHiddenLayers);
//	net->nTargets = nTargets;
//	snprintf(net->name, sizeof(CLChar) * BUFFER_STRING - 1, "%s", name);
//
//
//	net->testLayer = malloc(sizeof(CLMatrix));
//	net->inputLayer = malloc(sizeof(CLMatrix));
//	net->hiddenLayer = malloc(sizeof(CLMatrix));
//	net->outputLayer = malloc(sizeof(CLMatrix));
//	net->targetLayer = malloc(sizeof(CLMatrix));
//
//	//inputLayer + hiddenLayers + outputLayer
//	net->nLayers = 1 + net->nHiddenLayers + 1;
//	net->layers = malloc(sizeof(CLMatrix) * net->nLayers);
////	for (CLUInt i = 0; i < net->nLayers; ++i) {
////		net->layers[i] = malloc(sizeof(CLNetLayer));
////	}
//	net->jacobian = malloc(sizeof(CLMatrix));
//	net->hessian = malloc(sizeof(CLMatrix));
//	net->delta = malloc(sizeof(CLMatrix));
//	net->cholesky = malloc(sizeof(CLMatrix));
//
//
//	if (shufflePattners == CLTrue) {
//		shufflePatterns(inputs, targets, net->nTotalPatterns, net->nInputs, net->nTargets);
//	}
//
//	CLMatrixInit(net->testLayer, net->nTestPatterns, net->nInputs, "testLayer");
//	CLMatrixUpdateValues(net->testLayer, inputs);
//
//	CLMatrixInit(net->inputLayer, net->nTrainingPatterns, net->nInputs, "inputLayer");
//	CLMatrixUpdateValues(net->inputLayer, inputs + (net->testLayer->elements));
//
//	CLMatrixInit(net->targetLayer, net->nTrainingPatterns, net->nTargets, "targetLayer");
//
//	CLMatrixInit(net->weights, <#cl_uint rows#>, <#cl_uint columns#>, <#const char *name#>)
//	if (weights != NULL) {
//
//	}
//}
//
//void CLNetInitWithFile(CLNet * net, CLStringConst fileName);
//
//void CLNetTrainWithDeviceContext(CLNet * net, CLDeviceContext * devContext);
//
//void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLFloat * inputs);
//
//void CLNetRelease(CLNet * net);