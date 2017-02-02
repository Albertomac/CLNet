//
//  CLAnn.h
//  CLAnnLMA
//
//  Created by Albertomac on 11/6/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef CLAnn_h
#define CLAnn_h

#include <stdio.h>
#include "CLManager.h"
#include "CLMatrix.h"

#define CLType CLDouble

/*
 TODO:

 fare in modo che funzioni sia con i CLFloat che con i CLDouble

 benchmarks file with useful runs
 improve opencl kernels

 Aggiungere la possibilità di stabilire la % di training set e la % di test

 */

typedef enum CLActivation_ {
	CLActivationSigmoid = 0,
	CLActivationTansig,
	CLActivationRadbas,
	CLActivationLinear = 100
} CLActivation;

typedef struct {

	//OpenCL
	CLPlatform platform;
	CLDevice device;
	CLContext context;
	CLQueue queue;
	CLProgram program;

	CLKernel kernelClean;

	CLKernel * kernelActivation;

	CLKernel kernelChiSquared;
	CLKernel kernelChiSquaredReduce;
	CLKernel kernelJacobian;
	CLKernel kernelDelta;
	CLKernel kernelCholeskyDecomposition;

	//Neural Network
	CLString name;
	CLUInt nPatterns;
	CLUInt nInputs;
	CLUInt nHiddenLayers;
	CLActivation * activationPerLayer;
	CLUInt * neuronsPerLayer;
	CLUInt nTargets;

	CLMatrix * inputs;
	CLMatrix * targets;

	CLMatrix * weights;
	CLMatrix * weightsTemp;
	CLMatrix ** weightsForLayer;
	CLMatrix * outputs;


	CLMem chiSquaredError;
	CLMatrix ** hActivations;
	CLMatrix * jacobian;
	CLMatrix * hessian;

	CLMatrix * d;
	CLMatrix * delta;
	CLMatrix * cholesky;
	CLMem illMem;
	CLBool ill;

	CLFloat learningRate;

	CLUInt verbose;
	CLUInt maxIteration;
	CLFloat initialLambda;
	CLFloat upFactor;
	CLFloat downFactor;
	CLFloat targetDeltaError;
	CLFloat finalError;
	CLFloat finalDeltaError;

	CLBool inputsCopiedIntoJacobian;
} CLAnn;

void CLAnnInit(CLAnn * nn, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddenLayers, CLActivation * activationPerLayer,  CLUInt * neuronsPerLayer, CLUInt nTargets, CLStringConst name);
void CLAnnUpdateWithRandomWeights(CLAnn * nn);
void CLAnnShufflePatterns(CLAnn * nn);

void CLAnnSetupTrainingFor(CLAnn * nn, CLPlatform platform, CLDevice device);
void CLAnnForward(CLAnn * nn, CLUInt updateWeightsFromHost, CLUInt printOutputs);
CLFloat CLAnnChiSquared(CLAnn * nn);

CLUInt CLAnnTraining(CLAnn * nn);
void CLAnnPrintResults(CLAnn * nn);

void CLAnnRelease(CLAnn * nn);

#endif /* CLAnn_h */
