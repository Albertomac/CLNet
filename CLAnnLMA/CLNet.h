//
//  CLNet.h
//  CLAnnLMA
//
//  Created by Albertomac on 1/26/17.
//  Copyright © 2017 Albertomac. All rights reserved.
//

#ifndef CLNet_h
#define CLNet_h

#include <stdio.h>
#include "CLManager.h"
#include "CLMatrix.h"

#pragma mark CLDeviceContext

typedef struct {

	CLPlatform platform;
	CLDevice device;
	CLContext context;
	CLQueue queue;
	CLProgram program;

	CLKernel kernelMemSet;
	CLKernel * kernelsActivation;
	CLKernel kernelChiSquared;
	CLKernel kernelChiSquaredReduce;
	CLKernel kernelJacobian;
	CLKernel kernelDelta;
	CLKernel kernelCholeskyDecomposition;

} CLDeviceContext;

void CLDeviceContextInit(CLDeviceContext * devContext, CLPlatform platform, CLDevice device);
void CLDeviceContextCleanUp(CLDeviceContext * devContext);
#define CLDeviceContextRelease(devContext) do { CLDeviceContextCleanUp(devContext); devContext = NULL; } while(0);


#pragma mark CLNet

typedef enum CLActivation_ {
	CLActivationLinear,
	CLActivationSigmoid,
	CLActivationTansig,
	CLActivationRadbas,
	//TODO: programmarlo se viene facile
	//CLActivationPerceptron
} CLActivation;

typedef struct {

	CLString name;
	CLUInt bias;
	CLUInt nPatterns;
	CLUInt nInputs;
	CLUInt nLayers;
	CLUInt * neuronsPerLayer;
	CLActivation * activationFunctionPerLayer;
	CLUInt nTargets;

	CLUInt nTestPatterns;
	CLUInt nTrainingPatterns;
	CLUInt nWeights;
	CLNetDataType * p;
	CLNetDataType * w;
	CLNetDataType * t;

	CLMatrix * testPatterns;
	CLMatrix * trainingPatterns;
	CLMatrix * weights;
	CLMatrix * weightsTemp;
	CLMatrix ** weightsPerLayer;
	CLMatrix ** activationPerLayer;
	CLMatrix * outputs; //Pointer to last CLMatrix in activationPerLayer
	CLMatrix * testTargets;
	CLMatrix * trainingTargets;

	CLMatrix * chiSquaredError;
	CLNetDataType errorChiSquared;

	CLBool partialJacobianFilled;
	CLMatrix * jacobian;
	CLMatrix * hessian;
	CLMatrix * d;
	CLMatrix * delta;
	CLMatrix * cholesky;
	CLMatrix * choleskySums;
	CLBool ill;
	CLMem illMem;

	CLBool verbose;
	CLUInt maxIterations;
	CLNetDataType initialLambda;
	CLNetDataType upFactor;
	CLNetDataType downFactor;
	CLNetDataType targetDeltaError;
	CLNetDataType finalError;
	CLNetDataType finalDeltaError;

	CLBool benchmark;

} CLNet;

void CLNetInit(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * patterns,
			   CLUInt nLayers, CLUInt * neuronsPerLayer, CLActivation * activationFunctionPerLayer,
			   CLUInt nTargets, CLNetDataType * targets,
			   CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns);

//void CLNetInitWithFile(CLNet * net, CLStringConst fileName);

void CLNetTrainWithDeviceContext(CLNet * net, CLDeviceContext * devContext);

//void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLNetDataType * inputs);

void CLNetCleanUp(CLNet * net);
#define CLNetRelease(net) do {CLNetCleanUp(net); net = NULL; } while(0);

#endif /* CLNet_h */
