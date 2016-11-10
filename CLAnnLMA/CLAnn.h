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

typedef struct {

	//OpenCL
	CLPlatform platform;
	CLDevice device;
	CLContext context;
	CLQueue queue;
	CLProgram program;

	CLKernel kernelActivation;
	CLKernel kernelChiSquared;
	CLKernel kernelChiSquaredReduce;
	CLKernel kernelJacobian;
	CLKernel kernelDelta;
	CLKernel kernelCholeskyDecomposition;

	//Neural Network
	CLString name;
	CLUInt nPatterns;
	CLUInt nInputs;
	CLUInt nHiddens;
	CLUInt nTargets;

	CLMatrix * inputs;
	CLMatrix * targets;

	CLMatrix * weights;
	CLMatrix * outputs;

	CLMatrix * hActivations;
	CLMatrix * jacobian;
	CLMatrix * hessian;

	CLMatrix * delta;
	CLMatrix * cholesky;
	CLUInt ill;

	CLUInt verbose;
	CLUInt maxIteration;
	CLFloat initialLambda;
	CLFloat upFactor;
	CLFloat downFactor;
	CLFloat targetDeltaError;
	CLFloat finalError;
	CLFloat finalDeltaError;
	
} CLAnn;

void CLAnnInit(CLAnn * nn, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddens, CLUInt nTargets, CLStringConst name);

void CLAnnSetupTrainingFor(CLAnn *nn, CLPlatform platform, CLDevice device);
void CLAnnForward(CLAnn * nn, CLUInt updateWeightsFromHost, CLUInt printOutputs);
CLFloat CLAnnChiSquared(CLAnn * nn);
void CLAnnJacobian(CLAnn * nn);
void CLAnnHessian(CLAnn * nn);
void CLAnnCholeskyDecomposition(CLAnn * nn, CLFloat mult);
void CLAnnCholeskySolve(CLAnn * nn);

CLUInt CLAnnTraining(CLAnn * nn);

void CLAnnPrintResults(CLAnn * nn);

void CLAnnRelease(CLAnn * nn);

#endif /* CLAnn_h */
