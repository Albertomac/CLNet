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

/*
 TODO:
 use weightsTemps and copy weights when needed with clEnqueueCopyBuffers
 normalization of inputs using clblasSnrm2
 hidden multi layers
 
 benchmarks file with useful runs
 improve opencl kernels

 //[ NON SI PUO' FARE ]
 //Packed vuol dire che la matrice è linearizzata, tipo una matrice trinagolare viene linearizzata eliminando gli zeri che stanno sopra (o sotto) alla diagonale.
 //Banded vuol dire che la matrice contiene degli zeri e si può risparmiare spazio e di conseguenza è più veloce nell'effettuare operazioni
 //improve clblas with banded operations

 getting works clblas for cpu

 Aggiungere la possibilità di stabilire la % di training set e la % di test

 new Name for the project
 */

#define ACTIVATION_LINEAR  0
#define ACTIVATION_SIGMOID 1 //Logistic
#define ACTIVATION_TANSIG  2

typedef struct {

	//OpenCL
	CLPlatform platform;
	CLDevice device;
	CLContext context;
	CLQueue queue;
	CLProgram program;

	CLKernel kernelClean;
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
	CLUInt nHiddenLayers;
	CLUInt nNeuronsPerLayer;
	CLUInt nTargets;

	CLMatrix * inputs;
	CLMatrix * targets;

	CLMatrix * weights;
	CLMatrix * weightsTemp;
	CLMatrix * outputs;

	CLMatrix ** hActivations;
	CLMatrix * jacobian;
	CLMatrix * hessian;

	CLMatrix * delta;
	CLMatrix * cholesky;
	CLUInt ill;

	CLFloat learningRate;

	CLUInt verbose;
	CLUInt maxIteration;
	CLFloat initialLambda;
	CLFloat upFactor;
	CLFloat downFactor;
	CLFloat targetDeltaError;
	CLFloat finalError;
	CLFloat finalDeltaError;
	
} CLAnn;

void CLAnnInit(CLAnn * nn, CLUInt nPatterns, CLUInt nInputs, CLUInt nHiddenLayers, CLUInt nNeuronsPerLayer, CLUInt nTargets, CLStringConst name);
void CLAnnUpdateWithRandomWeights(CLAnn * nn);

void CLAnnSetupTrainingFor(CLAnn * nn, CLPlatform platform, CLDevice device, int activationFunction);
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
