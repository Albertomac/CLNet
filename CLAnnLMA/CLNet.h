////
////  CLNet.h
////  CLAnnLMA
////
////  Created by Albertomac on 1/26/17.
////  Copyright © 2017 Albertomac. All rights reserved.
////
//
//#ifndef CLNet_h
//#define CLNet_h
//
//#include <stdio.h>
//#include "CLManager.h"
//#include "CLMatrix.h"
//
//#pragma mark CLDeviceContext
//
//typedef struct {
//
//	CLPlatform platform;
//	CLDevice device;
//	CLContext context;
//	CLQueue queue;
//	CLProgram program;
//
//	CLKernel kernelMemset;
//	CLKernel * kernelActivation;
//	CLKernel kernelChiSquared;
//	CLKernel kernelChiSquaredReduce;
//	CLKernel kernelJacobian;
//	CLKernel kernelDelta;
//	CLKernel kernelCholeskyDecomposition;
//
//} CLDeviceContext;
//
//void CLDeviceContextInit(CLDeviceContext * devContext, CLPlatform platform, CLDevice device);
//
//
//#pragma mark CLNet
//
//typedef enum CLActivation_ {
//	CLActivationLinear,
//	CLActivationSigmoid,
//	CLActivationTansig,
//	CLActivationRadbas
//} CLActivation;
//
//typedef struct {
//
//	CLString name;
//	CLUInt nTotalPatterns;
//	CLUInt nTrainingPatterns;
//	CLUInt nTestPatterns;
//	CLUInt nInputs;
//	CLUInt nHiddenLayers;
//	CLUInt * nNeuronsPerHiddenLayer;
//	CLActivation * activationFunctionPerLayer;
//	CLUInt nTargets;
//
//	CLMatrix * tests;
//	CLMatrix * inputs;
//	CLMatrix * weights;
//	CLMatrix ** hiddenActivations;
//	CLMatrix * outputs;
//	CLMatrix * targets;
//
//	CLUInt nLayers;
//	CLMatrix ** layers;
//
//	CLMatrix * jacobian;
//	CLMatrix * hessian;
//	CLMatrix * delta;
//	CLMatrix * cholesky;
//	CLBool ill;
//
//	CLBool verbose;
//	CLUInt maxIterations;
//	CLFloat initialLambda;
//	CLFloat upFactor;
//	CLFloat downFactor;
//	CLFloat targetDeltaError;
//	CLFloat finalError;
//	CLFloat finalDeltaError;
//
//} CLNet;
//
//void CLNetInit(CLNet * net, CLUInt nTotalPatterns, CLUInt nInputs, CLFloat * inputs, CLUInt nHiddenLayers, CLUInt * nNeuronsPerHiddenLayer, CLActivation * activationFunctionPerLayer, CLFloat * weights, CLUInt nTargets, CLFloat * targets, CLStringConst name, CLBool shufflePattners, CLUInt nTestPatterns);
//
//void CLNetInitWithFile(CLNet * net, CLStringConst fileName);
//
//void CLNetTrainWithDeviceContext(CLNet * net, CLDeviceContext * devContext);
//
//void CLNetPrintResultsWithInputs(CLNet * net, CLUInt nPatterns, CLUInt nInputs, CLFloat * inputs);
//
//void CLNetRelease(CLNet * net);
//
//#endif /* CLNet_h */
