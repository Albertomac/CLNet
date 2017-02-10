//
//  main.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/5/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include <stdio.h>
#include <math.h>
#include "CLManager.h"
#include "CLNet.h"
#include "CLRandom.h"
#include "CLBenchmark.h"


//OpenCL stuff
CLInt platformIndex = 0;
CLInt deviceIndex = 2;
CLPlatform platform;
CLDevice device;

void fillRandom(CLNetDataType * values, CLUInt nValues, CLNetDataType mult, CLNetDataType shift)
{
	for(CLUInt i = 0; i < nValues; ++i) {
		values[i] = CLRandomValue() * mult + shift;
	}
}

void normalize(CLNetDataType * values, CLUInt nValues)
{
	CLNetDataType max = fabs(values[0]);
	for (CLUInt i = 1 ; i < nValues; ++i) {
		CLNetDataType val = values[i];
		if (fabs(val) > max) {
			max = fabs(val);
		}
	}

	for (CLUInt i = 0; i < nValues; ++i) {
		values[i] = values[i] / max;
	}
}

void setupNetForXOR(CLNet * net)
{
	CLString name = "XOR";
	CLUInt nPatterns = 4;
	CLUInt nInputs = 2;
	CLUInt nLayers = 2;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {10, 1};
	CLActivation activationFunctionPerLayer[] = {CLActivationTansig, CLActivationLinear};

	CLNetDataType _inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};

	CLNetDataType _targets[] = {0,	1,	1,	0};

	CLNetInit(net, nPatterns, nInputs, _inputs,
			  nLayers, neuronsPerLayer, activationFunctionPerLayer,
			  nTargets, _targets,
			  name, CLFalse, 0, CLFalse);

	fillRandom(net->w, net->nWeights, 4, -2);

//	CLFloat _weights[] = {0.787020, 0.032042, 0.269297, 0.313040, 0.376689, 0.549111, 0.036677, 0.876073, 0.774661, 0.946820, 0.281833, 0.848981, 0.232944, 0.142625, 0.053538, 0.771335, 0.506342, 0.752660, 0.213712, 0.204351, 0.887152, 0.767608, 0.607750, 0.058241, 0.386845, 0.691405, 0.941201, 0.962479, 0.075253, 0.369933, 0.863612, 0.554582, 0.720409, 0.537345, 0.268498, 0.841953};

//	for (CLUInt i = 0; i < net->nWeights; ++i) {
//		net->w[i] = _weights[i];
//	}
}

void setupNetForIris(CLNet * net)
{
	CLString name = "Iris";
	CLUInt nPatterns = 150;
	CLUInt nInputs = 4;
	CLUInt nLayers = 3;
	CLUInt nTargets = 3;

	CLUInt neuronsPerLayer[] = {7, 5, 3};
	CLActivation activationPerLayer[] = {CLActivationRadbas, CLActivationRadbas, CLActivationLinear};

	CLMatrix * patterns = calloc(1, sizeof(CLMatrix));
	CLMatrixInitWithCSV(patterns, "irisInputs.csv");
	CLMatrixNormalize(patterns);

	CLMatrix * targets = calloc(1, sizeof(CLMatrix));
	CLMatrixInitWithCSV(targets, "irisTargets.csv");

	CLNetInit(net, nPatterns, nInputs, patterns->values,
			  nLayers, neuronsPerLayer, activationPerLayer,
			  nTargets, targets->values,
			  name, CLTrue, 0, CLTrue);
	fillRandom(net->w, net->nWeights, 1, 0);
}

CLNetDataType function(CLNetDataType x)
{
//	return (sin(x) + 4 * cos(x)) / log(1 + sqrt(x));
//	return 2 * cos(10 * x) * sin(10 * x);
	return sin(x) + cos(x);
}

void setupForFunction(CLNet * net)
{
	CLString name = "Function";
	CLUInt nPatterns = 50;
	CLUInt nInputs = 1;
	CLUInt nLayers = 3;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {7, 5, 1};
	CLActivation activationPerLayer[] = {CLActivationTansig, CLActivationTansig, CLActivationLinear};

	CLNetDataType * _patterns = calloc(nPatterns, sizeof(CLNetDataType));
	CLNetDataType * _targets = calloc(nPatterns, sizeof(CLNetDataType));

	fillRandom(_patterns, nPatterns, 3, 2);

	for (CLUInt i = 0; i < nPatterns; ++i) {
		_targets[i] = function(_patterns[i]);
	}

	normalize(_patterns, nPatterns);
//	normalize(_targets, nPatterns);

	CLNetInit(net, nPatterns, nInputs, _patterns,
			  nLayers, neuronsPerLayer, activationPerLayer,
			  nTargets, _targets,
			  name, CLTrue, 0, CLTrue);

	fillRandom(net->w, net->nWeights, 1, 0);
}

int main(int argc, const char * argv[]) {
	
	CLRandomSetup();

	platform = CLSelectPlatform(platformIndex);
	device = CLSelectDevice(platform, deviceIndex);

	CLDeviceContext * devContext = calloc(1, sizeof(CLDeviceContext));
	CLDeviceContextInit(devContext, platform, device);

	CLNet * net = calloc(1, sizeof(CLNet));

	switch (0) {
		case 0:
			setupNetForXOR(net);
			break;

		case 1:
			setupNetForIris(net);
			break;

		case 2:
			setupForFunction(net);
			break;

		default:
			break;
	}

	CLNetTrainWithDeviceContext(net, devContext);

	printf("Would you like to save weights? [Y/N]\n");
	char ans = 'N';
	do {
		ans = getchar();
	} while (ans != 'N' && ans != 'Y' && ans != 'n' && ans != 'y');

	if (ans == 'Y' || ans == 'y') CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");

	CLNetRelease(net);
	CLDeviceContextRelease(devContext);
	
	return 0;
}