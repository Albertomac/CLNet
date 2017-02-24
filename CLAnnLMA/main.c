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

#include <signal.h>
#include "CLBenchmark.h"

//OpenCL stuff
CLInt platformIndex = 3;
CLInt deviceIndex = 0;
CLPlatform platform;
CLDevice device;

void closeHandler(int interrupt)
{
	CLBenchmarkClose();
}

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

	CLUInt neuronsPerLayer[] = {5, 1};
	CLFunction functionPerLayer[] = {CLFunctionSigmoid, CLFunctionLinear};

	CLNetDataType _inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};

	CLNetDataType _targets[] = {0,	1,	1,	0};

	CLNetInit(net, nPatterns, nInputs, _inputs,
			  nLayers, neuronsPerLayer, functionPerLayer,
			  nTargets, _targets,
			  name, CLFalse, 0, CLFalse);
}

void setupNetForIris(CLNet * net)
{
	CLString name = "Iris";
	CLUInt nPatterns = 150;
	CLUInt nInputs = 4;
	CLUInt nLayers = 3;
	CLUInt nTargets = 3;

	CLUInt neuronsPerLayer[] = {7, 5, 3};
	CLFunction functionPerLayer[] = {CLFunctionRadbas, CLFunctionRadbas, CLFunctionLinear};

	CLMatrix * patterns = calloc(1, sizeof(CLMatrix));
	CLMatrixInitWithCSV(patterns, "irisInputs.csv");
	CLMatrixNormalize(patterns);

	CLMatrix * targets = calloc(1, sizeof(CLMatrix));
	CLMatrixInitWithCSV(targets, "irisTargets.csv");

	CLNetInit(net, nPatterns, nInputs, patterns->values,
			  nLayers, neuronsPerLayer, functionPerLayer,
			  nTargets, targets->values,
			  name, CLTrue, 0, CLFalse);
}

CLNetDataType function(CLNetDataType x)
{
//	return (sin(x) + 4 * cos(x)) / log(1 + sqrt(x));
	//return 2 * cos(10 * x) * sin(10 * x);
	return cos(x);
}

void setupForFunction(CLNet * net)
{
	CLString name = "Function";
	CLUInt nPatterns = 2000;
	CLUInt nInputs = 1;
	CLUInt nLayers = 6;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {33, 27, 17, 13, 7, 1};
	CLFunction functionPerLayer[] = {CLFunctionTansig, CLFunctionSigmoid, CLFunctionTansig, CLFunctionRadbas, CLFunctionRadbas, CLFunctionLinear};

	CLNetDataType * _patterns = calloc(nPatterns, sizeof(CLNetDataType));
	CLNetDataType * _targets = calloc(nPatterns, sizeof(CLNetDataType));

	fillRandom(_patterns, nPatterns, 2, -1);

	for (CLUInt i = 0; i < nPatterns; ++i) {
		_targets[i] = function(_patterns[i]);
	}

	normalize(_patterns, nPatterns);

	CLNetInit(net, nPatterns, nInputs, _patterns,
			  nLayers, neuronsPerLayer, functionPerLayer,
			  nTargets, _targets,
			  name, CLTrue, 0, CLFalse);
}

int main(int argc, const char * argv[]) {

	signal(SIGINT, closeHandler);

	CLRandomSetup();

	platform = CLSelectPlatform(platformIndex);
	device = CLSelectDevice(platform, deviceIndex);

	CLDeviceContext * devContext = calloc(1, sizeof(CLDeviceContext));
	CLDeviceContextInit(devContext, platform, device);

	CLNet * net = calloc(1, sizeof(CLNet));

	switch (2) {
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

	net->maxIterations = 10;
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