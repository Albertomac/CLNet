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
//		printf("%g, ", values[i]);
	}
//	printf("\n");
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
	CLActivation activationFunctionPerLayer[] = {CLActivationSigmoid, CLActivationLinear};

	CLNetDataType _inputs[] = {0, 0, 0, 1, 1, 0, 1, 1};

	CLNetDataType _targets[] = {0,	1,	1,	0};

	CLNetInit(net, nPatterns, nInputs, _inputs,
			  nLayers, neuronsPerLayer, activationFunctionPerLayer,
			  nTargets, _targets,
			  name, CLFalse, 0, CLFalse);

	fillRandom(net->w, net->nWeights, 2, -1);

	//2x2x1 CLActivationTansig
//	CLFloat _weights[] =
//	{
//		0.56087324868804345, 0.28466180930578844, 0.48483193599725993, 0.06400278452469621, 0.21066945268565362, 0.71573373999852663
//	};

	//2x10x1 CLActivationTansig
//	CLNetDataType _weights[] =
//	{
//		-0.358296722, -1.11181891, -0.875998139, -0.833521724, -0.770012856, -0.597140372, -1.11013341, -0.271296352, -0.373541415, -0.202476129,
//		-0.825716734, -0.258527756, -0.874694645, -0.964434803, -1.05426013, -0.33673653, -0.601961553, -0.355955094, -0.895164907, -0.904570162,
//		1.07225227, -0.858597457, -2.14572215, -2.50301027, -0.867024899, 3.39165139, -0.0731235817, 5.88008213, -0.408898056, -0.321799964
//	};
//
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
	fillRandom(net->w, net->nWeights, 2, -1);
}

CLNetDataType function(CLNetDataType x)
{
//	return (sin(x) + 4 * cos(x)) / log(1 + sqrt(x));
	return 2 * cos(10 * x) * sin(10 * x);
}

void setupForFunction(CLNet * net)
{
	CLString name = "Function";
	CLUInt nPatterns = 50;
	CLUInt nInputs = 1;
	CLUInt nLayers = 2;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {8, 1};
	CLActivation activationPerLayer[] = {CLActivationTansig, CLActivationLinear};

	CLNetDataType * _patterns = calloc(nPatterns, sizeof(CLNetDataType));
	CLNetDataType * _targets = calloc(nPatterns, sizeof(CLNetDataType));

	fillRandom(_patterns, nPatterns, 2, -1);

	for (CLUInt i = 0; i < nPatterns; ++i) {
		_targets[i] = function(_patterns[i]);
	}

	normalize(_patterns, nPatterns);
//	normalize(_targets, nPatterns);

	CLNetInit(net, nPatterns, nInputs, _patterns,
			  nLayers, neuronsPerLayer, activationPerLayer,
			  nTargets, _targets,
			  name, CLTrue, 0, CLFalse);

	fillRandom(net->w, net->nWeights, 2, -1);
}

int main(int argc, const char * argv[]) {
	
	CLRandomSetup();

	platform = CLSelectPlatform(platformIndex);
	device = CLSelectDevice(platform, deviceIndex);

	CLDeviceContext * devContext = calloc(1, sizeof(CLDeviceContext));
	CLDeviceContextInit(devContext, platform, device);

	CLNet * net = calloc(1, sizeof(CLNet));

	switch (1) {
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