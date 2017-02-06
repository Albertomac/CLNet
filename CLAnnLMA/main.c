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
			  name, CLFalse, 0, 1);

	fillRandom(net->w, net->nWeights, 1, 0);
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
			  name, CLTrue, 0, 1);
	fillRandom(net->w, net->nWeights, 1, 0);
}

CLNetDataType function(CLNetDataType x)
{
//	return (sin(x) + 4 * cos(x)) / log(1 + sqrt(x));
	return 2 * cos(10 * x) * sin(10 * x);
}

void setupForFunction(CLNet * net)
{
	CLString name = "Function";
	CLUInt nPatterns = 200;
	CLUInt nInputs = 1;
	CLUInt nLayers = 3;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {10, 7, 1};
	CLActivation activationPerLayer[] = {CLActivationRadbas, CLActivationTansig, CLActivationLinear};

	CLNetDataType * _patterns = calloc(nPatterns, sizeof(CLNetDataType));
	CLNetDataType * _targets = calloc(nPatterns, sizeof(CLNetDataType));

	fillRandom(_patterns, nPatterns, 1, 0);

	for (CLUInt i = 0; i < nPatterns; ++i) {
		_targets[i] = function(_patterns[i]);
	}

	normalize(_patterns, nPatterns);
//	normalize(_targets, nPatterns);

	CLNetInit(net, nPatterns, nInputs, _patterns,
			  nLayers, neuronsPerLayer, activationPerLayer,
			  nTargets, _targets,
			  name, CLTrue, 0, 1);

	fillRandom(net->w, net->nWeights, 1, 0);
}

void setupTEST(CLNet * net)
{
	CLString name = "TEST";
	CLUInt nPatterns = 4096;
	CLUInt nInputs = 4;
	CLUInt nLayers = 3;
	CLUInt nTargets = 8;

	CLUInt neuronsPerLayer[] = {8, 4, 8};
	CLActivation activationFunctionPerLayer[] = {CLActivationRadbas, CLActivationRadbas, CLActivationLinear};

	CLUInt _nInputs = nPatterns * nInputs;
	CLNetDataType * _inputs = calloc(_nInputs, sizeof(CLNetDataType));

	CLUInt _nWeights = nInputs * neuronsPerLayer[0] + neuronsPerLayer[0] * neuronsPerLayer[1] + neuronsPerLayer[1] * neuronsPerLayer[2];
	CLNetDataType * _weights = calloc(_nWeights, sizeof(CLNetDataType));

	CLUInt _nTargets = nPatterns * nTargets;
	CLNetDataType * _targets = calloc(_nTargets, sizeof(CLNetDataType));

	FILE * pFile = fopen("/Volumes/RamDisk/TESTForward/patterns.txt", "r");
	for (CLUInt i = 0; i < _nInputs; ++i) {
		fscanf(pFile, CLNetDataTypeScanf, &_inputs[i]);
	}
	fclose(pFile);

	FILE * wFile = fopen("/Volumes/RamDisk/TESTForward/weights.txt", "r");
	for (CLUInt i = 0; i < _nWeights; ++i) {
		fscanf(wFile, CLNetDataTypeScanf, &_weights[i]);
	}
	fclose(wFile);

	FILE * oFile = fopen("/Volumes/RamDisk/TESTForward/targets.txt", "r");
	for (CLUInt i = 0; i < _nTargets; ++i) {
		fscanf(oFile, CLNetDataTypeScanf, &_targets[i]);
	}
	fclose(oFile);

	CLNetInit(net, nPatterns, nInputs, _inputs,
			  nLayers, neuronsPerLayer, activationFunctionPerLayer,
			  nTargets, _targets,
			  name, CLFalse, 0, 0);

	for (CLUInt i = 0; i < _nWeights; ++i) {
		net->w[i] = _weights[i];
	}
}

int main(int argc, const char * argv[]) {
	
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

		case 3:
			setupTEST(net);
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

//	CLMatrixSaveCSV(net->weightsTemp, "/Volumes/RamDisk/weights.csv");
	CLNetRelease(net);
	CLDeviceContextRelease(devContext);
	
	return 0;
}