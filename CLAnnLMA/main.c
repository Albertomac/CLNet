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
#include "CLAnn.h"
#include "CLRandom.h"
#include "CLBenchmark.h"


//OpenCL stuff
CLInt platformIndex = 0;
CLInt deviceIndex = 2;
CLPlatform platform;
CLDevice device;

void fillRandom(CLFloat * values, CLUInt nValues)
{
	for(CLUInt i = 0; i < nValues; ++i) {
		values[i] = CLRandomValue();
	}
}

void fillInput(CLFloat * values, CLUInt nValues)
{
	for (CLUInt i = 0; i < nValues; ++i) {
		values[i] = (CLFloat)i / nValues;
	}
}


CLFloat poly(CLFloat a, CLFloat b)
{
	return 0.1 * a + 0.2 * b - 0.1;
	//return cos(a);

}

void setupNetForXOR(CLAnn * net)
{
	CLString name = "XOR";
	CLUInt nPattern = 4;
	CLUInt nInputs = 2;
	CLUInt nHiddenLayers = 1;
	CLUInt nNeuronsPerLayer = 10;
	CLUInt nOutputs = 1;

	CLFloat _inputs[] = {0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0};

	CLFloat _outputs[] = {0.0,
		1.0,
		1.0,
		0.0};

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, nNeuronsPerLayer, nOutputs, name);

	CLMatrixUpdateValues(net->inputs, _inputs);
	CLMatrixUpdateValues(net->targets, _outputs);
	CLAnnUpdateWithRandomWeights(net);
	//CLMatrixInitWithCSV(net->weights, "/Volumes/RamDisk/weights.csv");
}

void setupNetForPoly(CLAnn * net)
{
	CLString name = "Poly";
	CLUInt nPattern = 100;
	CLUInt nInputs = 3;
	CLUInt nHiddenLayers = 1;
	CLUInt nNeuronsPerLayer = 4;
	CLUInt nOutputs = 1;

	CLFloat * _inputs = malloc(sizeof(CLFloat) * nInputs * nPattern);
	fillInput(_inputs, nInputs * nPattern);

	for (CLUInt i = 0; i < nInputs * nPattern; i += nInputs) {
		_inputs[i] = 1;
	}

	CLFloat * _outputs = malloc(sizeof(CLFloat) * nOutputs * nPattern);
	for(CLUInt i = 1, o = 0; i < nInputs * nPattern; i += nInputs, ++o) {
		_outputs[o] = poly(_inputs[i], _inputs[i+1]);
	}

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, nNeuronsPerLayer, nOutputs, name);

	CLMatrixUpdateValues(net->inputs, _inputs);
	CLMatrixUpdateValues(net->targets, _outputs);
	CLAnnUpdateWithRandomWeights(net);
}

void setupNetForIris(CLAnn * net)
{
	CLString name = "Iris";
	CLUInt nPattern = 150;
	CLUInt nInputs = 4;
	CLUInt nHiddenLayers = 2;
	CLUInt nNeuronsPerLayer = 10;
	CLUInt nOutputs = 3;

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, nNeuronsPerLayer, nOutputs, name);

	CLMatrixInitWithCSV(net->inputs, "/Volumes/RamDisk/irisInputs.csv");
	CLMatrixInitWithCSV(net->targets, "/Volumes/RamDisk/irisTargets.csv");

	CLMatrixNormalize(net->inputs);

	CLAnnUpdateWithRandomWeights(net);
}

void setupNetForOrAnd(CLAnn * net)
{
	CLString name = "orAnd";
	CLUInt nPattern = 8;
	CLUInt nInputs = 3;
	CLUInt nHiddenLayers = 2;
	CLUInt nNeuronsPerLayer = 3;
	CLUInt nOutputs = 1;

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, nNeuronsPerLayer, nOutputs, name);

	CLMatrixInitWithCSV(net->inputs, "/Volumes/RamDisk/orAndInputs.csv");
	CLMatrixInitWithCSV(net->targets, "/Volumes/RamDisk/orAndTargets.csv");

	CLMatrixNormalize(net->inputs);

	CLAnnUpdateWithRandomWeights(net);
}


int main(int argc, const char * argv[]) {

	CLRandomSetup();

#if BENCHMARK
	time_t now = time(NULL);
	char pathBenchmark[256];
	snprintf(pathBenchmark, 255, "/Volumes/Ramdisk/Benchmark-%s.csv", ctime(&now));
	CLBenchmarkSetup(pathBenchmark);
#endif

	platform = CLSelectPlatform(platformIndex);
	device = CLSelectDevice(platform, deviceIndex);

	CLAnn * net = malloc(sizeof(CLAnn));

	switch (2) {

		case 0:
			setupNetForXOR(net);
			break;
		case 1:
			setupNetForPoly(net);
			break;
		case 2:
			setupNetForIris(net);
			break;
		case 3:
			setupNetForOrAnd(net);
			break;

		default:
			break;
	}


	CLAnnSetupTrainingFor(net, platform, device, ACTIVATION_TANSIG);

	CLAnnTraining(net);

	CLAnnForward(net, CLTrue, CLFalse);
	CLAnnPrintResults(net);

	printf("Would you like to save weights? [Y/N]\n");
	char ans = 'N';
	do {
		ans = getchar();
	} while (ans != 'N' && ans != 'Y' && ans != 'n' && ans != 'y');

	if (ans == 'Y' || ans == 'y') CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");

	CLAnnRelease(net);

	return 0;
}