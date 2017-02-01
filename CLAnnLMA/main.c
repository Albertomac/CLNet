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
	CLActivation activationPerLayer[] = {CLActivationTansig};
	CLUInt nNeuronsPerLayer[] = {12};
	CLUInt nOutputs = 1;

	CLFloat _inputs[] = {0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0};

	CLFloat _outputs[] = {0.0,
		1.0,
		1.0,
		0.0};

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, nNeuronsPerLayer, nOutputs, name);

	CLMatrixUpdateValues(net->inputs, _inputs);
	CLMatrixUpdateValues(net->targets, _outputs);

//	float _weights[] = {-2.945651,  3.838210,  0.744880, -0.536410,  6.317570,  4.007711,  2.186176,  4.534352, -0.412075,  0.093408,
//		6.255346,  3.802628, -0.179818,  1.505547, -2.968742,  3.895006, -0.920345,  4.484747,  1.136460,  0.448697,
//		-7.942422,  5.197197, -1.054466, -1.915232, -7.835067,  5.540699, -2.905327,  7.677764, -2.450965, -1.859408};
//	CLMatrixUpdateValues(net->weights, _weights);

	CLAnnUpdateWithRandomWeights(net);
	//CLMatrixInitWithCSV(net->weights, "/Users/Albertomac/Desktop/irisDataSet/weights36.csv");
}

void setupNetForPoly(CLAnn * net)
{
	CLString name = "Poly";
	CLUInt nPattern = 100;
	CLUInt nInputs = 3;
	CLUInt nHiddenLayers = 1;
	CLActivation activationPerLayer[] = {CLActivationTansig};
	CLUInt nNeuronsPerLayer[] = {4};
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

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, nNeuronsPerLayer, nOutputs, name);

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
	CLActivation activationPerLayer[] = {CLActivationRadbas, CLActivationTansig};
	CLUInt nNeuronsPerLayer[] = {7, 5};
	CLUInt nOutputs = 3;

	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, (CLUInt *)nNeuronsPerLayer, nOutputs, name);

	CLMatrixInitWithCSV(net->inputs, "/Users/Albertomac/Desktop/irisDataSet/irisInputs.csv");
	CLMatrixInitWithCSV(net->targets, "/Users/Albertomac/Desktop/irisDataSet/irisTargets.csv");

	CLMatrixNormalize(net->inputs);
	CLAnnShufflePatterns(net);
	
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

		default:
			break;
	}

	CLAnnSetupTrainingFor(net, platform, device);

	CLAnnTraining(net);

	CLAnnForward(net, CLTrue, CLFalse);
	CLAnnPrintResults(net);

//	printf("Would you like to save weights? [Y/N]\n");
//	char ans = 'N';
//	do {
//		ans = getchar();
//	} while (ans != 'N' && ans != 'Y' && ans != 'n' && ans != 'y');
//
//	if (ans == 'Y' || ans == 'y') CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");

	CLAnnRelease(net);

	return 0;
}