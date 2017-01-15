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
//	return 0.1 * a + 0.2 * b - 0.1;
	return cos(a);

}

void setupNetForXOR(CLAnn * net)
{
	CLString name = "XOR";
	CLUInt nPattern = 4;
	CLUInt nInputs = 2;
	CLUInt nHiddens = 10;
	CLUInt nOutputs = 1;

	CLFloat _inputs[] = {0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0};

	CLFloat _outputs[] = {0.0,
		1.0,
		1.0,
		0.0};

	CLAnnInit(net, nPattern, nInputs, nHiddens, nOutputs, name);

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
	CLUInt nHiddens = 4;
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

	CLAnnInit(net, nPattern, nInputs, nHiddens, nOutputs, name);

	CLMatrixUpdateValues(net->inputs, _inputs);
	CLMatrixUpdateValues(net->targets, _outputs);
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

	CLAnn * net = malloc(sizeof(CLAnn));
	setupNetForXOR(net);

	//OpenCL stuff
	CLInt platformIndex = 0;
	CLInt deviceIndex = 2;
	CLPlatform platform = CLSelectPlatform(platformIndex);
	CLDevice device = CLSelectDevice(platform, deviceIndex);

	CLAnnSetupTrainingFor(net, platform, device, ACTIVATION_TANSIG);
	CLAnnTraining(net, CLTrue);

	CLAnnForward(net, CLTrue, CLFalse);
	CLAnnPrintResults(net);

	//CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");

	CLAnnRelease(net);

	return 0;
}