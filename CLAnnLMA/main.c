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

int main(int argc, const char * argv[]) {

#if BENCHMARK
	time_t now = time(NULL);
	char pathBenchmark[256];
	snprintf(pathBenchmark, 255, "/Volumes/Ramdisk/Benchmark-%s.csv", ctime(&now));
	CLBenchmarkSetup(pathBenchmark);
	CLRandomSetup();
#endif

	CLString name;
	CLUInt nPattern;
	CLUInt nInputs;
	CLUInt nHiddens;
	CLUInt nOutputs;

#if 0
	name = "XOR";
	nPattern = 4;
	nInputs = 2;
	nHiddens = 10;
	nOutputs = 1;

    CLFloat _inputs[] = {0.0, 0.0,
						 0.0, 1.0,
						 1.0, 0.0,
						 1.0, 1.0};

	CLFloat _outputs[] = {0.0,
						  1.0,
						  1.0,
						  0.0};
#else
	name = "Poly";
	nPattern = 100;
	nInputs = 3;
	nHiddens = 4;
	nOutputs = 1;

	CLFloat * _inputs = malloc(sizeof(CLFloat) * nInputs * nPattern);
	fillInput(_inputs, nInputs * nPattern);

	for (CLUInt i = 0; i < nInputs * nPattern; i += nInputs) {
		_inputs[i] = 1;
	}

	CLFloat * _outputs = malloc(sizeof(CLFloat) * nOutputs * nPattern);
	for(CLUInt i = 1, o = 0; i < nInputs * nPattern; i += nInputs, ++o) {
		_outputs[o] = poly(_inputs[i], _inputs[i+1]);
	}

#endif

	CLAnn * net = malloc(sizeof(CLAnn));
	CLAnnInit(net, nPattern, nInputs, nHiddens, nOutputs, name);


	CLUInt _nWeights = nInputs * nHiddens + nHiddens * nOutputs;
	CLFloat * _weights = malloc(sizeof(CLFloat) * _nWeights);
	for(CLUInt i = 0; i < _nWeights; ++i) {
		_weights[i] = CLRandomValue();
	}
	CLMatrixUpdateValues(net->inputs, _inputs);
	CLMatrixUpdateValues(net->targets, _outputs);
	CLMatrixUpdateValues(net->weights, _weights);

	//OpenCL stuff
	CLInt platformIndex = 0;
	CLInt deviceIndex = 2;
	CLPlatform platform = CLSelectPlatform(platformIndex);
	CLDevice device = CLSelectDevice(platform, deviceIndex);

	CLAnnSetupTrainingFor(net, platform, device);
	CLAnnTraining(net);


	CLAnnForward(net, CLTrue, CLFalse);
	CLAnnPrintResults(net);

//	printf("\n");
//	fillRandom(_inputs, nInputs * nPattern);
//	for(CLUInt i = 0, o = 0; i < nInputs * nPattern; i += nInputs, ++o) {
//		_outputs[o] = poly(_inputs[i], _inputs[i + 1]);
//	}

//	CLMatrixUpdateValues(net->inputs, _inputs);
//	CLMatrixReleaseMem(net->inputs);
//	CLMatrixCreateMemHostVar(net->inputs, net->context, CL_MEM_READ_ONLY);
//	CLMatrixUpdateValues(net->targets, _outputs);
//	CLMatrixReleaseMem(net->targets);
//	CLMatrixCreateMemHostVar(net->targets, net->context, CL_MEM_READ_ONLY);
//
//	CLAnnForward(net, CLTrue, CLFalse);
//	CLAnnPrintResults(net);

	CLAnnRelease(net);

	return 0;
}