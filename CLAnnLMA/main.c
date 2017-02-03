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

void fillRandom(CLNetDataType * values, CLUInt nValues)
{
	for(CLUInt i = 0; i < nValues; ++i) {
		values[i] = CLRandomValue();
	}
}

void fillInput(CLNetDataType * values, CLUInt nValues)
{
	for (CLUInt i = 0; i < nValues; ++i) {
		values[i] = (CLNetDataType)i / nValues;
	}
}


CLNetDataType poly(CLNetDataType a, CLNetDataType b)
{
	return 0.1 * a + 0.2 * b - 0.1;
	//return cos(a);

}

void setupNetForXOR(CLNet * net)
{
	CLString name = "TEST";
	CLUInt nPatterns = 4;
	CLUInt nInputs = 2;
	CLUInt nLayers = 2;
	CLUInt nTargets = 1;

	CLUInt neuronsPerLayer[] = {10, nTargets};
	CLActivation activationFunctionPerLayer[] = {CLActivationTansig, CLActivationLinear};


	CLNetDataType _inputs[] = {0.0, 0.0,
		0.0, 1.0,
		1.0, 0.0,
		1.0, 1.0};

	CLNetDataType _targets[] = {0.0,
		1.0,
		1.0,
		0.0};

	CLNetInit(net, nPatterns, nInputs, _inputs,
			  nLayers, neuronsPerLayer, activationFunctionPerLayer,
			  nTargets, _targets,
			  name, CLFalse, 0);

	//	float _weights[] = {-2.945651,  3.838210,  0.744880, -0.536410,  6.317570,  4.007711,  2.186176,  4.534352, -0.412075,  0.093408,
	//		6.255346,  3.802628, -0.179818,  1.505547, -2.968742,  3.895006, -0.920345,  4.484747,  1.136460,  0.448697,
	//		-7.942422,  5.197197, -1.054466, -1.915232, -7.835067,  5.540699, -2.905327,  7.677764, -2.450965, -1.859408};
	//	CLMatrixUpdateValues(net->weights, _weights);

	fillRandom(net->w, net->nWeights);
	//CLMatrixInitWithCSV(net->weights, "/Users/Albertomac/Desktop/irisDataSet/weights36.csv");
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
			  name, CLFalse, 0);

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

	CLNet * net = malloc(sizeof(CLNet));
	setupNetForXOR(net);

	CLNetTrainWithDeviceContext(net, devContext);

	//	printf("Would you like to save weights? [Y/N]\n");
	//	char ans = 'N';
	//	do {
	//		ans = getchar();
	//	} while (ans != 'N' && ans != 'Y' && ans != 'n' && ans != 'y');
	//
	//	if (ans == 'Y' || ans == 'y') CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");
	
	CLNetRelease(net);
	
	return 0;
}

////
////  main.c
////  CLAnnLMA
////
////  Created by Albertomac on 11/5/16.
////  Copyright © 2016 Albertomac. All rights reserved.
////
//
//#include <stdio.h>
//#include <math.h>
//#include "CLManager.h"
//#include "CLAnn.h"
//#include "CLRandom.h"
//#include "CLBenchmark.h"
//
//
////OpenCL stuff
//CLInt platformIndex = 0;
//CLInt deviceIndex = 2;
//CLPlatform platform;
//CLDevice device;
//
//void fillRandom(CLNetDataType * values, CLUInt nValues)
//{
//	for(CLUInt i = 0; i < nValues; ++i) {
//		values[i] = CLRandomValue();
//	}
//}
//
//void fillInput(CLNetDataType * values, CLUInt nValues)
//{
//	for (CLUInt i = 0; i < nValues; ++i) {
//		values[i] = (CLNetDataType)i / nValues;
//	}
//}
//
//
//CLNetDataType poly(CLNetDataType a, CLNetDataType b)
//{
//	return 0.1 * a + 0.2 * b - 0.1;
//	//return cos(a);
//
//}
//
//void setupNetForXOR(CLAnn * net)
//{
//	CLString name = "XOR";
//	CLUInt nPattern = 4;
//	CLUInt nInputs = 2;
//	CLUInt nHiddenLayers = 1;
//	CLActivation activationPerLayer[] = {CLActivationTansig};
//	CLUInt nNeuronsPerLayer[] = {12};
//	CLUInt nOutputs = 1;
//
//	CLNetDataType _inputs[] = {0.0, 0.0,
//		0.0, 1.0,
//		1.0, 0.0,
//		1.0, 1.0};
//
//	CLNetDataType _outputs[] = {0.0,
//		1.0,
//		1.0,
//		0.0};
//
//	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, nNeuronsPerLayer, nOutputs, name);
//
//	CLMatrixUpdateValues(net->inputs, _inputs);
//	CLMatrixUpdateValues(net->targets, _outputs);
//
////	float _weights[] = {-2.945651,  3.838210,  0.744880, -0.536410,  6.317570,  4.007711,  2.186176,  4.534352, -0.412075,  0.093408,
////		6.255346,  3.802628, -0.179818,  1.505547, -2.968742,  3.895006, -0.920345,  4.484747,  1.136460,  0.448697,
////		-7.942422,  5.197197, -1.054466, -1.915232, -7.835067,  5.540699, -2.905327,  7.677764, -2.450965, -1.859408};
////	CLMatrixUpdateValues(net->weights, _weights);
//
//	CLAnnUpdateWithRandomWeights(net);
//	//CLMatrixInitWithCSV(net->weights, "/Users/Albertomac/Desktop/irisDataSet/weights36.csv");
//}
//
//void setupNetForPoly(CLAnn * net)
//{
//	CLString name = "Poly";
//	CLUInt nPattern = 100;
//	CLUInt nInputs = 3;
//	CLUInt nHiddenLayers = 1;
//	CLActivation activationPerLayer[] = {CLActivationTansig};
//	CLUInt nNeuronsPerLayer[] = {4};
//	CLUInt nOutputs = 1;
//
//	CLNetDataType * _inputs = malloc(sizeof(CLNetDataType) * nInputs * nPattern);
//	fillInput(_inputs, nInputs * nPattern);
//
//	for (CLUInt i = 0; i < nInputs * nPattern; i += nInputs) {
//		_inputs[i] = 1;
//	}
//
//	CLNetDataType * _outputs = malloc(sizeof(CLNetDataType) * nOutputs * nPattern);
//	for(CLUInt i = 1, o = 0; i < nInputs * nPattern; i += nInputs, ++o) {
//		_outputs[o] = poly(_inputs[i], _inputs[i+1]);
//	}
//
//	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, nNeuronsPerLayer, nOutputs, name);
//
//	CLMatrixUpdateValues(net->inputs, _inputs);
//	CLMatrixUpdateValues(net->targets, _outputs);
//	CLAnnUpdateWithRandomWeights(net);
//}
//
//void setupNetForIris(CLAnn * net)
//{
//	CLString name = "Iris";
//	CLUInt nPattern = 150;
//	CLUInt nInputs = 4;
//	CLUInt nHiddenLayers = 2;
//	CLActivation activationPerLayer[] = {CLActivationRadbas, CLActivationTansig};
//	CLUInt nNeuronsPerLayer[] = {7, 5};
//	CLUInt nOutputs = 3;
//
//	CLAnnInit(net, nPattern, nInputs, nHiddenLayers, activationPerLayer, (CLUInt *)nNeuronsPerLayer, nOutputs, name);
//
//	CLMatrixInitWithCSV(net->inputs, "irisInputs.csv");
//	CLMatrixInitWithCSV(net->targets, "irisTargets.csv");
//
//	CLMatrixNormalize(net->inputs);
//	CLAnnShufflePatterns(net);
//	
//	CLAnnUpdateWithRandomWeights(net);
//}
//
//void freeArray(CLNetDataType * array) {
//	free(array);
//	array = NULL;
//}
//
//
//int main(int argc, const char * argv[]) {
//
//	CLRandomSetup();
//
//#if BENCHMARK
//	time_t now = time(NULL);
//	char pathBenchmark[256];
//	snprintf(pathBenchmark, 255, "/Volumes/Ramdisk/Benchmark-%s.csv", ctime(&now));
//	CLBenchmarkSetup(pathBenchmark);
//#endif
//
//	platform = CLSelectPlatform(platformIndex);
//	device = CLSelectDevice(platform, deviceIndex);
//
//	CLAnn * net = malloc(sizeof(CLAnn));
//
//	switch (0) {
//
//		case 0:
//			setupNetForXOR(net);
//			break;
//		case 1:
//			setupNetForPoly(net);
//			break;
//		case 2:
//			setupNetForIris(net);
//			break;
//
//		default:
//			break;
//	}
//
//	CLAnnSetupTrainingFor(net, platform, device);
//
//	CLAnnTraining(net);
//
//	CLAnnForward(net, CLTrue, CLFalse);
//	CLAnnPrintResults(net);
//
////	printf("Would you like to save weights? [Y/N]\n");
////	char ans = 'N';
////	do {
////		ans = getchar();
////	} while (ans != 'N' && ans != 'Y' && ans != 'n' && ans != 'y');
////
////	if (ans == 'Y' || ans == 'y') CLMatrixSaveCSV(net->weights, "/Volumes/RamDisk/weights.csv");
//
//	CLAnnRelease(net);
//
//	return 0;
//}