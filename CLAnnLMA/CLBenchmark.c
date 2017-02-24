//
//  CLBenchmark.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/9/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLBenchmark.h"

FILE * fB;
FILE * fE;

#define BUFFER_SIZE 128

void CLBenchmarkSetup(CLStringConst path)
{
	CLString fBPath = malloc(BUFFER_SIZE);
	snprintf(fBPath, BUFFER_SIZE - 1, "%s/benchmark_bandwidth.csv", path);
	CLString fEPath = malloc(BUFFER_SIZE);
	snprintf(fEPath, BUFFER_SIZE - 1, "%s/benchmark_elements.csv", path);

	fB = fopen(fBPath, "w+");
	fE = fopen(fEPath, "w+");
	fprintf(fB, "Label;Elements;Time(ms);Bandwidth(GB/s);GFlops\n");
	fprintf(fE, "Label;Elements;Time(ms);GE/s;GFlops\n");

	printf("benchmark saved!\n");

	free(fBPath);
	free(fEPath);
}

void CLBenchmarkBandwidthLog(CLEvent start, CLEvent finish, CLSize elements, CLSize dataSize, CLSize operations, CLStringConst name)
{
	if (fB == NULL) return;

	CLDouble totalTimeNS, totalTimeMS, bandwidth, gflops;

	totalTimeNS = timeBetweenEventsNS(start, finish);
	totalTimeMS = totalTimeNS * 1e-6;

	bandwidth = dataSize / totalTimeNS;
	gflops = operations / totalTimeNS;
	fprintf(fB, "%s;%zu;%g;%g;%g\n", name, elements, totalTimeMS, bandwidth, gflops);
}

void CLBenchmarkElementsLog(CLEvent start, CLEvent finish, CLSize elements, CLSize operations, CLStringConst name)
{
	if (fE == NULL) return;

	CLDouble totalTimeNS, totalTimeMS, gElements, gFlops;

	totalTimeNS = timeBetweenEventsNS(start, finish);
	totalTimeMS = totalTimeNS * 1e-6;

	gElements = elements / totalTimeNS;
	gFlops = operations / totalTimeNS;
	fprintf(fE, "%s;%zu;%g;%g;%g\n", name, elements, totalTimeMS, gElements, gFlops);
}

void CLBenchmarkClose()
{
	if (fB != NULL)
		fclose(fB);

	if (fE != NULL) {
		fclose(fE);
	}
}

