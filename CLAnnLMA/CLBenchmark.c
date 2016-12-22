//
//  CLBenchmark.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/9/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLBenchmark.h"

FILE * f;

void CLBenchmarkSetup(CLStringConst path)
{
	f = fopen(path, "a");
}

void CLBenchmarkLog(CLEvent start, CLEvent finish, CLSize loads, CLSize stores, CLSize elements, CLSize dataSize, CLSize operations, CLStringConst name)
{
	if (f == NULL) return;

	CLDouble totalTimeNS, totalTimeMS, bandwidth, flops;

	totalTimeNS = timeBetweenEventsNS(start, finish);
	totalTimeMS = timeBetweenEventsMS(start, finish);

	bandwidth = dataSize / totalTimeNS;
	flops = operations / totalTimeNS;
	//name - elements - time - bandwidth - flops
	fprintf(f, "%s\t%zu\t%10g\t%10g\t%10g\n", name, elements, totalTimeMS, bandwidth, flops);

}
