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
	f = fopen(path, "w+");
	fprintf(f, "Label;Elements;Time(ms);Bandwidth(GB/s);GFlops\n");
}

void CLBenchmarkLog(CLEvent start, CLEvent finish, CLSize loads, CLSize stores, CLSize elements, CLSize dataSize, CLSize operations, CLStringConst name)
{
	if (f == NULL) return;

	CLDouble totalTimeNS, totalTimeMS, bandwidth, flops;

	totalTimeNS = timeBetweenEventsNS(start, finish);
	totalTimeMS = totalTimeNS * 1e-6;

	bandwidth = dataSize / totalTimeNS;
	flops = operations / totalTimeNS;
	//name - elements - time - bandwidth - flops
	fprintf(f, "%s;%zu;%10g;%10g;%10g\n", name, elements, totalTimeMS, bandwidth, flops);

}

void CLBenchmarkClose()
{
	if (f != NULL)
		fclose(f);
}

