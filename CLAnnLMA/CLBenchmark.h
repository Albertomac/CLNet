//
//  CLBenchmark.h
//  CLAnnLMA
//
//  Created by Albertomac on 11/9/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef CLBenchmark_h
#define CLBenchmark_h

#include <stdio.h>
#include "CLManager.h"

void CLBenchmarkSetup(CLStringConst path);
void CLBenchmarkReset(CLStringConst path);
void CLBenchmarkBandwidthLog(CLEvent start, CLEvent finish, CLSize elements, CLSize dataSize, CLSize operations, CLStringConst name);
void CLBenchmarkElementsLog(CLEvent start, CLEvent finish, CLSize elements, CLSize operations, CLStringConst name);
void CLBenchmarkClose();
#endif /* CLBenchmark_h */
