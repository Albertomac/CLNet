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
void CLBenchmarkLog(CLEvent start, CLEvent finish, CLSize loads, CLSize stores, CLSize elements, CLSize dataSize, CLSize operations, CLStringConst name);
void CLBenchmarkClose();
#endif /* CLBenchmark_h */
