//
//  CLRandom.h
//  CLAnnLMA
//
//  Created by Albertomac on 11/8/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef CLRandom_h
#define CLRandom_h

#include <stdio.h>

#ifdef __APPLE__
#include <stdlib.h>
#else
#include <stdlib.h>
#include <bsd/stdlib.h>
#endif

void CLRandomSetup();

#define CLRandomValue() drand48()

#endif /* CLRandom_h */
