//
//  CLRandom.c
//  CLAnnLMA
//
//  Created by Albertomac on 11/8/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLRandom.h"
#include <stdlib.h>

void CLRandomSetup()
{
	srand48(arc4random());
}
