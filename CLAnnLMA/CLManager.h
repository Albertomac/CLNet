//
//  CLManager.h
//  CLNNBackpropagation
//
//  Created by Albertomac on 9/30/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef CLManager_h
#define CLManager_h


#define CHECK_EXIT CLTrue
#define CHECK_NOT_EXIT CLFalse

#include "Common.h"

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>


#pragma mark Print

void CLPrintPlatforms();
void CLPrintDevices(CLPlatform platform);
void CLPrintStats(CLEvent start, CLEvent finish, CLUInt operations, CLStringConst name);
CLDouble timeBetweenEventsNS(CLEvent start, CLEvent finish);
CLDouble timeBetweenEventsMS(CLEvent start, CLEvent finish);
CLDouble timeBetweenEventsS(CLEvent start, CLEvent finish);

#pragma mark Error

void CLErrorCheck(CLInt error, CLStringConst function, CLStringConst message, CLInt needExit);


#pragma mark Select

CLPlatform CLSelectPlatform(CLInt platformIndex);
CLDevice CLSelectDevice(CLPlatform platform, CLInt deviceIndex);


#pragma mark Create

CLContext CLCreateContext(CLPlatform platform, CLDevice device);
CLQueue CLCreateQueue(CLContext context, CLDevice device);
CLProgram CLCreateProgram(CLContext context, CLDevice device, CLStringConst fileName);
CLProgram CLCreateProgramWithMacro(CLContext context, CLDevice device, CLStringConst fileName, CLStringConst macro);

CLKernel CLCreateKernel(CLProgram program, CLStringConst name);
CLMem CLCreateBufferHostVar(CLContext context, CLMemFlags flags, CLSize size, void * hostVar, CLStringConst name);
CLMem CLCreateBuffer(CLContext context, CLMemFlags flags, CLSize size, CLStringConst name);
CLMem CLCreateSubBuffer(CLMem mem, CLMemFlags flags, CLSize offset, CLSize size, CLStringConst name);


#pragma mark Kernel Stuff

void CLSetKernelArg(CLKernel kernel, CLUInt index, CLSize size, const void * arg, CLStringConst name);
void CLEnqueueNDRangeKernel(CLQueue queue, CLKernel kernel, const CLInt workDim, const CLSize * globalWorkOffset, const CLSize * globalWorkSize, const CLSize * localWorkSize, CLUInt numberOfEventsWaitList, const CLEvent * eventsWaitList, CLEvent * event, CLStringConst name);
CLSize CLGetPreferredWorkGroupSizeMultiple(CLKernel kernel, CLDevice device, CLStringConst name);
CLSize CLGetOptimalGlobalWorkItemsSize(CLSize numberOfElements, CLSize lws);

#pragma mark Queue

void CLWaitForEvent(CLEvent * event, CLStringConst name);
void * CLEnqueueReadBufferWithEvent(CLQueue queue, CLMem mem, CLSize size, CLEvent * event, CLStringConst name);
void * CLEnqueueReadBuffer(CLQueue queue, CLMem mem, CLSize size, CLStringConst name);

#pragma mark Finish and Relase

void CLFinish(CLQueue queue);
void CLReleaseDevice(CLDevice device, CLStringConst name);
void CLReleaseContext(CLContext context, CLStringConst name);
void CLReleaseQueue(CLQueue queue, CLStringConst name);
void CLReleaseProgram(CLProgram program, CLStringConst name);
void CLReleaseKernel(CLKernel kernel, CLStringConst name);
void CLReleaseMemObject(CLMem var, CLStringConst name);
void CLReleaseEvent(CLEvent event, CLStringConst name);


#pragma mark Useful Stuff

CLSize roundUpSize(CLSize elements, CLSize multiple);
CLUInt roundUpUInt(CLUInt elements, CLUInt multiple);
CLSize divUpSize(CLSize elements, CLSize divider);
CLUInt divUpUInt(CLUInt elements, CLUInt divider);

CLString getErrorName(CLInt errorCode);
CLDeviceType CLGetDeviceType(CLDevice device);


#endif /* CLManager_h */
