//
//  Common.h
//  CLNNBackpropagation
//
//  Created by Albertomac on 9/30/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#ifndef Common_h
#define Common_h

//OpenCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//Bolean
#define CLTrue 1
#define CLFalse 0

//Benchmark
#define BENCHMARK CLFalse

//Debug
#define DEBUG_LOG CLTrue

#define debugLog(fmt, ...) \
do { if (DEBUG_LOG) fprintf(stderr, fmt, __VA_ARGS__); } while (0)



//Types
#define CLPlatform cl_platform_id
#define CLDevice   cl_device_id
#define CLContext  cl_context
#define CLQueue    cl_command_queue
#define CLProgram  cl_program
#define CLKernel   cl_kernel
#define CLEvent    cl_event
#define CLMem      cl_mem
#define CLMemFlags cl_mem_flags
#define CLDeviceType cl_device_type


#define CLBool	 cl_int
#define CLChar   cl_char
#define CLUChar  cl_uchar
#define CLShort  cl_short
#define CLUShort cl_ushort
#define CLInt    cl_int
#define CLUInt   cl_uint
#define CLLong   cl_long
#define CLULong  cl_ulong
#define CLHalf   cl_half
#define CLFloat  cl_float
#define CLDouble cl_double

#define CLString char *
#define CLStringConst const CLString
#define CLSize size_t
#define CLSSize ssize_t

//Precision of values
#define CLNetPrecisionDouble CLTrue

#if CLNetPrecisionDouble
	#define CLNetDataType CLDouble
	#define CLNetDataTypeScanf "%lf"
	#define CLFTollerance 1.0e-10

	#define GEMM clblasDgemm
	#define TRSV clblasDtrsv
	#define AXPY clblasDaxpy

#else

	#define CLNetDataType CLFloat
	#define CLNetDataTypeScanf "%f"
	#define CLFTollerance 1.0e-5

	#define GEMM clblasSgemm
	#define TRSV clblasStrsv
	#define AXPY clblasSaxpy
#endif


//Kernels
#define kMemSet "arrayMemSet"

#define nActivationFunctions 3
#define kActivationSigmoid "activationSigmoid"
#define kActivationTansig "activationTansig"
#define kActivationRadbas "activationRadbas"
//#define kActivationPerceptron "activationPerceptron"

#define kChiSquared "chiSquared"
#define kChiSquaredReduce "chiSquaredReduce"
#define kJacobian "jacobian"
#define kDelta "delta"
#define kUpdateDiagonal "updateDiagonal"
#define kCholeskyDecomposition "choleskyDecomposition"


#endif /* Common_h */
