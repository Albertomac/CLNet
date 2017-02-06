//
//  CLManager.c
//  CLNNBackpropagation
//
//  Created by Albertomac on 9/30/16.
//  Copyright © 2016 Albertomac. All rights reserved.
//

#include "CLManager.h"

#define BUFFER_SIZE (64 * 1024)

#pragma mark Prints

void CLPrintPlatformInfo(CLPlatform platformID)
{
	CLInt error;
	CLSize nBuffer;
	CLString charBuffer;

	const CLInt nInfo = 5;
	CLString infoNames[]= {"Profile","Platform Name", "Vendor", "Version", "Extensions"};
	cl_platform_info info[] = {CL_PLATFORM_PROFILE, CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION, CL_PLATFORM_EXTENSIONS};

	printf("Platforms:\n");
	for (CLInt i = 0; i < nInfo; ++i) {
		error = clGetPlatformInfo(platformID, info[i], 0, NULL, &nBuffer);
		CLErrorCheck(error, "clGetPlatformInfo", "get nBuffer", CHECK_EXIT);
		charBuffer = calloc(nBuffer, sizeof(charBuffer));
		error = clGetPlatformInfo(platformID, info[i], nBuffer, charBuffer, NULL);
		CLErrorCheck(error, "clGetPlatformInfo", "get platform info", CHECK_EXIT);
		printf("%s: %s\n", infoNames[i], charBuffer);
	}
	printf("\n");
}

void CLPrintPlatforms()
{
	CLInt error;
	CLUInt nPlatforms;
	CLPlatform * platforms;

	error = clGetPlatformIDs(0, NULL, &nPlatforms);
	CLErrorCheck(error, "clGetPlatformIDs", "get number of platforms", CHECK_EXIT);
	platforms = (CLPlatform *)calloc(nPlatforms, sizeof(CLPlatform));
	error = clGetPlatformIDs(nPlatforms, platforms, NULL);
	CLErrorCheck(error, "clGetPlatformIDs", "get platform IDs", CHECK_EXIT);

	for (CLInt i = 0; i < nPlatforms; ++i) {
		printf("PLATFORM #%d\n", i);
		CLPrintPlatformInfo(platforms[i]);
	}
}

void CLPrintDeviceInfo(CLDevice deviceID)
{
	CLInt error;
	CLSize nBuffer;
	CLString charBuffer;
	CLULong ulongBuffer;


	CLInt charNInfo = 3;
	CLString charInfoNames[] = {"Vendor", "Name", "OpenCL Ver."};
	cl_device_info charInfo[] = {CL_DEVICE_VENDOR, CL_DEVICE_NAME, CL_DEVICE_VERSION};

	CLInt ulongNInfo = 5;
	CLString ulongInfoNames[] = {"Frequency", "Global Memroy", "Global Memory Cache", "Global Memory CacheLine", "Local Memory", "Max Compute Units"};
	cl_device_info ulongInfo[] = {CL_DEVICE_MAX_CLOCK_FREQUENCY, CL_DEVICE_GLOBAL_MEM_SIZE, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_MAX_COMPUTE_UNITS};

	printf("Devices Info\n");
	for (CLInt i = 0; i < charNInfo; ++i) {
		error = clGetDeviceInfo(deviceID, charInfo[i], 0, NULL, &nBuffer);
		CLErrorCheck(error, "clGetDeviceInfo", "get nBuffer", CHECK_EXIT);

		charBuffer = calloc(nBuffer, sizeof(charBuffer));

		error = clGetDeviceInfo(deviceID, charInfo[i], nBuffer, charBuffer, NULL);
		CLErrorCheck(error, "clGetDeviceInfo", "get device info", CHECK_EXIT);
		printf("%s: %s\n", charInfoNames[i], charBuffer);
	}


	for (CLInt i = 0; i < ulongNInfo; ++i) {
		error = clGetDeviceInfo(deviceID, ulongInfo[i], sizeof(ulongBuffer), &ulongBuffer, NULL);
		CLErrorCheck(error, "clGetDeviceInfo", "get device info", CHECK_EXIT);
		printf("%s: %llu\n", ulongInfoNames[i], ulongBuffer);
	}
	printf("\n");

}

void CLPrintDevices(CLPlatform platform)
{
	CLInt error;
	CLUInt nDevices;
	CLDevice * devices;

	error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nDevices);
	CLErrorCheck(error, "clGetDeviceIDs", "get number of devices", CHECK_EXIT);

	devices = (CLDevice *)calloc(nDevices, sizeof(CLDevice));

	error = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL);
	CLErrorCheck(error, "clGetDeviceIDs", "get device IDs", CHECK_EXIT);

	for (CLInt i = 0; i < nDevices; ++i) {
		printf("DEVICE #%d\n", i);
		CLPrintDeviceInfo(devices[i]);
	}
}

void CLPrintStats(CLEvent start, CLEvent finish, CLUInt operations, CLStringConst name)
{
	CLFloat time = timeBetweenEventsNS(start, finish);
	CLFloat flops = operations / time;
	printf("%s: %0.4f - %0.4f GFlops/s\n", name, time, flops);
}

CLFloat timeBetweenEventsNS(CLEvent start, CLEvent finish)
{
	if (start == NULL || finish == NULL) {
		return 0.0f;
	}

	CLULong timeStart, timeEnd;
	clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, NULL);
	clGetEventProfilingInfo(finish, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, NULL);

	return timeEnd - timeStart;
}

CLFloat timeBetweenEventsMS(CLEvent start, CLEvent finish)
{
	return 1.0e-6 * timeBetweenEventsNS(start, finish);
}

CLFloat timeBetweenEventsS(CLEvent start, CLEvent finish)
{
	return 1.0e-9 * timeBetweenEventsNS(start, finish);
}


#pragma mark Error

void CLErrorCheck(CLInt error, CLStringConst function, CLStringConst message, CLInt needExit)
{
	if (error != CL_SUCCESS) {
		fprintf(stderr, "%s - %s - errorCode: %d - error: %s\n", function, message, error, getErrorName(error));

		if (needExit == CHECK_EXIT) {
			exit(error);
		}
	}
}


#pragma mark Select

CLPlatform CLSelectPlatform(CLInt platformIndex)
{
	if (platformIndex >= 0) {

		CLInt error;
		CLUInt nPlatforms;
		CLPlatform *platforms;

		error = clGetPlatformIDs(0, NULL, &nPlatforms);
		CLErrorCheck(error, "clGetPlatformIDs", "get nPlatforms", CHECK_EXIT);

		platforms = (CLPlatform *)calloc(nPlatforms, sizeof(CLPlatform));

		error = clGetPlatformIDs(nPlatforms, platforms, NULL);
		CLErrorCheck(error, "clGetPlatformIDs", "get platforms IDs", CHECK_EXIT);

		if (platformIndex < nPlatforms) {
			return platforms[platformIndex];
		}
	}
	return NULL;
}

CLDevice CLSelectDevice(CLPlatform platform, CLInt deviceIndex)
{
	if (platform && deviceIndex >= 0) {

		CLInt error;
		CLUInt nDevices;
		CLDevice *devices;

		error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &nDevices);
		CLErrorCheck(error, "clGetDeviceIDs", "get nDevices", CHECK_EXIT);

		devices = (CLDevice *)calloc(nDevices, sizeof(CLDevice));

		error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, nDevices, devices, NULL);
		CLErrorCheck(error, "clGetDeviceIDs", "get device IDs", CHECK_EXIT);


		if (deviceIndex < nDevices) {
			return devices[deviceIndex];
		}
	}
	return NULL;
}


#pragma mark Create

CLContext CLCreateContext(CLPlatform platform, CLDevice device)
{
	CLInt error;
	cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
	CLContext context = clCreateContext(contextProperties, 1, &device, NULL, NULL, &error);
	CLErrorCheck(error, "clCreateContext", "create context", CHECK_EXIT);
	return context;
}

CLQueue CLCreateQueue(CLContext context, CLDevice device)
{
	CLInt error;
	CLQueue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &error);
	CLErrorCheck(error, "clCreateCommandQueue", "create queue", CHECK_EXIT);
	return queue;
}

CLProgram CLCreateProgramWithMacro(CLContext context, CLDevice device, CLStringConst fileName, CLStringConst macro)
{
	CLInt error;
	CLProgram program;

	CLString buffer = (CLString)calloc(BUFFER_SIZE, sizeof(CLChar));
	time_t now = time(NULL);

	if (macro != NULL) {
		snprintf(buffer, BUFFER_SIZE-1, "//%s\n%s\n#include \"%s\"\n", ctime(&now), macro, fileName);
	} else {
		snprintf(buffer, BUFFER_SIZE-1, "//%s#include \"%s\"\n", ctime(&now), fileName);
	}

	debugLog("'codice':\n%s\n", buffer);
	CLStringConst ptrBuff = buffer;
	program = clCreateProgramWithSource(context, 1, &ptrBuff, NULL, &error);
	CLErrorCheck(error, "clCreateProgramWithSource", "create program", CHECK_EXIT);

	error = clBuildProgram(program, 1, &device, "-I. -Werror", NULL, NULL);
	CLErrorCheck(error, "clBuildProgram", "build program", CHECK_NOT_EXIT);

	error = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, BUFFER_SIZE, buffer, NULL);
	CLErrorCheck(error, "clGetProgramBuildInfo", "get program build info", CHECK_NOT_EXIT);

	debugLog("=== BUILD LOG ===\n"
			 "%s\n", buffer);

	return program;
}

CLProgram CLCreateProgram(CLContext context, CLDevice device, CLStringConst fileName)
{
	return CLCreateProgramWithMacro(context, device, fileName, NULL);
}

CLKernel CLCreateKernel(CLProgram program, CLStringConst name)
{
	CLInt error;
	CLKernel kernel;

	kernel = clCreateKernel(program, name, &error);
	CLErrorCheck(error, "clCreateKernel", name, CHECK_EXIT);

	return kernel;
}

CLMem CLCreateBufferHostVar(CLContext context, CLMemFlags flags, CLSize size, void * hostVar, CLStringConst name)
{
	CLInt error;
	CLMem var = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, size, hostVar, &error);
	CLErrorCheck(error, "clCreateBuffer", name, CHECK_EXIT);

	return var;
}

CLMem CLCreateBuffer(CLContext context, CLMemFlags flags, CLSize size, CLStringConst name)
{
	CLInt error;
	CLMem var = clCreateBuffer(context, flags, size, NULL, &error);
	CLErrorCheck(error, "clCreateBuffer", name, CHECK_EXIT);

	return var;
}

CLMem CLCreateSubBuffer(CLMem mem, CLMemFlags flags, CLSize offset, CLSize size, CLStringConst name)
{
	CLInt error;
	cl_buffer_region region = {offset, size};
	CLMem var = clCreateSubBuffer(mem, flags, CL_BUFFER_CREATE_TYPE_REGION, &region, &error);
	CLErrorCheck(error, "clCreateSubBuffer", name, CHECK_EXIT);
	return var;
}

#pragma mark Kernel Stuff

void CLSetKernelArg(CLKernel kernel, CLUInt index, CLSize size, const void * arg, CLStringConst name)
{
	CLInt error;
	error = clSetKernelArg(kernel, index, size, arg);
	CLErrorCheck(error, "clSetKernelArg", name, CHECK_NOT_EXIT);
}

void CLEnqueueNDRangeKernel(CLQueue queue, CLKernel kernel, const CLInt workDim, const CLSize * globalWorkOffset, const CLSize * globalWorkSize, const CLSize * localWorkSize, CLUInt numberOfEventsWaitList, const CLEvent * eventsWaitList, CLEvent * event, CLStringConst name)
{
	CLInt error;
	error = clEnqueueNDRangeKernel(queue, kernel, workDim, globalWorkOffset, globalWorkSize, localWorkSize, numberOfEventsWaitList, eventsWaitList, event);
	CLErrorCheck(error, "clEnqueueNDRangeKernel", name, CHECK_EXIT);
}

CLSize CLGetPreferredWorkGroupSizeMultiple(CLKernel kernel, CLDevice device, CLStringConst name)
{
	CLInt error;
	CLSize preferredWorkGroupSizeMultiple = 1;

	error = clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(preferredWorkGroupSizeMultiple), &preferredWorkGroupSizeMultiple, NULL);
	CLErrorCheck(error, "clGetKernelWorkGroupInfo", "CLGetPreferredWorkGroupSizeMultiple", CHECK_NOT_EXIT);
	return preferredWorkGroupSizeMultiple;
}

CLSize CLGetOptimalGlobalWorkItemsSize(CLSize numberOfElements, CLSize lws)
{
	return roundUpSize(numberOfElements, lws);
}


#pragma mark Queue

void CLWaitForEvent(CLEvent * event, CLStringConst name)
{
	CLInt error;
	error = clWaitForEvents(1, event);
	CLErrorCheck(error, "clWaitForEvents", name, CHECK_EXIT);
}

void * CLEnqueueReadBufferWithEvent(CLQueue queue, CLMem mem, CLSize size, CLEvent * event, CLStringConst name)
{
	CLInt error;
	void * buffer = malloc(size);
	error = clEnqueueReadBuffer(queue, mem, CL_TRUE, 0, size, buffer, 0, NULL, event);
	CLWaitForEvent(event, name);
	CLReleaseEvent(*event, name);
	CLErrorCheck(error, "clEnqueueReadBuffer", name, CHECK_EXIT);
	return buffer;
}

void * CLEnqueueReadBuffer(CLQueue queue, CLMem mem, CLSize size, CLStringConst name)
{
	CLEvent event;
	return CLEnqueueReadBufferWithEvent(queue, mem, size, &event, name);
}


#pragma mark Finish and Release

void CLFinish(CLQueue queue)
{
	CLInt error;
	error = clFinish(queue);
	CLErrorCheck(error, "clFinish", "", CHECK_EXIT);
}

void CLReleaseDevice(CLDevice device, CLStringConst name)
{
	CLInt error;
	error = clReleaseDevice(device);
	CLErrorCheck(error, "clReleaseDevice", name, CHECK_NOT_EXIT);
}

void CLReleaseContext(CLContext context, CLStringConst name)
{
	CLInt error;
	error = clReleaseContext(context);
	CLErrorCheck(error, "clReleaseContext", name, CHECK_NOT_EXIT);
}

void CLReleaseQueue(CLQueue queue, CLStringConst name)
{
	CLInt error;
	error = clReleaseCommandQueue(queue);
	CLErrorCheck(error, "clReleaseCommandQueue", name, CHECK_NOT_EXIT);
}

void CLReleaseProgram(CLProgram program, CLStringConst name)
{
	CLInt error;
	error = clReleaseProgram(program);
	CLErrorCheck(error, "clReleaseProgram", name, CHECK_NOT_EXIT);
}

void CLReleaseKernel(CLKernel kernel, CLStringConst name)
{
	CLInt error;
	error = clReleaseKernel(kernel);
	CLErrorCheck(error, "clReleaseKernel", name, CHECK_NOT_EXIT);
}

void CLReleaseMemObject(CLMem var, CLStringConst name)
{
	if (var == NULL) {
		fprintf(stderr, "Trying to free a NULL CLMem: %s\n", name);
		return;
	}
	CLInt error;
	error = clReleaseMemObject(var);
	CLErrorCheck(error, "clReleaseMemObject", name, CHECK_NOT_EXIT);
}

void CLReleaseEvent(CLEvent event, CLStringConst name)
{
	if (event == NULL) {
		fprintf(stderr, "Trying to free a NULL CLEvent: %s\n", name);
		return;
	}

	CLInt error;
	error = clReleaseEvent(event);
	CLErrorCheck(error, "clReleaseEvent", name, CHECK_NOT_EXIT);
}


#pragma mark Useful Stuff

CLSize roundUpSize(CLSize elements, CLSize multiple)
{
	return (elements + multiple - 1) / multiple * multiple;
}

CLUInt roundUpUInt(CLUInt elements, CLUInt multiple)
{
	return (elements + multiple - 1) / multiple * multiple;
}

CLSize divUpSize(CLSize elements, CLSize divider)
{
	return (elements + divider - 1) / divider;
}

CLUInt divUpUInt(CLUInt elements, CLUInt divider)
{
	return (elements + divider - 1) / divider;
}

CLDeviceType CLGetDeviceType(CLDevice device)
{
	CLInt error;
	CLDeviceType deviceType;

	error = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
	CLErrorCheck(error, "clGetDeviceInfo", "CLGetDeviceType", CHECK_EXIT);

	return deviceType;
}

CLString getErrorName(CLInt errorCode) {

	switch (errorCode) {
		case CL_DEVICE_NOT_FOUND:                 return "Device not found";
		case CL_DEVICE_NOT_AVAILABLE:             return "Device not available";
		case CL_COMPILER_NOT_AVAILABLE:           return "Compiler not available";
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "Memory object allocation failure";
		case CL_OUT_OF_RESOURCES:                 return "Out of resources";
		case CL_OUT_OF_HOST_MEMORY:               return "Out of host memory";
		case CL_PROFILING_INFO_NOT_AVAILABLE:     return "Profiling information not available";
		case CL_MEM_COPY_OVERLAP:                 return "Memory copy overlap";
		case CL_IMAGE_FORMAT_MISMATCH:            return "Image format mismatch";
		case CL_IMAGE_FORMAT_NOT_SUPPORTED:       return "Image format not supported";
		case CL_BUILD_PROGRAM_FAILURE:            return "Program build failure";
		case CL_MAP_FAILURE:                      return "Map failure";
		case CL_INVALID_VALUE:                    return "Invalid value";
		case CL_INVALID_DEVICE_TYPE:              return "Invalid device type";
		case CL_INVALID_PLATFORM:                 return "Invalid platform";
		case CL_INVALID_DEVICE:                   return "Invalid device";
		case CL_INVALID_CONTEXT:                  return "Invalid context";
		case CL_INVALID_QUEUE_PROPERTIES:         return "Invalid queue properties";
		case CL_INVALID_COMMAND_QUEUE:            return "Invalid command queue";
		case CL_INVALID_HOST_PTR:                 return "Invalid host pointer";
		case CL_INVALID_MEM_OBJECT:               return "Invalid memory object";
		case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return "Invalid image format descriptor";
		case CL_INVALID_IMAGE_SIZE:               return "Invalid image size";
		case CL_INVALID_SAMPLER:                  return "Invalid sampler";
		case CL_INVALID_BINARY:                   return "Invalid binary";
		case CL_INVALID_BUILD_OPTIONS:            return "Invalid build options";
		case CL_INVALID_PROGRAM:                  return "Invalid program";
		case CL_INVALID_PROGRAM_EXECUTABLE:       return "Invalid program executable";
		case CL_INVALID_KERNEL_NAME:              return "Invalid kernel name";
		case CL_INVALID_KERNEL_DEFINITION:        return "Invalid kernel definition";
		case CL_INVALID_KERNEL:                   return "Invalid kernel";
		case CL_INVALID_ARG_INDEX:                return "Invalid argument index";
		case CL_INVALID_ARG_VALUE:                return "Invalid argument value";
		case CL_INVALID_ARG_SIZE:                 return "Invalid argument size";
		case CL_INVALID_KERNEL_ARGS:              return "Invalid kernel arguments";
		case CL_INVALID_WORK_DIMENSION:           return "Invalid work dimensionsension";
		case CL_INVALID_WORK_GROUP_SIZE:          return "Invalid work group size";
		case CL_INVALID_WORK_ITEM_SIZE:           return "Invalid work item size";
		case CL_INVALID_GLOBAL_OFFSET:            return "Invalid global offset";
		case CL_INVALID_EVENT_WAIT_LIST:          return "Invalid event wait list";
		case CL_INVALID_EVENT:                    return "Invalid event";
		case CL_INVALID_OPERATION:                return "Invalid operation";
		case CL_INVALID_GL_OBJECT:                return "Invalid OpenGL object";
		case CL_INVALID_BUFFER_SIZE:              return "Invalid buffer size";
		case CL_INVALID_MIP_LEVEL:                return "Invalid mip-map level";
		case CL_INVALID_GLOBAL_WORK_SIZE:		  return "Invalid global work size";
	}
	return "";
}