CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native

CLBLAS_PATH=/usr/local
#CLBLAS=libclBLAS.2.11.0.dylib

uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)
	LDLIBS=-framework OpenCL
else
	LDLIBS=-lrt -lOpenCL
endif

CFLAGS+=-I $(CLBLAS_PATH)/include
LDLIBS+=-L $(CLBLAS_PATH)/lib -l clBLAS

CLNet:
	$(CC) $(CFLAGS) *.c -o CLNet $(LDLIBS)

clean:
	rm CLNet