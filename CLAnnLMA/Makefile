CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native -Wno-unknown-pragmas

CLBLAS_PATH=/usr/local

uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)
	LDLIBS=-framework OpenCL

	CFLAGS+=-I $(CLBLAS_PATH)/include
	LDLIBS+=-L $(CLBLAS_PATH)/lib -lclBLAS

else
	LDLIBS=-lrt -lOpenCL -lbsd -lclBLAS
endif



CLNet:
	$(CC) $(CFLAGS) *.c -o CLNet $(LDLIBS)

clean:
	rm CLNet