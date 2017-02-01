CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native -Wno-unknown-programs

CLBLAS_PATH=/usr/local

uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)
	LDLIBS=-framework OpenCL

	CFLAGS+=-I $(CLBLAS_PATH)/include
	LDLIBS+=-L $(CLBLAS_PATH)/lib -l clBLAS

else
	LDLIBS=-lrt -lOpenCL -lbsd
endif



CLNet:
	$(CC) $(CFLAGS) *.c -o CLNet $(LDLIBS)

clean:
	rm CLNet