CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native -Wno-unknown-pragmas

CLBLAS_PATH=/usr/local

uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)
	LDLIBS=-framework OpenCL

	CFLAGS+=-I $(CLBLAS_PATH)/include
	LDLIBS+=-L $(CLBLAS_PATH)/lib -lclBLAS

else
	LDLIBS=-lrt -lOpenCL -lbsd -lclBLAS
	CFLAGS+= -D_DEFAULT_SOURCE
endif


#CLBLAS_PATH=/home/bilotta/src/clBLAS
#CFLAGS+=-I $(CLBLAS_PATH)/src
#LDLIBS+=-L $(CLBLAS_PATH)/build/library -lclBLAS -lm

CLNet:
	$(CC) $(CFLAGS) *.c -o CLNet $(LDLIBS)

clean:
	rm CLNet