CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native -Wno-unknown-pragmas


uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)

	CLBLAS_PATH=/usr/local

	CFLAGS+=-I $(CLBLAS_PATH)/include
	LDLIBS=-framework OpenCL -L $(CLBLAS_PATH)/lib -lclBLAS

else

	CLBLAS_PATH=/home/studenti/Documents/clBLAS

	CFLAGS+= -D_DEFAULT_SOURCE -I $(CLBLAS_PATH)/src
	LDLIBS=-lrt -lOpenCL -lbsd -L $(CLBLAS_PATH)/build/library -lclBLAS -lm
endif


#CLBLAS_PATH=/home/bilotta/src/clBLAS
#CFLAGS+=-I $(CLBLAS_PATH)/src
#LDLIBS+=-L $(CLBLAS_PATH)/build/library -lclBLAS -lm

CLNet:
	$(CC) $(CFLAGS) *.c -o CLNet $(LDLIBS)

clean:
	rm CLNet