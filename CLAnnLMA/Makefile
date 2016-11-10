CFLAGS=-std=c99 -Wall -Wextra -g -O3 -march=native

CLBLAS=libclBLAS.2.6.0.dylib

uname_S := $(shell uname -s)

ifeq ($(uname_S),Darwin)
	LDLIBS=-framework OpenCL
else
	LDLIBS=-lrt -lOpenCL -lcrypto
endif

PDFCrackCL:
	$(CC) $(CFLAGS) $(CLBLAS) *.c -o PDFCrackCL $(LDLIBS)

clean:
	rm PDFCrackCL