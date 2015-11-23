ARGS = 32 32 32 32

all:
	nvcc -std=c++11 main.cu --compiler-options -Wall -Xcompiler -fopenmp -lgomp -lrt -o main

run:
	./main $(ARGS)

debug:
	gdb --args ./main $(ARGS)

clear:
	rm main