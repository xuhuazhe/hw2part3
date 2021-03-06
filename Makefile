# Load CUDA using the following command
# module load cuda

#
# Stampede
#
CC = nvcc
MPCC = nvcc
OPENMP =
CFLAGS = -O3 -arch=sm_35
NVCCFLAGS = -D_FORCE_INLINES -O3 -arch=sm_35
LIBS = -lm

TARGETS = serial gpu gpu_part3 autograder

all:	$(TARGETS)

serial: serial.o common.o
	$(CC) -o $@ $(LIBS) serial.o common.o
gpu: gpu.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu.o common.o
gpu_part3: gpu_part3.o common.o
	$(CC) -o $@ $(NVCCLIBS) gpu_part3.o common.o
autograder: autograder.o common.o
	$(CC) -o $@ $(NVCCLIBS) autograder.o common.o

serial.o: serial.cpp common.h
	$(CC) -c $(CFLAGS) serial.cpp
gpu.o: gpu.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu.cu
gpu_part3.o: gpu_part3.cu common.h
	$(CC) -c $(NVCCFLAGS) gpu_part3.cu
autograder.o: autograder.cu common.h
	$(CC) -c $(NVCCFLAGS) autograder.cu
common.o: common.cu common.h
	$(CC) -c $(CFLAGS) common.cu

clean:
	rm -f *.o $(TARGETS)
