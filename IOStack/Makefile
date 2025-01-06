NVCC = nvcc

all: test

test: test.cu common.cuh iostack.cuh ssdqp.cuh
	$(NVCC) $(CFLAGS) -o $@ $< -g

clean:
	rm -f test decouple
