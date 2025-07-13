NVCC = nvcc

all: test_ssd test_pcie

test_pcie: pcie_bw.cu 
	$(NVCC) $(CFLAGS) -o $@ $< -g

test_ssd: ssd_bw.cu common.cuh iostack.cuh ssdqp.cuh
	$(NVCC) $(CFLAGS) -o $@ $< -g

clean:
	rm -f test_ssd test_pcie
