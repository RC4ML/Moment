/**
 * pcie_bw.cu  --  Measure PCIe bandwidth by cudaMemcpy().
 *
 * Build:
 *   nvcc -O3 -std=c++17 pcie_bw.cu -o pcie_bw
 *
 * Run (root NOT required):
 *   ./pcie_bw 4096 2048
 *   ↑↑ 4096 MiB total copy, per-iteration 2 MiB
 *
 * Output:
 *   H2D: 23.5 GiB/s,  D2H: 24.1 GiB/s  (PCIe 4.0×16 ~ 25 GB/s)
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <fstream>

void check(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        exit(1);
    }
}

double gb_per_sec(size_t bytes, float ms)
{
    return static_cast<double>(bytes) / (1<<30) / (ms * 1e-3);
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: %s <totalMiB> <chunkMiB>\n", argv[0]);
        return 0;
    }
    size_t totalMiB  = std::atoll(argv[1]);
    size_t chunkMiB  = std::atoll(argv[2]);
    size_t totalBytes  = totalMiB  * (1ull << 20);
    size_t chunkBytes  = chunkMiB  * (1ull << 20);
    size_t iter = totalBytes / chunkBytes;

    printf("Total %.1f MiB, chunk %.1f MiB, iterations %zu\n",
           totalMiB * 1.0, chunkMiB * 1.0, iter);

    void *h_buf;
    void *d_buf;
    check(cudaMallocHost(&h_buf, chunkBytes), "cudaMallocHost");
    check(cudaMalloc   (&d_buf, chunkBytes), "cudaMalloc");
    memset(h_buf, 0xa5, chunkBytes);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    float ms_h2d = 0.0f, ms_d2h = 0.0f;

    // ------------ Host → Device ------------
    cudaEventRecord(t0);
    for (size_t i = 0; i < iter; ++i)
        check(cudaMemcpy(d_buf, h_buf, chunkBytes, cudaMemcpyHostToDevice),
              "H2D memcpy");
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_h2d, t0, t1);

    // ------------ Device → Host ------------
    cudaEventRecord(t0);
    for (size_t i = 0; i < iter; ++i)
        check(cudaMemcpy(h_buf, d_buf, chunkBytes, cudaMemcpyDeviceToHost),
              "D2H memcpy");
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_d2h, t0, t1);

    printf("Host→Device: %.2f GiB/s  |  Device→Host: %.2f GiB/s\n",
           gb_per_sec(totalBytes, ms_h2d), gb_per_sec(totalBytes, ms_d2h));
    
    std::ofstream ofs("bandwidth_results.txt", std::ios::app);
    if (!ofs) {
        std::fprintf(stderr, "Error: cannot open bandwidth_results.txt for writing\n");
        return 0;
    }
    ofs << gb_per_sec(totalBytes, ms_h2d) << "\n";
    ofs.close();
    
    cudaFreeHost(h_buf);
    cudaFree(d_buf);
    return 0;
}
