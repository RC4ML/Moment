#include "iostack.cuh"
#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>

// Configuration parameters (replace macros with variables)
constexpr size_t TEST_SIZE = 0x10000000;         // Test data size, in bytes
constexpr size_t APP_BUF_SIZE = 0x10000000;     // Application buffer size
constexpr int NUM_QUEUES_PER_SSD = 128;         // Number of queues per SSD

// Kernel to initialize application buffer
__global__ void fill_app_buf(uint64_t *app_buf) {
    for (int i = 0; i < TEST_SIZE / 8; i++) {
        app_buf[i] = 0;
    }
}

// Main function
int main(int argc, char** argv) {
    // Initialize IOStack
    int num_ssd = atoi(argv[1]);
    int io_size = atoi(argv[2]);
    IOStack iostack(num_ssd, NUM_QUEUES_PER_SSD, 1, 32);

    // Allocate application buffer
    uint64_t *app_buf;
    CHECK(cudaMalloc(&app_buf, APP_BUF_SIZE));
    fill_app_buf<<<1, 1>>>(app_buf);

    // Allocate and initialize request buffers
    int num_reqs = TEST_SIZE / io_size;
    IOReq *reqs;
    CHECK(cudaMalloc(&reqs, sizeof(IOReq) * num_reqs));
    IOReq *h_reqs;
    CHECK(cudaHostAlloc(&h_reqs, sizeof(IOReq) * num_reqs, cudaHostAllocMapped));

    std::unordered_set<uint64_t> lbs;
    srand(time(NULL));

    int percent = 1;
    clock_t clstart = clock();
    
    // Generate test requests
    for (int i = 0; i < num_reqs; i++) {
        uint64_t lb;
        do {
            uint64_t idx = (((unsigned long)rand() << 31) | rand());
            lb = (idx % num_ssd) * (NUM_LBS_PER_SSD / MAX_ITEMS) + idx % (NUM_LBS_PER_SSD / MAX_ITEMS);
        } while (lbs.find(lb) != lbs.end());
        lbs.insert(lb);
        
        h_reqs[i].start_lb = lb * MAX_ITEMS;
        h_reqs[i].num_items = MAX_ITEMS;
        for (int j = 0; j < MAX_ITEMS; j++) {
            h_reqs[i].dest_addr[j] = (uint64_t)(app_buf + (1ll * i * io_size + j * LBS) % APP_BUF_SIZE / sizeof(uint64_t));
        }

        if (i >= num_reqs / 100 * percent) {
            double eta = (clock() - clstart) / (double)CLOCKS_PER_SEC / percent * (100 - percent);
            fprintf(stderr, "generating test data: %d%% done, eta %.0lfs\r", percent, eta);
            percent++;
        }
    }
    CHECK(cudaDeviceSynchronize());

    // Copy requests to device memory
    CHECK(cudaMemcpy(reqs, h_reqs, sizeof(IOReq) * num_reqs, cudaMemcpyHostToDevice));

    // Run IO requests multiple times and measure performance
    int repeat = 10;
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));

    fprintf(stderr, "starting do_io_req...\n");
    for (int i = 0; i < repeat; i++) {
        iostack.io_submission(reqs, 10240, 0);
        iostack.io_submission(reqs + 10240, 10240, 0);
        iostack.io_submission(reqs + 20480, 10240, 0);
        iostack.io_submission(reqs + 30720, num_reqs - 30720, 0);
        iostack.io_completion(0);
    }

    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Output performance results
    fprintf(stderr, "do_io_req takes %f ms\n", ms);
    fprintf(stderr, "%dB random read bandwidth: %f MiB/s\n", io_size, TEST_SIZE * repeat / (1024.0 * 1024.0) / (ms / 1000));
    double read_bw_mib_s = TEST_SIZE * repeat
                           / (1024.0 * 1024.0)
                           / (ms / 1000.0);
    std::ofstream ofs("bandwidth_results.txt", std::ios::out);
    if (!ofs) {
        std::fprintf(stderr, "Error: cannot open bandwidth_results.txt for writing\n");
        return 0;
    }
    ofs << read_bw_mib_s/1000.0 << "\n";
    ofs.close();
    return 0;
}
