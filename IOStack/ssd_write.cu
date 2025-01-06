#include "iostack_decouple.cuh"
#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>
#include <iostream>
#define TEST_SIZE 0x17D7840000
#define NUM_QUEUES_PER_SSD 128
#define NUM_SSDS 6

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }


__device__ float **IO_buf_base;

__device__ uint64_t seed;
__global__ void gen_test_data(int ssd_id, int64_t req_id, int block_id)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        // seed = seed * 0x5deece66d + 0xb;
        IO_buf_base[ssd_id][i] = req_id;
        // if(i % (ITEM_SIZE / 4) == 0){
        //     IO_buf_base[ssd_id][i] = block_id * 8 + (i / (ITEM_SIZE / 4));//i + block_id * MAX_IO_SIZE / 4;//req_id;// * MAX_IO_SIZE / 8 + i;
        // }else{
        //     IO_buf_base[ssd_id][i] = i%(ITEM_SIZE / 4);
        // }
    }
}

__global__ void gen_feat_data(int ssd_id, int block_id, float* feature){
    for (int64_t i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        IO_buf_base[ssd_id][i] = feature[i + int64_t((MAX_IO_SIZE / 4))*block_id];
    }
}

__global__ void check_test_data(float *app_buf, int idx)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        seed = seed * 0x5deece66d + 0xb;
        if (app_buf[i] != idx * MAX_IO_SIZE / 4 + i)
        {
            printf("check failed at block %d, i = %d, read %lx, expected %x\n", idx, i, app_buf[i], idx * MAX_IO_SIZE / 8 + i);
            assert(0);
        }
    }
}

__global__ void fill_app_buf(float *app_buf)
{
    for (int i = 0; i < TEST_SIZE / 4; i++)
        app_buf[i] = 0;
}

void mmap_features_read(std::string &features_file, float* features){
    int64_t n_idx = 0;
    int32_t fd = open(features_file.c_str(), O_RDONLY);
    if(fd == -1){
        std::cout<<"cannout open file: "<<features_file<<"\n";
    }
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const float *buf = (float *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const float* buf_end = buf + buf_len/sizeof(float);
    float temp;
    while(buf < buf_end){
        temp = *buf;
        features[n_idx++] = temp;
        buf++;
    }
    close(fd);
    return;
}

int main(int argc, char** argv)
{
    // int32_t node_num = 111059956;
    // int32_t nf = 128;
    // float* host_float_feature;
    // cudaHostAlloc(&host_float_feature, int64_t(int64_t(int64_t(node_num) * nf) * sizeof(float)), cudaHostAllocMapped);
    // cudaCheckError();

    // std::string features_path = "/home/sunjie/dataset/paper100M/features";

    // std::cout<<features_path<<"\n";

    // mmap_features_read(features_path, host_float_feature);

    IOStack iostack(NUM_SSDS, NUM_QUEUES_PER_SSD);
    float **d_IO_buf_base = (float**)iostack.get_d_io_buf_base();
    CHECK(cudaMemcpyToSymbol(IO_buf_base, &d_IO_buf_base, sizeof(float **)));

    int64_t num_reqs = uint64_t(NUM_LBS_PER_SSD) * NUM_SSDS / MAX_ITEMS;// 111059956 / 8;
    printf("req num %ld\n", num_reqs);

    std::unordered_set<uint64_t> lbs;

    int percent = 1;
    clock_t clstart = clock();
    cudaCheckError();

    for (int64_t i = 0; i < num_reqs; i++)
    {
        uint64_t lb;
        lb = i;

        int ssd_id = lb * MAX_ITEMS / NUM_LBS_PER_SSD;

        // gen_feat_data<<<1,1>>>(ssd_id, i, host_float_feature);
        gen_test_data<<<1, 1>>>(ssd_id, i, lb);
        // cudaCheckError();

        iostack.write_data(ssd_id, (lb * MAX_ITEMS) % NUM_LBS_PER_SSD, MAX_IO_SIZE / LB_SIZE);
        cudaCheckError();

        if(i % 10000000 == 0){
            printf("req %lu\n", i);
        }
        if (i >= num_reqs / 1000 * percent)
        {
            double eta = (clock() - clstart) / (double)CLOCKS_PER_SEC / percent * (1000 - percent);
            fprintf(stderr, "generating test data: %d%% done, eta %.0lfs\r", percent/10, eta);
            percent++;
        }
        cudaCheckError();
    }
    CHECK(cudaDeviceSynchronize());
    std::cout<<"Finish Writing SSD\n";
}