#pragma once
#include "ssdqp.cuh"
#include "system_config.cuh"
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <cstdint>
#define _CUDA
#include "ioctl.h"
#include <sys/ioctl.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <map>

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                             \
    {                                                                \
        cudaError_t e = cudaGetLastError();                          \
        if (e != cudaSuccess)                                        \
        {                                                            \
            printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                           \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

uint64_t sdiv (uint64_t a, uint64_t b) {
    return (a+b-1)/b;
}

typedef uint64_t app_addr_t;
struct IOReq
{
    uint64_t start_lb;
    app_addr_t dest_addr[MAX_ITEMS];
    int num_items;

    __forceinline__ __host__ __device__ IOReq(){};

    __forceinline__
        __host__ __device__
        IOReq(uint64_t ssd, uint64_t length) : start_lb(ssd), num_items(length)
    {
        for (int i = 0; i < num_items; i++)
            dest_addr[i] = ~0ULL;
    }

    __host__ __device__ bool operator<(const IOReq &lhs) const
    {
        return this->start_lb < lhs.start_lb;
    }

    __forceinline__
        __host__ __device__ ~IOReq()
    {
        // delete[] gpu_addr;
    }
};

__device__ int req_id_to_ssd_id(int req_id, int num_ssds, int *ssd_num_reqs_prefix_sum)
{
    int ssd_id = 0;
    for (; ssd_id < num_ssds; ssd_id++)
        if (ssd_num_reqs_prefix_sum[ssd_id] > req_id)
            break;
    return ssd_id;
}

__global__ static void submit_io_req_kernel(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, int *ssd_num_reqs_prefix_sum)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);
        int queue_pos = (ssdqp[global_queue_id].sq_tail + id_in_queue) % QUEUE_DEPTH;
        // sq真实的物理地址
        uint64_t io_addr = prp1[ssd_id] + queue_id * QUEUE_IOBUF_SIZE + queue_pos * MAX_IO_SIZE; // assume contiguous!
        uint64_t io_addr2 = io_addr / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
        if (reqs[i].num_items * ITEM_SIZE > HOST_PGSZ * 2)
        {
            int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
            uint64_t offset = queue_id * PRP_SIZE * QUEUE_DEPTH + queue_pos * PRP_SIZE;
            io_addr2 = prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ + offset / HOST_PGSZ] + offset % HOST_PGSZ;
        }
        // printf("submitting device %d: req %d ssd %d queue %d queue_pos %d io_addr %lx io_addr2 %lx\n", deviceId, ssd_id, queue_id, queue_pos, io_addr, io_addr2);
        ssdqp[global_queue_id].fill_sq(
            ssdqp[global_queue_id].cmd_id + id_in_queue,                   // command id
            queue_pos,                                                     // position in SQ
            OPCODE_READ,                                                   // opcode
            io_addr,                                                       // prp1
            io_addr2,                                                      // prp2
            reqs[i].start_lb & 0xffffffff,                                 // start lb low
            (reqs[i].start_lb >> 32) & 0xffffffff,                         // start lb high
            RW_RETRY_MASK | (reqs[i].num_items * ITEM_SIZE / LB_SIZE - 1), // number of LBs
            i                                                              // req id
        );
    }
}

__global__ static void ring_sq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int32_t* num_reqs)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);
        // 判断==0的原因？
        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (QUEUE_DEPTH - 1);
            if (cnt > QUEUE_DEPTH - 1)
                cnt = QUEUE_DEPTH - 1;
            ssdqp[global_queue_id].cmd_id += cnt;
            ssdqp[global_queue_id].sq_tail = (ssdqp[global_queue_id].sq_tail + cnt) % QUEUE_DEPTH;
            // printf("thread %d ssd %d queue %d end req %d cnt %d\n", thread_id, ssd_id, queue_id, ssd_num_reqs_prefix_sum[ssd_id], cnt);
            *ssdqp[global_queue_id].sqtdbl = ssdqp[global_queue_id].sq_tail;
        }
    }
}

/*
__global__ static void poll_io_req_kernel(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, int* ssd_num_reqs_prefix_sum)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int ssd_id = req_id_to_ssd_id(thread_id, num_ssds, ssd_num_reqs_prefix_sum);
    if (ssd_id >= num_ssds)
        return;
    int req_offset = thread_id - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
    int queue_id = req_offset / (QUEUE_DEPTH - 1);
    assert(queue_id < num_queues_per_ssd);
    int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
    int complete_id = ssdqp[global_queue_id].num_completed + req_offset % (QUEUE_DEPTH - 1);
    int queue_pos = complete_id % QUEUE_DEPTH;

    uint32_t current_phase = (complete_id / QUEUE_DEPTH) & 1;
    while (((ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & PHASE_MASK) >> 16) == current_phase)
        ;
    uint32_t status = ssdqp[global_queue_id].cq[queue_pos * 4 + 3];
    uint32_t cmd_id = status & CID_MASK;
    if ((status >> 17) & SC_MASK)
    {
        printf("thread %d cq[%d] status: 0x%x, cid: %d\n", thread_id, queue_pos, (status >> 17) & SC_MASK, cmd_id);
        assert(0);
    }
}
*/

__global__ static void copy_io_req_kernel(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2, int *ssd_num_reqs_prefix_sum)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARP_SIZE;
    int lane_id = thread_id % WARP_SIZE;
    int num_warps = blockDim.x * gridDim.x / WARP_SIZE;
    for (int i = warp_id; i < num_reqs[0]; i += num_warps)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);
        int complete_id = ssdqp[global_queue_id].num_completed + id_in_queue;
        int queue_pos = complete_id % QUEUE_DEPTH;

        if (lane_id == 0)
        {
            // printf("polling req %d ssd %d queue %d complete_id %d queue_pos %d num_completed %d\n", i, ssd_id, queue_id, complete_id, queue_pos, ssdqp[global_queue_id].num_completed);
            uint32_t current_phase = (complete_id / QUEUE_DEPTH) & 1;
            while (((ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & PHASE_MASK) >> 16) == current_phase)
                ;
            uint32_t status = ssdqp[global_queue_id].cq[queue_pos * 4 + 3];
            uint32_t cmd_id = status & CID_MASK;
            if ((status >> 17) & SC_MASK)
            {
                printf("thread %d cq[%d] status: 0x%x, cid: %d\n", thread_id, queue_pos, (status >> 17) & SC_MASK, cmd_id);
                assert(0);
            }
        }
        int cmd_id = ssdqp[global_queue_id].cq[queue_pos * 4 + 3] & CID_MASK;
        int req_id = ssdqp[global_queue_id].cmd_id_to_req_id[cmd_id % QUEUE_DEPTH];
        int sq_pos = ssdqp[global_queue_id].cmd_id_to_sq_pos[cmd_id % QUEUE_DEPTH];

        // 此时数据已经在IO_buf_base中了
        for (int j = 0; j < reqs[req_id].num_items; j++)
        {
            // printf("reqs[req_id].dest_addr[j]): %ld\n",reqs[req_id].dest_addr[j]);
            for (int k = lane_id; k < ITEM_SIZE / 8; k += WARP_SIZE)
            {
                
                ((uint64_t *)reqs[req_id].dest_addr[j])[k] = IO_buf_base[ssd_id][queue_id * QUEUE_DEPTH * MAX_IO_SIZE / 8 + sq_pos * MAX_IO_SIZE / 8 + j * ITEM_SIZE / 8 + k];
                // *(((uint64_t *)reqs[req_id].dest_addr[j]) + k) = IO_buf_base[ssd_id][queue_id * QUEUE_DEPTH * MAX_IO_SIZE / 8 + sq_pos * MAX_IO_SIZE / 8 + j * ITEM_SIZE / 8 + k];
                // printf("ld\n",IO_buf_base[ssd_id][queue_id * QUEUE_DEPTH * MAX_IO_SIZE / 8 + sq_pos * MAX_IO_SIZE / 8 + j * ITEM_SIZE / 8 + k]);
            }
        }
    }
    
}

__global__ static void ring_cq_doorbell_kernel(int num_ssds, int num_queues_per_ssd, SSDQueuePair *ssdqp, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum, int32_t* num_reqs)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = thread_id; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = req_id_to_ssd_id(i, num_ssds, ssd_num_reqs_prefix_sum);
        if (ssd_id >= num_ssds)
            break;
        int req_offset = i - (ssd_id == 0 ? 0 : ssd_num_reqs_prefix_sum[ssd_id - 1]);
        int queue_id = req_offset / (QUEUE_DEPTH - 1);
        assert(queue_id < num_queues_per_ssd);
        int global_queue_id = ssd_id * num_queues_per_ssd + queue_id;
        int id_in_queue = req_offset % (QUEUE_DEPTH - 1);

        if (id_in_queue == 0)
        {
            int cnt = ssd_num_reqs[ssd_id] - queue_id * (QUEUE_DEPTH - 1);
            if (cnt > QUEUE_DEPTH - 1)
                cnt = QUEUE_DEPTH - 1;
            ssdqp[global_queue_id].num_completed += cnt;
            ssdqp[global_queue_id].cq_head = (ssdqp[global_queue_id].cq_head + cnt) % QUEUE_DEPTH;
            *ssdqp[global_queue_id].cqhdbl = ssdqp[global_queue_id].cq_head;
            // printf("queue %d num_completed %d cq_head %d\n", global_queue_id, ssdqp[global_queue_id].num_completed, ssdqp[global_queue_id].cq_head);
        }
    }
}

__global__ static void rw_data_kernel(uint32_t opcode, int ssd_id, uint64_t start_lb, uint64_t num_lb, int num_queues_per_ssd, SSDQueuePair *ssdqp, uint64_t *prp1, uint64_t **IO_buf_base, uint64_t *prp2)
{
    uint32_t cid;
    int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
    ssdqp[ssd_id * num_queues_per_ssd].submit(cid, opcode, prp1[ssd_id], LB_SIZE * num_lb <= HOST_PGSZ * 2 ? prp1[ssd_id] + HOST_PGSZ : prp2[ssd_id * prp_size_per_ssd / HOST_PGSZ], start_lb & 0xffffffff, (start_lb >> 32) & 0xffffffff, RW_RETRY_MASK | (num_lb - 1));
    uint32_t status;
    ssdqp[ssd_id * num_queues_per_ssd].poll(status, cid);
    if (status != 0)
    {
        printf("read/write failed with status 0x%x\n", status);
        assert(0);
    }
}

// 给每个ssd上填充上对应的reqs数量
__global__ static void preprocess_io_req_1(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, int *ssd_num_reqs)
{
    // printf("%lx, %d, %d, %d, %lx\n",reqs, num_reqs, num_ssds, num_queues_per_ssd, ssd_num_reqs);
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        // assert(ssd_id < num_ssds);
        if (ssd_id < num_ssds && ssd_id >= 0){
            atomicAdd(&ssd_num_reqs[ssd_id], 1);
        }
        else
        {
            printf("ssd_id: %d\n", ssd_id);
        }
    }
}

__global__ static void preprocess_io_req_2(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, int *ssd_num_reqs, int *ssd_num_reqs_prefix_sum)
{
    for (int i = 0; i < num_ssds; i++)
    {
        // assert(ssd_num_reqs[i] <= num_queues_per_ssd * (QUEUE_DEPTH - 1));
        if (ssd_num_reqs[i] > num_queues_per_ssd * (QUEUE_DEPTH - 1))
        {
            printf("ssd_num_reqs[%d]: %d\n", i, ssd_num_reqs[i]);
        }
        ssd_num_reqs_prefix_sum[i] = ssd_num_reqs[i];
        if (i > 0)
            ssd_num_reqs_prefix_sum[i] += ssd_num_reqs_prefix_sum[i - 1];
    }
}

__device__ int req_ids[MAX_SSDS_SUPPORTED];
__global__ static void distribute_io_req_1(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs, int *ssd_num_reqs_prefix_sum)
{
    for (int i = 0; i < num_ssds; i++)
    {
        req_ids[i] = i ? ssd_num_reqs_prefix_sum[i - 1] : 0;
        // printf("req_ids[i]: %d\n", req_ids[i]);
    }
}

__global__ static void distribute_io_req_2(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int num_threads = blockDim.x * gridDim.x;
    for (int i = tid; i < num_reqs[0]; i += num_threads)
    {
        int ssd_id = reqs[i].start_lb / NUM_LBS_PER_SSD;
        // assert(ssd_id < num_ssds);
        if (ssd_id < num_ssds && ssd_id >= 0)
        {
            // 获取当前ssd的第一个request id
            int req_id = atomicAdd(&req_ids[ssd_id], 1);
            distributed_reqs[req_id] = reqs[i];
            distributed_reqs[req_id].start_lb %= NUM_LBS_PER_SSD;
        }
    }
}

__global__ static void distribute_io_req_3(IOReq *reqs, int32_t* num_reqs, int num_ssds, int num_queues_per_ssd, IOReq *distributed_reqs, int *ssd_num_reqs_prefix_sum)
{
    
    for (int i = 0; i < num_ssds; i++)
    {
        if (req_ids[i] != ssd_num_reqs_prefix_sum[i])
        {
            printf("req id %d %d\n", req_ids[i], ssd_num_reqs_prefix_sum[i]);
        }
        assert(req_ids[i] == ssd_num_reqs_prefix_sum[i]);
    }
}

class IOStack
{
public:
    int num_ssds_;
    int num_queues_per_ssd_;
    int num_queues_per_ssd_per_gpu;
    int nums_gpus;
    SSDQueuePair **d_ssdqp_;
    // 每个ssd都有一个uint64_t的指针指向一个4k的page,n个ssd就有n个指针。
    uint64_t **d_prp1_;
    // 所有ssd中所有queue可以装满的情况下，需要x个4k的page对应的uint64_t的指针。
    uint64_t **d_prp2_;
    uint64_t ***d_IO_buf_base_;
    uint64_t ***h_IO_buf_base_;

    void *reg_ptr_;
    void *h_admin_queue_;

    void **d_io_buf_;
    void **d_io_queue_;

    IOReq **distributed_reqs_;
    int **ssd_num_reqs_;
    int **ssd_num_reqs_prefix_sum_;

    std::map<int, uint64_t **> d_IO_buf_base_map;
    std::map<int, uint64_t **> h_IO_buf_base_map;

    IOStack(int num_ssds, int num_queues_per_ssd, int nums_gpus) : num_ssds_(num_ssds), num_queues_per_ssd_(num_queues_per_ssd), nums_gpus(nums_gpus)
    {
        num_queues_per_ssd_per_gpu = num_queues_per_ssd_ / nums_gpus;
        // printf("nums_gpu: %d\tnum_queues_per_ssd_per_gpu: %d\n", nums_gpus,num_queues_per_ssd_per_gpu);
        d_ssdqp_ = (SSDQueuePair **)malloc(sizeof(SSDQueuePair *) * nums_gpus);
        d_prp1_ = (uint64_t **)malloc(sizeof(uint64_t *) * nums_gpus);
        d_prp2_ = (uint64_t **)malloc(sizeof(uint64_t *) * nums_gpus);

        d_IO_buf_base_ = (uint64_t ***)malloc(sizeof(uint64_t **) * nums_gpus);
        h_IO_buf_base_ = (uint64_t ***)malloc(sizeof(uint64_t **) * nums_gpus);

        d_io_buf_ = (void **)malloc(sizeof(void *) * nums_gpus);
        d_io_queue_ = (void **)malloc(sizeof(void *) * nums_gpus);

        distributed_reqs_ = (IOReq **)malloc(sizeof(IOReq *) * nums_gpus);
        ssd_num_reqs_ = (int **)malloc(sizeof(int *) * nums_gpus);
        ssd_num_reqs_prefix_sum_ = (int **)malloc(sizeof(int *) * nums_gpus);

        for (int gpu_id = 0; gpu_id < nums_gpus; ++gpu_id)
        {
            CHECK(cudaSetDevice(gpu_id));
            CHECK(cudaMalloc(&d_ssdqp_[gpu_id], num_ssds_ * num_queues_per_ssd_per_gpu * sizeof(SSDQueuePair)));
            CHECK(cudaMalloc(&d_prp1_[gpu_id], num_ssds_ * sizeof(uint64_t)));
            int prp_size_per_ssd = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd_per_gpu / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ;
            CHECK(cudaMalloc(&d_prp2_[gpu_id], num_ssds_ * prp_size_per_ssd / HOST_PGSZ * sizeof(uint64_t)));

            CHECK(cudaMalloc(&d_IO_buf_base_[gpu_id], num_ssds_ * sizeof(uint64_t *)));
            h_IO_buf_base_[gpu_id] = (uint64_t **)malloc(sizeof(uint64_t *) * num_ssds_);
            CHECK(cudaMalloc(&distributed_reqs_[gpu_id], 4000000 * sizeof(IOReq)));
            CHECK(cudaMalloc(&ssd_num_reqs_[gpu_id], MAX_SSDS_SUPPORTED * sizeof(int)));
            CHECK(cudaMalloc(&ssd_num_reqs_prefix_sum_[gpu_id], MAX_SSDS_SUPPORTED * sizeof(int)));
        }

        // std::vector<int> physsd_ids = {0,1,2};
        // for(int i = 0; i < num_ssds; i++)
        //     init_ssd_per_gpu(physsd_ids[i], i);
        for (int ssd_id = 0; ssd_id < num_ssds; ssd_id++)
            init_ssd_per_gpu(ssd_id);
        // init_ssd_per_gpu(0);
    }

    ~IOStack() {
        // 释放 GPU 资源
        for (int gpu_id = 0; gpu_id < nums_gpus; ++gpu_id) {
            cudaSetDevice(gpu_id); 
            if (d_ssdqp_[gpu_id] != nullptr) {
                cudaFree(d_ssdqp_[gpu_id]);
            }
            if (d_prp1_[gpu_id] != nullptr) {
                cudaFree(d_prp1_[gpu_id]);
            }
            if (d_prp2_[gpu_id] != nullptr) {
                cudaFree(d_prp2_[gpu_id]);
            }
            if (d_IO_buf_base_[gpu_id] != nullptr) {
                cudaFree(d_IO_buf_base_[gpu_id]);
            }
            if (distributed_reqs_[gpu_id] != nullptr) {
                cudaFree(distributed_reqs_[gpu_id]);
            }
            if (ssd_num_reqs_[gpu_id] != nullptr) {
                cudaFree(ssd_num_reqs_[gpu_id]);
            }
            if (ssd_num_reqs_prefix_sum_[gpu_id] != nullptr) {
                cudaFree(ssd_num_reqs_prefix_sum_[gpu_id]);
            }
        }
        // // 释放 CPU 资源
        for (int gpu_id = 0; gpu_id < nums_gpus; ++gpu_id) {
            if (h_IO_buf_base_[gpu_id] != nullptr) {
                free(h_IO_buf_base_[gpu_id]);
            }
        }
        // // 释放其他动态分配的资源
        free(d_ssdqp_);
        free(d_prp1_);
        free(d_prp2_);
        free(d_IO_buf_base_);
        free(h_IO_buf_base_);
        free(d_io_buf_);
        free(d_io_queue_);
        free(distributed_reqs_);
        free(ssd_num_reqs_);
        free(ssd_num_reqs_prefix_sum_);
        // 释放可能映射的内存
        if (reg_ptr_ != nullptr) {
            munmap(reg_ptr_, REG_SIZE);
            reg_ptr_ = nullptr;
        }
        if (h_admin_queue_ != nullptr) {
            free(h_admin_queue_);
            h_admin_queue_ = nullptr;
        }
        std::cerr << "IOStack resources have been freed." << std::endl;
    }

    void submit_io_req(IOReq* reqs, int32_t* num_reqs, int gpu_id, cudaStream_t stream)
    {
        // int reqs_per_gpu = sdiv(num_reqs[0], nums_gpus);
        // printf("reqs_per_gpu: %d\n", reqs_per_gpu);
 
        cudaSetDevice(gpu_id);
        cudaMemsetAsync(ssd_num_reqs_[gpu_id], 0, sizeof(int) * num_ssds_, stream);
        cudaCheckError();
        cudaMemsetAsync(ssd_num_reqs_prefix_sum_[gpu_id], 0, sizeof(int) * num_ssds_, stream);
        cudaCheckError();

        // 每个ssd上装填request数量
        // uint64_t current_reqs = (gpu_id + 1) * reqs_per_gpu > num_reqs[0] ? num_reqs[0] - gpu_id * reqs_per_gpu : reqs_per_gpu;
        // printf("gpu_id: %d, current_reqs: %ld\n", gpu_id, current_reqs);

        preprocess_io_req_1<<<32, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, ssd_num_reqs_[gpu_id]);
        cudaCheckError();

        // 装填每块ssd requests的前缀和 += ssd_num_reqs_prefix_sum[i - 1];
        preprocess_io_req_2<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, ssd_num_reqs_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id]);
        cudaCheckError();

        // 装填req_ids[i],指的是当前ssd前面所有ssd的request数量总和
        distribute_io_req_1<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, distributed_reqs_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id]);
        cudaCheckError();

        distribute_io_req_2<<<32, NUM_THREADS_PER_BLOCK, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, distributed_reqs_[gpu_id]);
        cudaCheckError();

        distribute_io_req_3<<<1, 1, 0, stream>>>(reqs, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, distributed_reqs_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id]);
        cudaCheckError();

        int num_blocks = 32;
        submit_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs_[gpu_id], num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], d_prp1_[gpu_id], d_IO_buf_base_[gpu_id], d_prp2_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id]);
        cudaCheckError();

        // fprintf(stderr, "submit_io_req_kernel done\n");
        ring_sq_doorbell_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(num_ssds_, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], ssd_num_reqs_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id], num_reqs);
        cudaCheckError();
        // fprintf(stderr, "ring_sq_doorbell_kernel done\n");
        // CHECK(cudaDeviceSynchronize());
        // poll_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs_, num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, d_ssdqp_, d_prp1_, d_IO_buf_base_, d_prp2_);
        // CHECK(cudaDeviceSynchronize());
        /////////////////////////////////potential risk!!!!!!!!!////////////////////////////////////////////////////

        copy_io_req_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(distributed_reqs_[gpu_id], num_reqs, num_ssds_, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], d_prp1_[gpu_id], d_IO_buf_base_[gpu_id], d_prp2_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id]);
        cudaCheckError();

        ring_cq_doorbell_kernel<<<num_blocks, NUM_THREADS_PER_BLOCK, 0, stream>>>(num_ssds_, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], ssd_num_reqs_[gpu_id], ssd_num_reqs_prefix_sum_[gpu_id], num_reqs);
        cudaCheckError();
        // printf("dev_id: %d\tring_cq_doorbell_kernel done\n",gpu_id);
        // cudaDeviceSynchronize();
        
    }

    void read_data(int ssd_id, uint64_t start_lb, uint64_t num_lb, int gpu_id)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_READ, ssd_id, start_lb, num_lb, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], d_prp1_[gpu_id], d_IO_buf_base_[gpu_id], d_prp2_[gpu_id]);
    }

    void write_data(int ssd_id, uint64_t start_lb, uint64_t num_lb, int gpu_id)
    {
        rw_data_kernel<<<1, 1>>>(OPCODE_WRITE, ssd_id, start_lb, num_lb, num_queues_per_ssd_per_gpu, d_ssdqp_[gpu_id], d_prp1_[gpu_id], d_IO_buf_base_[gpu_id], d_prp2_[gpu_id]);
    }

    uint64_t ***get_d_io_buf_base()
    {
        return d_IO_buf_base_;
    }

    uint64_t ***get_h_io_buf_base()
    {
        return h_IO_buf_base_;
    }

    void init_ssd_per_gpu(int ssd_id)
    {
        // open file and map BAR
        fprintf(stderr, "setting up SSD %d\n", ssd_id);
        char fname[20];
        // if(ssd_id == 0){
        //     sprintf(fname, "/dev/libnvm7");
        // }else{
        //     sprintf(fname, "/dev/libnvm%d", ssd_id-1);
        // }
        sprintf(fname, "/dev/libnvm%d", ssd_id);
        
        int fd = open(fname, O_RDWR);
        if (fd < 0)
        {
            fprintf(stderr, "Failed to open: %s\n", strerror(errno));
            exit(1);
        }
        // 返回SSD寄存器空间起始地址
        reg_ptr_ = mmap(NULL, REG_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_LOCKED, fd, 0);
        if (reg_ptr_ == MAP_FAILED)
        {
            fprintf(stderr, "Failed to mmap: %s\n", strerror(errno));
            exit(1);
        }
        CHECK(cudaHostRegister(reg_ptr_, REG_SIZE, cudaHostRegisterIoMemory));

        // reset controller
        uint64_t h_reg_ptr = (uint64_t)reg_ptr_;
        *(uint32_t *)(h_reg_ptr + REG_CC) &= ~REG_CC_EN;
        // SSD 是否成功启动
        while (*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY)
            ;
        fprintf(stderr, "reset done\n");

        // set admin_qp queue attributes
        *(uint32_t *)(h_reg_ptr + REG_AQA) = ((ADMIN_QUEUE_DEPTH - 1) << 16) | (ADMIN_QUEUE_DEPTH - 1);

        posix_memalign(&h_admin_queue_, HOST_PGSZ, HOST_PGSZ * 2);
        memset(h_admin_queue_, 0, HOST_PGSZ * 2);
        // SSD访问CPU buffer
        nvm_ioctl_map req; // convert to physical address
        req.vaddr_start = (uint64_t)h_admin_queue_;
        req.n_pages = 2;
        // req.ioaddrs[x]内是这一段内存的物理地址;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * 2);
        int err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
        if (err)
        {
            fprintf(stderr, "Failed to map admin_qp queue: %s\n", strerror(errno));
            exit(1);
        }
        uint64_t asq = (uint64_t)h_admin_queue_;
        *(uint64_t *)(h_reg_ptr + REG_ASQ) = req.ioaddrs[0];
        uint64_t acq = (uint64_t)h_admin_queue_ + HOST_PGSZ;
        *(uint64_t *)(h_reg_ptr + REG_ACQ) = req.ioaddrs[1];
        // fprintf(stderr,"asq:%ld\n",asq);
        // fprintf(stderr,"Address at asq: %p\n", (volatile uint32_t *)asq);
        SSDQueuePair admin_qp((volatile uint32_t *)asq, (volatile uint32_t *)acq, BROADCAST_NSID, (uint32_t *)(h_reg_ptr + REG_SQTDBL), (uint32_t *)(h_reg_ptr + REG_CQHDBL), ADMIN_QUEUE_DEPTH);
        fprintf(stderr, "set admin_qp queue attributes done\n");

        // enable controller
        *(uint32_t *)(h_reg_ptr + REG_CC) |= REG_CC_EN;
        while (!(*(uint32_t volatile *)(h_reg_ptr + REG_CSTS) & REG_CSTS_RDY))
            ;
        fprintf(stderr, "enable controller done\n");

        // set number of I/O queues, NVMe 中的命令长度均为 16 DW（64 Bytes）
        uint32_t cid;
        admin_qp.submit(cid, OPCODE_SET_FEATURES, 0x0, 0x0, FID_NUM_QUEUES,
                        ((num_queues_per_ssd_ - 1) << 16) | (num_queues_per_ssd_ - 1));
        uint32_t status;
        admin_qp.poll(status, cid);
        if (status != 0)
        {
            fprintf(stderr, "set number of queues failed with status 0x%x\n", status);
            exit(1);
        }
        fprintf(stderr, "set number of queues done!\n");

        for (int gpu_id = 0; gpu_id < nums_gpus; ++gpu_id)
        {
            cudaSetDevice(gpu_id);
            // create I/O queues
            int sq_size = QUEUE_DEPTH * SQ_ITEM_SIZE;
            assert(sq_size % HOST_PGSZ == 0);
            // 经过这个操作后，d_io_queue_内存的内存地址变了，指针自身的地址没变。
            uint64_t *phys = cudaMallocAlignedMapped(d_io_queue_[gpu_id], sq_size * 2 * num_queues_per_ssd_per_gpu, fd); // 2 stands for SQ and CQ
            // printf("phys: %lx\n", (uint64_t)phys);
            // printf("&d_io_queue_:%lx\n",&d_io_queue_);
            CHECK(cudaMemset(d_io_queue_[gpu_id], 0, sq_size * 2 * num_queues_per_ssd_per_gpu));
            for (int i = 0; i < num_queues_per_ssd_per_gpu; i++)
            {
                uint64_t sq = (uint64_t)d_io_queue_[gpu_id] + sq_size * (2 * i);
                uint64_t cq = (uint64_t)d_io_queue_[gpu_id] + sq_size * (2 * i + 1);
                int qid = gpu_id * num_queues_per_ssd_per_gpu + i + 1;
                // printf("qid: %d\n", qid);
                int offset = sq_size * (2 * i + 1);
                uint64_t prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
                // 0x1指 物理地址连续
                admin_qp.submit(cid, OPCODE_CREATE_IO_CQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, 0x1);
                admin_qp.poll(status, cid);
                if (status != 0)
                {
                    fprintf(stderr, "create I/O CQ failed with status 0x%x\n", status);
                    exit(1);
                }
                offset = sq_size * (2 * i);
                prp1 = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
                admin_qp.submit(cid, OPCODE_CREATE_IO_SQ, prp1, 0x0, ((QUEUE_DEPTH - 1) << 16) | qid, (qid << 16) | 0x1);
                admin_qp.poll(status, cid);
                if (status != 0)
                {
                    fprintf(stderr, "create I/O SQ failed with status 0x%x\n", status);
                    exit(1);
                }
                uint32_t *cmd_id_to_req_id;
                CHECK(cudaMalloc(&cmd_id_to_req_id, QUEUE_DEPTH * sizeof(uint32_t)));
                uint32_t *cmd_id_to_sq_pos;
                CHECK(cudaMalloc(&cmd_id_to_sq_pos, QUEUE_DEPTH * sizeof(uint32_t)));
                bool *sq_entry_busy;
                CHECK(cudaMalloc(&sq_entry_busy, QUEUE_DEPTH));
                CHECK(cudaMemset(sq_entry_busy, 0, QUEUE_DEPTH));
                SSDQueuePair current_qp((volatile uint32_t *)sq, (volatile uint32_t *)cq, 0x1, (uint32_t *)(h_reg_ptr + REG_SQTDBL + DBL_STRIDE * qid), (uint32_t *)(h_reg_ptr + REG_CQHDBL + DBL_STRIDE * qid), QUEUE_DEPTH, cmd_id_to_req_id, cmd_id_to_sq_pos, sq_entry_busy);
                CHECK(cudaMemcpy(d_ssdqp_[gpu_id] + ssd_id * num_queues_per_ssd_per_gpu + i, &current_qp, sizeof(SSDQueuePair), cudaMemcpyHostToDevice));
            }
            // free(phys);
            fprintf(stderr, "create GPU%d I/O queues done!\n", gpu_id);

            // alloc IO buffer
            phys = cudaMallocAlignedMapped(d_io_buf_[gpu_id], (size_t)QUEUE_IOBUF_SIZE * num_queues_per_ssd_per_gpu, fd);
            // printf("phys: %lx\n\n", (uint64_t)phys);

            // d_prp1存的是ssd个(指向开辟的一堆phys address的指针)的起始地址
            CHECK(cudaMemcpy(d_prp1_[gpu_id] + ssd_id, phys, sizeof(uint64_t), cudaMemcpyHostToDevice));

            // cudaMemcpyHostToDevice 而不是 cudaMemcpyDeviceToDevice?
            CHECK(cudaMemcpy(d_IO_buf_base_[gpu_id] + ssd_id, &d_io_buf_[gpu_id], sizeof(uint64_t), cudaMemcpyHostToDevice));

            d_IO_buf_base_map[gpu_id] = d_IO_buf_base_[gpu_id];

            // fprintf(stderr, "d_io_buf_: %ld\n",d_io_buf_);
            // fprintf(stderr, "phys: %ld\n",phys);
            h_IO_buf_base_[gpu_id][ssd_id] = (uint64_t *)d_io_buf_[gpu_id];
            fprintf(stderr, "create GPU%d I/O buffers done!\n", gpu_id);
            // for (int i = 0; i < QUEUE_IOBUF_SIZE * num_queues_per_ssd_ / DEVICE_PGSZ; i++)
            //     printf("%lx\n", phys[i]);

            // build PRP list
            assert(PRP_SIZE <= HOST_PGSZ);
            int prp_size_per_ssd_per_gpu = PRP_SIZE * QUEUE_DEPTH * num_queues_per_ssd_per_gpu / HOST_PGSZ * HOST_PGSZ + HOST_PGSZ; // 1 table per ssd in host memory
            // printf("prp_size_per_ssd: %d\n", prp_size_per_ssd);
            void *tmp;
            posix_memalign(&tmp, HOST_PGSZ, prp_size_per_ssd_per_gpu);
            memset(tmp, 0, prp_size_per_ssd_per_gpu);
            uint64_t *prp = (uint64_t *)tmp;
            for (int i = 0; i < QUEUE_DEPTH * num_queues_per_ssd_per_gpu; i++)
                for (int j = 1; j < NUM_PRP_ENTRIES; j++)
                {
                    int prp_idx = i * NUM_PRP_ENTRIES + j;
                    int offset = i * MAX_IO_SIZE + j * HOST_PGSZ;
                    prp[prp_idx - 1] = phys[offset / DEVICE_PGSZ] + offset % DEVICE_PGSZ;
                }
            // Fill in each PRP table
            // free(phys);
            // for (int i = 0; i < QUEUE_DEPTH * num_queues_per_ssd_ * NUM_PRP_ENTRIES; i++)
            //     printf("%lx\n", prp[i]);
            req.vaddr_start = (uint64_t)prp;
            req.n_pages = prp_size_per_ssd_per_gpu / HOST_PGSZ;
            req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * req.n_pages);
            // req.ioaddrs[0] is a physical pointer to PRP table
            err = ioctl(fd, NVM_MAP_HOST_MEMORY, &req);
            if (err)
            {
                fprintf(stderr, "Failed to map: %s\n", strerror(errno));
                exit(1);
            }
            // for (int i = 0; i < req.n_pages; i++)
            //     printf("%lx ", req.ioaddrs[i]);
            // printf("\n");
            CHECK(cudaMemcpy(d_prp2_[gpu_id] + ssd_id * req.n_pages, req.ioaddrs, req.n_pages * sizeof(uint64_t), cudaMemcpyHostToDevice));
            // d_prp2_ is an array of physical pointer to PRP table
        }
        fprintf(stderr, "SSD %d Init done!\n", ssd_id);
    }

    uint64_t *cudaMallocAlignedMapped(void *&vaddr, size_t size, int fd)
    {
        size = size / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ;
        uint64_t *ptr;
        CHECK(cudaMalloc(&ptr, size + DEVICE_PGSZ));
        vaddr = (void *)((uint64_t)ptr / DEVICE_PGSZ * DEVICE_PGSZ + DEVICE_PGSZ);
        int flag = 0;
        // printf("ptr:%lx\n",ptr);
        // printf("vaddr: %lx\n", (uint64_t)vaddr);
        if ((uint64_t)vaddr != (uint64_t)ptr)
        {
            flag = 1;
        }
        nvm_ioctl_map req;
        req.ioaddrs = (uint64_t *)malloc(sizeof(uint64_t) * (size / DEVICE_PGSZ + flag));
        req.n_pages = size / DEVICE_PGSZ + flag;
        req.vaddr_start = (uint64_t)ptr;
        int err = ioctl(fd, NVM_MAP_DEVICE_MEMORY, &req);
        if (err)
        {
            printf("Failed to map: %s\n", strerror(errno));
            return nullptr;
        }
        // printf("(phys add)req.ioaddrs[1]: %lx\n", req.ioaddrs[1]);
        // printf("(virtual add)req.ioaddrs+flag: %lx\n", (uint64_t)(req.ioaddrs + 1));
        return req.ioaddrs + flag;
    }
};