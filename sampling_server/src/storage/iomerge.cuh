#include <iostream>
#include <climits>
// #include <iomanip>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <algorithm>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <unistd.h>
#include <iostack.cuh>
#include <cub/cub.cuh>
// #include

#define my_ULLONG_MAX (~0ULL)
// #define MAX_ITEMS 16//when setting to 32 will cause compilation error
using namespace std;

typedef uint64_t app_addr_t;


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


struct cache_id{
    uint64_t buffer_id;
    uint64_t ssd_id;
    __forceinline__
    __host__ __device__
    cache_id(){
        buffer_id=my_ULLONG_MAX;
        ssd_id=my_ULLONG_MAX;
    }
    __forceinline__
    __host__ __device__
    cache_id(uint64_t ssd,uint64_t buffer):ssd_id(ssd),buffer_id(buffer){}
    
    __host__ __device__
    bool operator < (const cache_id& lhs)const{
        return this->ssd_id<lhs.ssd_id;
    }
};

struct Compare{
    __device__ 
    bool operator()(const IOReq& a, const IOReq& b) const
        {
            return a.start_lb < b.start_lb;
        }

    __device__ 
    bool operator()(const cache_id& a,const cache_id& b) const{
        return a.ssd_id < b.ssd_id;
    }
};

__global__
void dequeue_kernel(char* sample_bin_ids,int64_t* sample_orders, int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd);
                        
// __host__ __device__
// bool cmp(IOReq a,IOReq b){
//     return a.ssd_addr<b.ssd_addr;
// }

struct IOMerge{
    int num_gpus;
    void *ssd_start_addr;
    uint64_t grid_size{80}; //cuda param
    uint64_t block_size{1024};
    uint64_t merge_lenth_{8};
    uint64_t ssd_block_size{ITEM_SIZE}; //block size
    uint64_t init_buffer_cnt_;
    
    
    cache_id** d_ssd_plus_buffer_; 
    size_t temp_storage_bytes_{102400000};  //if the input size is too big, should set it when constructing
    void **d_temp_storage_; 
    uint64_t** d_split_flag_; 
    IOReq **d_ret_; 
    uint64_t **d_ret_ssd_; 
    int32_t** cache_miss_; 
    int32_t** p_miss_cnt_; 

    // __forceinline__
    __host__ __device__
    IOMerge(uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t ssd_block_size,uint64_t init_buffer_cnt,size_t temp_storage_bytes, int num_gpus):
            grid_size(grid_size),block_size(block_size),merge_lenth_(merge_lenth),
            ssd_block_size(ssd_block_size),init_buffer_cnt_(init_buffer_cnt),temp_storage_bytes_(temp_storage_bytes),num_gpus(num_gpus){

            d_ssd_plus_buffer_ = (cache_id **)malloc(sizeof(cache_id *) * num_gpus);
            d_temp_storage_ = (void **)malloc(sizeof(void *) * num_gpus);
            d_split_flag_ = (uint64_t **)malloc(sizeof(uint64_t *) * num_gpus);
            d_ret_ = (IOReq **)malloc(sizeof(IOReq *) * num_gpus);
            d_ret_ssd_ = (uint64_t **)malloc(sizeof(uint64_t *) * num_gpus);
            cache_miss_ = (int32_t **)malloc(sizeof(int32_t *) * num_gpus);
            p_miss_cnt_ = (int32_t **)malloc(sizeof(int32_t *) * num_gpus);

            for(int gpu_id = 0; gpu_id < num_gpus; gpu_id++){
                cudaSetDevice(gpu_id);
                CHECK(cudaMalloc((void **)&p_miss_cnt_[gpu_id],sizeof(uint64_t)));

                CHECK(cudaMalloc((void **)&d_ssd_plus_buffer_[gpu_id],sizeof(cache_id)*init_buffer_cnt_));
                CHECK(cudaMalloc((void **)&d_temp_storage_[gpu_id],temp_storage_bytes_));
                CHECK(cudaMalloc((void **)&d_split_flag_[gpu_id],sizeof(uint64_t)*init_buffer_cnt_));
                CHECK(cudaMalloc((void **)&d_ret_[gpu_id],sizeof(IOReq)*init_buffer_cnt_));
                CHECK(cudaMalloc((void **)&d_ret_ssd_[gpu_id],sizeof(uint64_t)*init_buffer_cnt_)); 
                CHECK(cudaMalloc((void **)&cache_miss_[gpu_id],sizeof(int32_t)*init_buffer_cnt_)); 
            }            
            // printf("IOMerge init finished!\n");
            // cudaMalloc((void **)&input_ids,sizeof(uint64_t)*init_buffer_cnt_);
            // cudaMalloc((void **)&cache_index,sizeof(int32_t)*init_buffer_cnt_);               
    };
    
    // __forceinline__
    __host__ __device__
    IOMerge(uint64_t grid_size,uint64_t block_size,uint64_t merge_lenth,uint64_t ssd_block_size,uint64_t init_buffer_cnt):
            grid_size(grid_size),block_size(block_size),merge_lenth_(merge_lenth),
            ssd_block_size(ssd_block_size),init_buffer_cnt_(init_buffer_cnt){
            cudaMalloc((void **)&p_miss_cnt_,sizeof(uint64_t));

            cudaMalloc((void **)&d_ssd_plus_buffer_,sizeof(cache_id)*init_buffer_cnt_);
            cudaMalloc((void **)&d_temp_storage_,temp_storage_bytes_);
            cudaMalloc((void **)&d_split_flag_,sizeof(uint64_t)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_,sizeof(IOReq)*init_buffer_cnt_);
            cudaMalloc((void **)&d_ret_ssd_,sizeof(uint64_t)*init_buffer_cnt_); 
            cudaMalloc((void **)&cache_miss_,sizeof(int32_t)*init_buffer_cnt_); 

            // cudaMalloc((void **)&input_ids,sizeof(uint64_t)*init_buffer_cnt_);
            // cudaMalloc((void **)&cache_index,sizeof(int32_t)*init_buffer_cnt_);               
            };

    __forceinline__
    __host__ __device__
    ~IOMerge(){
        //maybe free the cuda memory
    }

    __host__
    IOReq* dequeue(char* sample_bin_ids, int64_t* sample_orders, int32_t* node_counter, int32_t op_id, int32_t* cache_index, int32_t* input_ids, int32_t* p_miss_cnt, float* dst_float_buffer, int32_t float_feature_len, int num_ssd, int dev_id, cudaStream_t stream){
        cudaSetDevice(dev_id);
        dequeue_kernel<<<grid_size, block_size, 0, stream>>>(sample_bin_ids, sample_orders, node_counter, op_id, d_ret_[dev_id], p_miss_cnt_[dev_id], input_ids, cache_index, float_feature_len, dst_float_buffer, num_ssd);
        cudaDeviceSynchronize();
        cudaMemcpyAsync(p_miss_cnt, p_miss_cnt_[dev_id], sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(p_miss_cnt_[dev_id], 0, sizeof(int32_t), stream);
        
        return d_ret_[dev_id];        
    }

};

__global__
void dequeue_kernel(char* sample_bin_ids, int64_t* sample_orders,int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd){
                    
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;  
    int32_t  node_off = node_counter[(op_id % INTRABATCH_CON) * 2];
    int32_t  batch_size = node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    int feature_block_size = float_feature_len * sizeof(float) / ITEM_SIZE;            
    for(;thread_id<batch_size;thread_id+=blockDim.x*gridDim.x){
        int32_t input_id = input_ids[thread_id];
        int32_t cache_idx = cache_index[thread_id];
        if(cache_index < 0 || sample_bin_ids[input_id] >= 2){
            uint64_t offset = atomicAdd(p_miss_cnt, 1);
            // printf("offset:%d\n", offset);
            // d_ret[offset].start_lb = uint64_t(input_id % num_ssd) * NUM_LBS_PER_SSD + uint64_t(input_id / num_ssd) * feature_block_size;//raid0
            d_ret[offset].start_lb = (sample_bin_ids[input_id]-2) * NUM_LBS_PER_SSD + sample_orders[input_id] * feature_block_size;// + 1000000000;//raid0
            d_ret[offset].num_items = feature_block_size;
            for(int j = 0; j < feature_block_size; j++){
                d_ret[offset].dest_addr[j] = (app_addr_t)(dst_float_buffer + (int64_t(node_off) * float_feature_len) + (1ll * thread_id * float_feature_len + j * ITEM_SIZE) / sizeof(float));  
            }
        }
    }
}

