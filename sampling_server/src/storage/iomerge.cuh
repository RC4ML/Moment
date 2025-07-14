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
void print_cacheid(cache_id *D_array,uint64_t len){
    for(uint64_t i=0;i<len;i++){
        printf("D_array[%lu]=%lu\n",i,D_array[i].ssd_id);
    }
}

__global__
void print_uint64(uint64_t *D_array,uint64_t len){
    for(uint64_t i=0;i<len;i++){
        printf("D_array[%lu]=%lu\n",i,D_array[i]);
    }
}

__global__
void print_Dret(IOReq* d_ret,uint64_t len,void *ssd_start_addr,void *dst_float_buffer,uint64_t ssd_block_size){
    printf("print_Dret\n");
    for(uint64_t i=0;i<len;i++){
        printf("cache:%lu\n",((uint64_t)d_ret[i].start_lb-(uint64_t)ssd_start_addr)/ssd_block_size);
        printf("gpu0:%lu\n",((uint64_t)d_ret[i].dest_addr[0]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu1:%lu\n",((uint64_t)d_ret[i].dest_addr[1]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu2:%lu\n",((uint64_t)d_ret[i].dest_addr[2]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu3:%lu\n",((uint64_t)d_ret[i].dest_addr[3]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu4:%lu\n",((uint64_t)d_ret[i].dest_addr[4]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu5:%lu\n",((uint64_t)d_ret[i].dest_addr[5]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu6:%lu\n",((uint64_t)d_ret[i].dest_addr[6]-(uint64_t)dst_float_buffer)/ssd_block_size);
        printf("gpu7:%lu\n\n",((uint64_t)d_ret[i].dest_addr[7]-(uint64_t)dst_float_buffer)/ssd_block_size);
    }
}

__global__
void print_Dret_ssd(uint64_t* d_ret_ssd,uint64_t len,void *ssd_start_addr,uint64_t ssd_block_size){
    printf("print_Dret_ssd\n");
    for(uint64_t i=0;i<len;i++){
        // if(i==0){
        //     printf("the first origin ssd:%lu\n",(uint64_t)d_ret_ssd[i]);
        // }
        printf("ssd:%lu\n",((uint64_t)d_ret_ssd[i]-(uint64_t)ssd_start_addr)/ssd_block_size);
    }
}
__global__
void get_miss(cache_id *d_ssd_plus_buffer_,int32_t *cache_index,int32_t *input_ids,uint64_t input_num);

__global__
void merge_kernel(cache_id *d_ssd_plus_buffer_,IOReq *d_ret,uint64_t *d_ret_ssd,uint64_t* d_split_flag_,uint64_t lenth,
                    void *dst_float_buffer,uint64_t merge_lenth,uint64_t ssd_block_size);

__global__
void count_miss(uint64_t input_num, int32_t* cache_miss, int32_t* cache_index);

__global__
void no_merge_kernel(IOReq *d_ret, uint64_t* d_ret_ssd, uint64_t input_num, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                        float *dst_float_buffer, int32_t num_ssd);

__global__
void dequeue_kernel(char* sample_bin_ids,int64_t* sample_orders, int32_t* node_counter, int op_id, IOReq *d_ret, int32_t* p_miss_cnt, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd);
                        
__global__
void split_flag(uint64_t *d_split_flag_,cache_id *d_ssd_plus_buffer_,uint64_t lenth,uint64_t merge_lenth);   

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

    // __host__
    // IOReq* naive_merge(int32_t* cache_index, int32_t* input_ids, int32_t& input_num, float* dst_float_buffer, cudaStream_t stream){

    //     uint64_t miss_cnt;
    //     uint64_t *p_miss_cnt_;
    //     cub::DeviceReduce::Sum(NULL,temp_storage_bytes_,cache_index,p_miss_cnt_,input_num,stream);
    //     cub::DeviceReduce::Sum(d_temp_storage_,temp_storage_bytes_,cache_index,p_miss_cnt_,input_num,stream);
    //     cudaMemcpy(&miss_cnt,p_miss_cnt_,sizeof(uint64_t),cudaMemcpyDeviceToHost);
        
    //     get_miss<<<grid_size,block_size,0,stream>>>(d_ssd_plus_buffer_,cache_index,input_ids,input_num);
        
    //     cub::DeviceRadixSort::SortPairs(NULL,temp_storage_bytes_,input_ids,input_ids,
    //                                     d_ssd_plus_buffer_,d_ssd_plus_buffer_,miss_cnt,
    //                                     0,sizeof(uint64_t)*8,stream);

    //     cub::DeviceRadixSort::SortPairs(d_temp_storage_,temp_storage_bytes_,input_ids,input_ids,
    //                                     d_ssd_plus_buffer_,d_ssd_plus_buffer_,miss_cnt,
    //                                     0,sizeof(uint64_t)*8,stream);

    //     split_flag<<<grid_size,block_size,0,stream>>>(d_split_flag_,d_ssd_plus_buffer_,miss_cnt,merge_lenth_);

    //     merge_kernel<<<grid_size,block_size,0,stream>>>(d_ssd_plus_buffer_,d_ret_,d_ret_ssd_,d_split_flag_,
    //                                             miss_cnt,dst_float_buffer,merge_lenth_,ssd_block_size);
            
    //     cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
    //                                     d_ret_, d_ret_, miss_cnt,
    //                                     0,sizeof(uint64_t)*8,stream);
    //     // printf("temp_storage_bytes_:%d\n",temp_storage_bytes_);
    //     cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
    //                                     d_ret_, d_ret_, miss_cnt,
    //                                     0,sizeof(uint64_t)*8,stream);
        
    //     cout<<"miss_cnt:"<<miss_cnt<<endl;
    //     return d_ret_;
    // }

    // __host__
    // IOReq* no_merge(int32_t* cache_index, int32_t* input_ids, int32_t input_num, uint64_t* p_miss_cnt, float* dst_float_buffer, int32_t float_feature_len, int num_ssd, cudaStream_t stream){
        
    //     count_miss<<<grid_size, block_size, 0, stream>>>(input_num, cache_miss_, cache_index);
    //     cub::DeviceReduce::Sum(NULL,temp_storage_bytes_,cache_miss_,p_miss_cnt_,input_num,stream);
    //     cub::DeviceReduce::Sum(d_temp_storage_,temp_storage_bytes_,cache_miss_,p_miss_cnt_,input_num,stream);
    //     cudaMemcpyAsync(p_miss_cnt,p_miss_cnt_,sizeof(uint64_t),cudaMemcpyDeviceToDevice, stream);
        
    //     no_merge_kernel<<<grid_size, block_size, 0, stream>>>(d_ret_, d_ret_ssd_, input_num, input_ids, cache_index, float_feature_len, dst_float_buffer, num_ssd);
    //     cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
    //                 d_ret_, d_ret_, input_num,
    //                 0,sizeof(uint64_t)*8,stream);
    //     // printf("temp_storage_bytes_:%d\n",temp_storage_bytes_);
    //     cub::DeviceRadixSort::SortPairs(d_temp_storage_, temp_storage_bytes_, d_ret_ssd_, d_ret_ssd_, 
    //                 d_ret_, d_ret_, input_num,
    //                 0,sizeof(uint64_t)*8,stream);
    //     return d_ret_;
    // }

    __host__
    IOReq* dequeue(char* sample_bin_ids, int64_t* sample_orders, int32_t* node_counter, int32_t op_id, int32_t* cache_index, int32_t* input_ids, int32_t* p_miss_cnt, float* dst_float_buffer, int32_t float_feature_len, int num_ssd, int dev_id, cudaStream_t stream){
        // printf("dev_id:%d\n", dev_id);
        // if(dev_id == 1){
            cudaSetDevice(dev_id);
        // printf("dequeue_kenel start\n");
            dequeue_kernel<<<grid_size, block_size, 0, stream>>>(sample_bin_ids, sample_orders, node_counter, op_id, d_ret_[dev_id], p_miss_cnt_[dev_id], input_ids, cache_index, float_feature_len, dst_float_buffer, num_ssd);
            cudaDeviceSynchronize();
            cudaMemcpyAsync(p_miss_cnt, p_miss_cnt_[dev_id], sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemsetAsync(p_miss_cnt_[dev_id], 0, sizeof(int32_t), stream);

            // int32_t* h_miss_cnt = (int32_t*)malloc(sizeof(int32_t));
            // cudaMemcpy(h_miss_cnt, p_miss_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);

            // IOReq* h_ret = (IOReq*)malloc(sizeof(IOReq) * init_buffer_cnt_);
            // cudaMemcpyAsync(h_ret, d_ret_[dev_id], sizeof(IOReq) * init_buffer_cnt_, cudaMemcpyDeviceToHost, stream);
            // for(int i = 0; i < h_miss_cnt[0]; i++){
            //     if(h_ret[i].start_lb < 0){
            //         printf("Error: h_ret[%d].start_lb < 0 dev_id: %d\n", i, dev_id);
            //         exit(0);
            //     }
            //     if (h_ret[i].dest_addr[0] < 0)
            //     {
            //         printf("Error: h_ret[%d].dest_addr[0] < 0 dev_id:%d\n", i, dev_id);
            //         exit(0);
            //     }
            //     // printf("%ld %d %ld %d\n", h_ret[i].start_lb, h_ret[i].num_items, h_ret[i].dest_addr[0], dev_id);
            // }
            // int32_t* h_miss_cnt = (int32_t*)malloc(sizeof(int32_t));
            // cudaMemcpy(h_miss_cnt, p_miss_cnt_[dev_id], sizeof(int32_t), cudaMemcpyDeviceToHost);
            // std::cout<<"op "<<op_id<<" cache miss "<<h_miss_cnt[0]<<" "<<dev_id<<"\n";
            // cudaDeviceSynchronize();
            // cudaMemset(p_miss_cnt_[dev_id], 0, sizeof(int32_t));
            // cudaMemcpyAsync(p_miss_cnt, p_miss_cnt_[dev_id], sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            // cudaMemsetAsync(p_miss_cnt_[dev_id], 0, sizeof(int32_t), stream);
            // printf("dequeue_kenel finish\n");
            return d_ret_[dev_id];
        // }
        
    }

};

__global__
void get_miss(cache_id *d_ssd_plus_buffer_,int32_t *cache_index,int32_t *input_ids,uint64_t input_num){
    int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    for(int i=thread_id;i<input_num;i+=blockDim.x*gridDim.x){
        // printf("cache_index[%d]=%d input_ids[%d]=%lu\n",i,cache_index[i],i,input_ids[i]);
        cache_id tmp(my_ULLONG_MAX,my_ULLONG_MAX);
        if(cache_index[i]==1){
            tmp.ssd_id=input_ids[i];
            tmp.buffer_id=i;
            d_ssd_plus_buffer_[i]=tmp;
        }
        else{
            d_ssd_plus_buffer_[i]=tmp;
            input_ids[i]=my_ULLONG_MAX;
        }
    }
    return;
}

// __global__
// void merge_kernel(cache_id *d_ssd_plus_buffer_,IOReq *d_ret,uint64_t *d_ret_ssd,uint64_t* d_split_flag_,uint64_t lenth,
//                     void *dst_float_buffer,uint64_t merge_lenth,uint64_t ssd_block_size){
//     uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
//     for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
//         // printf("d_ssd_plus_buffer_[%lu].ssd_id=%lu\n",i,d_ssd_plus_buffer_[i].ssd_id);
//         if(d_split_flag_[i]==1){
//             // printf("d_ssd_plus_buffer_[%lu].ssd_id=%lu\n",i,d_ssd_plus_buffer_[i].ssd_id);
//             uint64_t base_id=d_ssd_plus_buffer_[i].ssd_id/merge_lenth*merge_lenth;
//             IOReq req(base_id,merge_lenth);
//             d_ret_ssd[i]=base_id;
//             uint64_t start=i+1;
//             req.dest_addr[d_ssd_plus_buffer_[i].ssd_id-base_id]=(uint64_t)dst_float_buffer+ssd_block_size*d_ssd_plus_buffer_[i].buffer_id;
//             while(d_ssd_plus_buffer_[start].ssd_id!=my_ULLONG_MAX&&d_split_flag_[start]==0){
//                 req.dest_addr[d_ssd_plus_buffer_[start].ssd_id-base_id]=(uint64_t)dst_float_buffer+ssd_block_size*d_ssd_plus_buffer_[start].buffer_id;
//                 start++;
//             }
//             d_ret[i]=req;
//         }else{
//             IOReq req(my_ULLONG_MAX,merge_lenth);
//             d_ret_ssd[i]=my_ULLONG_MAX;
//             d_ret[i]=req;
//         }
        
//     }
// }

__global__
void no_merge_kernel(IOReq *d_ret, uint64_t* d_ret_ssd, uint64_t input_num, int32_t* input_ids, int32_t* cache_index, int32_t float_feature_len,
                    float *dst_float_buffer, int32_t num_ssd){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;  
    int feature_block_size = float_feature_len * sizeof(float) / ITEM_SIZE;            
    for(;thread_id<input_num;thread_id+=blockDim.x*gridDim.x){
        int32_t input_id = input_ids[thread_id];
        int32_t cache_idx = cache_index[thread_id];
        if(cache_idx < 0){
            d_ret[thread_id].start_lb = (input_id % num_ssd) * NUM_LBS_PER_SSD + (input_id / num_ssd) * feature_block_size;//raid0
            d_ret[thread_id].num_items = feature_block_size;
            for(int j = 0; j < feature_block_size; j++){
                d_ret[thread_id].dest_addr[j] = (app_addr_t)(dst_float_buffer + (1ll * thread_id * float_feature_len + j * ITEM_SIZE) / sizeof(float));
            }
            d_ret_ssd[thread_id] = (input_id % num_ssd) * NUM_LBS_PER_SSD + (input_id / num_ssd) * feature_block_size;//raid0
        }else{
            d_ret[thread_id].start_lb = my_ULLONG_MAX;
            d_ret[thread_id].num_items = 0;
            d_ret[thread_id].dest_addr[0] = 0;
            d_ret_ssd[thread_id] = my_ULLONG_MAX;//raid0
        }

    }
}

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
            d_ret[offset].start_lb = (sample_bin_ids[input_id]-2) * NUM_LBS_PER_SSD + sample_orders[input_id] * feature_block_size + 1000000000;//raid0
            d_ret[offset].num_items = feature_block_size;
            for(int j = 0; j < feature_block_size; j++){
                // if(thread_id == 0 && j == 0)
                    // printf("offset:%d, node_off:%d, thread_id:%d, j:%d, float_feature_len:%d, input_id:%d, cache_idx:%d, d_ret[offset].start_lb:%lu\n", offset, node_off, thread_id, j, float_feature_len, input_id, cache_idx, d_ret[offset].start_lb);
                d_ret[offset].dest_addr[j] = (app_addr_t)(dst_float_buffer + (int64_t(node_off) * float_feature_len) + (1ll * thread_id * float_feature_len + j * ITEM_SIZE) / sizeof(float));  
                // printf("no problem\n");
            }
        }
    }
    // printf("dequeue_kernel finish\n");
}

// __global__
// void no_merge_kernel(IOReq *d_ret,int32_t *cache_index,int32_t *input_ids,uint64_t input_num,
//                     void *dst_float_buffer,uint64_t ssd_block_size){
//     uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;              
//     for(uint64_t i=thread_id;i<input_num;i+=blockDim.x*gridDim.x){
//         int32_t cache_idx = cache_index[i];
//             d_ret[i].start_lb=i%2;//input_ids[i];
//             d_ret[i].dest_addr[0]=(uint64_t)(dst_float_buffer+ssd_block_size*i/4);
//             d_ret[i].num_items=1;
//     }
// }

__global__
void count_miss(uint64_t input_num, int32_t* cache_miss, int32_t* cache_index){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;              
    for(;thread_id<input_num;thread_id+=blockDim.x*gridDim.x){
        if(cache_index[thread_id] < 0){
            cache_miss[thread_id] = 1; 
            // printf("Miss %d\n", thread_id);
        }else{
            cache_miss[thread_id] = 0;
        }
    }
}

__global__
void split_flag(uint64_t *d_split_flag_,cache_id *d_ssd_plus_buffer_,uint64_t lenth,uint64_t merge_lenth){
    uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
    if(thread_id==0){
        d_split_flag_[0]=1;
        thread_id+=blockDim.x*gridDim.x;
    }
    for(uint64_t i=thread_id;i<lenth;i+=blockDim.x*gridDim.x){
        if(d_ssd_plus_buffer_[i].ssd_id/merge_lenth==d_ssd_plus_buffer_[i-1].ssd_id/merge_lenth){
            d_split_flag_[i]=0;
        }
        else{
            d_split_flag_[i]=1;
        }
        // printf("D_ssds_plus_buffer[%lu].ssd_id=%lu  d_split_flag_[%lu]=%lu\n ",i,d_ssd_plus_buffer_[i].ssd_id,i,d_split_flag_[i]);
    }
}

