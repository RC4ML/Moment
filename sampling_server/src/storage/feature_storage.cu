#include "feature_storage.cuh"
#include "feature_storage_impl.cuh"
#include <iostream>

// __global__ void check_float(float* data, int32_t len, int op_id){
//     for(int i = 0; i < len; i++){
//         printf("%i %f %d\n", i, data[i], op_id);
//         // printf("base %d %p %d\n", i, data+i, op_id);
//     }
// }


#include <unordered_set>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>
#define TEST_SIZE 0x10000000
#define APP_BUF_SIZE 0x10000000
#define NUM_QUEUES_PER_SSD 128
#define NUM_SSDS 1

__device__ float **IO_buf_base;

__device__ uint64_t seed;
__global__ void gen_test_data(int ssd_id, int req_id)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        seed = seed * 0x5deece66d + 0xb;
        IO_buf_base[ssd_id][i] = req_id * MAX_IO_SIZE / 4 + i; 
    }
}

__global__ void check_test_data(float *app_buf, int idx)
{
    for (int i = 0; i < MAX_IO_SIZE / 4; i++)
    {
        if(i < 10){
            printf("%f\n", app_buf[i]);
        }
        // if (app_buf[i] != idx * MAX_IO_SIZE / 4 + i)
        // {
        //     printf("check failed at block %d, i = %d, read %lx, expected %x\n", idx, i, app_buf[i], idx * MAX_IO_SIZE / 4 + i);
        //     assert(0);
        // }
    }
}

__global__ void fill_app_buf(float *app_buf)
{
    for (int i = 0; i < TEST_SIZE / 4; i++)
        app_buf[i] = 0;
}

__global__ void check_float(int32_t* sampled_ids, float* data, int32_t len, int op_id){
    for(int i = 0; i < len; i++){
        // if((int)data[i] != (i / 1024)){
        //     printf("%i %f %d\n", i, data[i], i/1024);
        // }
        if(i % 128 == 0){
            if((((int)data[i]/ 1000) != ((sampled_ids[i/128]) / 1000)) && (((int)data[i]/ 1000) != ((sampled_ids[i/128]) / 1000 + 1))){
                printf("%i %f %d %d\n", i, data[i], (int)data[i]/ 1000, ((sampled_ids[i/128])));
            }
        }else{
            if(((int)data[i] / 100) != (i % 128 / 100)){
                printf("%i %f %d\n", i, data[i], i % 128);
            } 
        }

    }
}

// __global__
// void no_merge_kernel(IOReq *d_ret,uint64_t input_num, int32_t* input_ids,
//                     float *dst_float_buffer,uint64_t ssd_block_size, int32_t num_ssd){
//     uint64_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;              
//     for(;thread_id<input_num;thread_id+=blockDim.x*gridDim.x){
//         d_ret[thread_id].start_lb = (input_ids[thread_id] % num_ssd) * NUM_LBS_PER_SSD + (input_ids[thread_id] / num_ssd);//i;
//         d_ret[thread_id].num_items = 1;
//         d_ret[thread_id].dest_addr[0] = (app_addr_t)(dst_float_buffer + (1ll * thread_id * ITEM_SIZE) / 4);
//     }
// }


class CompleteFeatureStorage : public FeatureStorage{
public: 
    CompleteFeatureStorage(){
    }

    virtual ~CompleteFeatureStorage(){};

    void Build(BuildInfo* info) override {
        // printf("cudaDeviceid: %ld\n", info->cudaDeviceId);
        iostack_ = new IOStack(info->num_ssd, info->num_queues_per_ssd, info->partition_count);
        num_ssd_ = info->num_ssd;
        // std::cout<<"IOStack built\n";
        iomerge_ = new IOMerge(32, 1024, 8, 512, 4000000, 1000000000, info->partition_count);
        // std::cout<<"IOMerge built\n";
 
        int32_t partition_count = info->partition_count;
        total_num_nodes_ = info->total_num_nodes;
        float_feature_len_ = info->float_feature_len;
        float* host_float_feature = info->host_float_feature;
        sample_bin_ids.resize(partition_count);
        sample_orders.resize(partition_count);
        // sample_bin_ids = info->sample_bin_ids;
        for(int i=0;i<partition_count;i++){
            cudaSetDevice(i);
            cudaMalloc(&sample_bin_ids[i], sizeof(char) * int64_t(total_num_nodes_));
            cudaMemcpy(sample_bin_ids[i], &info->sample_bin_ids[0], sizeof(char) * int64_t(total_num_nodes_), cudaMemcpyHostToDevice);
            cudaMalloc(&sample_orders[i], sizeof(int64_t) * int64_t(total_num_nodes_));
            cudaMemcpy(sample_orders[i], &info->sample_orders[0], sizeof(int64_t) * int64_t(total_num_nodes_), cudaMemcpyHostToDevice);
        }
        
        // sample_orders = info->sample_orders;
        // cudaMalloc(&sample_orders, sizeof(int64_t) * total_num_nodes_);


        // for(int i = 0; i < total_num_nodes_; i++){
        //     iostack_->write_data(0, i, 1);
        // }
        
        // if(float_feature_len_ > 0){
        //     cudaHostGetDevicePointer(&float_feature_, host_float_feature, 0);
        // }
        // cudaCheckError();

        // Test t;
        // t.build();
        // CHECK(cudaMalloc(&app_buf_, APP_BUF_SIZE));
        // CHECK(cudaHostAlloc(&h_reqs_, sizeof(IOReq) * 65536 * 8, cudaHostAllocMapped));
        // CHECK(cudaMalloc(&d_reqs_, sizeof(IOReq) * 65536 * 8));

        // cudaSetDevice(0);
        // cudaMalloc(&d_num_req_, sizeof(int32_t));
        // cudaMemset(d_num_req_, 0, sizeof(int32_t));

        d_num_req_ = (int32_t**)malloc(sizeof(int32_t*) * partition_count);
        for(int i = 0;i < partition_count;i++){
            cudaSetDevice(i);
            cudaMalloc(&d_num_req_[i], sizeof(int32_t));
            cudaMemset(d_num_req_[i], 0, sizeof(int32_t));
        }

        training_set_num_.resize(partition_count);
        training_set_ids_.resize(partition_count);
        training_labels_.resize(partition_count);

        validation_set_num_.resize(partition_count);
        validation_set_ids_.resize(partition_count);
        validation_labels_.resize(partition_count);

        testing_set_num_.resize(partition_count);
        testing_set_ids_.resize(partition_count);
        testing_labels_.resize(partition_count);

        partition_count_ = partition_count;

        for(int32_t i = 0; i < info->shard_to_partition.size(); i++){
            int32_t part_id = info->shard_to_partition[i];
            int32_t device_id = info->shard_to_device[i];
            /*part id = 0, 1, 2...*/

            training_set_num_[part_id] = info->training_set_num[part_id];
            // std::cout<<"Training set count "<<training_set_num_[part_id]<<" "<<info->training_set_num[part_id]<<"\n";

            validation_set_num_[part_id] = info->validation_set_num[part_id];
            testing_set_num_[part_id] = info->testing_set_num[part_id];

            cudaSetDevice(device_id);
            cudaCheckError();

            // std::cout<<"Training set on device "<<part_id<<" "<<training_set_num_[part_id]<<"\n";
            // std::cout<<"Testing set on device "<<part_id<<" "<<testing_set_num_[part_id]<<"\n";

            int32_t* train_ids;
            cudaMalloc(&train_ids, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_ids, info->training_set_ids[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_set_ids_[part_id] = train_ids;
            cudaCheckError();

            int32_t* valid_ids;
            cudaMalloc(&valid_ids, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_ids, info->validation_set_ids[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_set_ids_[part_id] = valid_ids;
            cudaCheckError();

            int32_t* test_ids;
            cudaMalloc(&test_ids, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_ids, info->testing_set_ids[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_set_ids_[part_id] = test_ids;
            cudaCheckError();

            int32_t* train_labels;
            cudaMalloc(&train_labels, training_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(train_labels, info->training_labels[part_id].data(), training_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            training_labels_[part_id] = train_labels;
            cudaCheckError();

            int32_t* valid_labels;
            cudaMalloc(&valid_labels, validation_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(valid_labels, info->validation_labels[part_id].data(), validation_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            validation_labels_[part_id] = valid_labels;
            cudaCheckError();

            int32_t* test_labels;
            cudaMalloc(&test_labels, testing_set_num_[part_id] * sizeof(int32_t));
            cudaMemcpy(test_labels, info->testing_labels[part_id].data(), testing_set_num_[part_id] * sizeof(int32_t), cudaMemcpyHostToDevice);
            testing_labels_[part_id] = test_labels;
            cudaCheckError();

        }

        cudaMalloc(&d_req_count_, sizeof(unsigned long long));
        cudaMemset(d_req_count_, 0, sizeof(unsigned long long));
        cudaCheckError();

    };

    void Finalize() override {
        // cudaFreeHost(float_feature_);
        for(int32_t i = 0; i < partition_count_; i++){
            cudaSetDevice(i);
            cudaFree(training_set_ids_[i]);
            cudaFree(validation_set_ids_[i]);
            cudaFree(testing_set_ids_[i]);
            cudaFree(training_labels_[i]);
            cudaFree(validation_labels_[i]);
            cudaFree(testing_labels_[i]);
        }
    }

    int32_t* GetTrainingSetIds(int32_t part_id) const override {
        return training_set_ids_[part_id];
    }
    int32_t* GetValidationSetIds(int32_t part_id) const override {
        return validation_set_ids_[part_id];
    }
    int32_t* GetTestingSetIds(int32_t part_id) const override {
        return testing_set_ids_[part_id];
    }

	int32_t* GetTrainingLabels(int32_t part_id) const override {
        return training_labels_[part_id];
    };
    int32_t* GetValidationLabels(int32_t part_id) const override {
        return validation_labels_[part_id];
    }
    int32_t* GetTestingLabels(int32_t part_id) const override {
        return testing_labels_[part_id];
    }

    int32_t TrainingSetSize(int32_t part_id) const override {
        return training_set_num_[part_id];
    }
    int32_t ValidationSetSize(int32_t part_id) const override {
        return validation_set_num_[part_id];
    }
    int32_t TestingSetSize(int32_t part_id) const override {
        return testing_set_num_[part_id];
    }

    int32_t TotalNodeNum() const override {
        return total_num_nodes_;
    }

    float* GetAllFloatFeature() const override {
        return float_feature_;
    }
    int32_t GetFloatFeatureLen() const override {
        return float_feature_len_;
    }

    void Print(BuildInfo* info) override {
    }

    void IOSubmit(int32_t* sampled_ids, int32_t* cache_index,
                  int32_t* node_counter, float* dst_float_buffer,
                  int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl) override {
		
        // int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
		// cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
		// cudaCheckError();
        // int32_t node_off = 0;
        // int32_t batch_size = 0;
                    
        // node_off   = h_node_counter[(op_id % INTRABATCH_CON) * 2];
        // batch_size = (h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1]);
        // printf("start merge\n");
        IOReq* reqs = iomerge_->dequeue(sample_bin_ids[dev_id], sample_orders[dev_id], node_counter, op_id, cache_index, sampled_ids, d_num_req_[dev_id], dst_float_buffer, float_feature_len_, num_ssd_, dev_id, strm_hdl);
        
        cudaCheckError();
        // printf("d_num_req_:%d\n", d_num_req_[0]);
        // printf("start submit\n");
        iostack_->submit_io_req(reqs, d_num_req_[dev_id], dev_id, strm_hdl);
        cudaCheckError();
    }

private:
    std::vector<int> training_set_num_;
    std::vector<int> validation_set_num_;
    std::vector<int> testing_set_num_;

    std::vector<int32_t*> training_set_ids_;
    std::vector<int32_t*> validation_set_ids_;
    std::vector<int32_t*> testing_set_ids_;

    std::vector<int32_t*> training_labels_;
    std::vector<int32_t*> validation_labels_;
    std::vector<int32_t*> testing_labels_;

    int32_t partition_count_;
    int64_t total_num_nodes_;
    float* float_feature_;
    int32_t float_feature_len_;

    unsigned long long* d_req_count_;

    int32_t num_ssd_;

    IOStack* iostack_;//single GPU multi-SSD
    IOMerge* iomerge_;
    int32_t** d_num_req_;
    // IOReq* h_reqs_;
    // IOReq* d_reqs_;
    // float *app_buf_;
    std::vector<char*> sample_bin_ids;
    std::vector<int64_t*> sample_orders;
    friend FeatureStorage* NewCompleteFeatureStorage();
};

extern "C" 
FeatureStorage* NewCompleteFeatureStorage(){
    CompleteFeatureStorage* ret = new CompleteFeatureStorage();
    return ret;
}
