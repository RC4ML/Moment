#include "cache.cuh"
#include "cache_impl.cuh"
#include <cstdio>      // 用于 fopen / fprintf / fclose
#include <cstdlib>     // 用于 malloc / free
#include <cstdint>     // 用于 int64_t
#include <cstring>     // 可选：strerror
#include <iostream>

bool dump_access_times_to_txt(const char* file_path,
                              const unsigned long long* data,
                              std::size_t num_nodes) {
    FILE* fp = std::fopen(file_path, "w");
    if (!fp) {
        std::cerr << "Failed to open file " << file_path
                  << " : " << std::strerror(errno) << '\n';
        return false;
    }

    for (std::size_t i = 0; i < num_nodes; ++i) {
        // 每行写一个十进制整数，末尾加换行
        if (std::fprintf(fp, "%llu\n", data[i]) < 0) {
            std::cerr << "Write error at index " << i << '\n';
            std::fclose(fp);
            return false;
        }
    }

    std::fclose(fp);
    return true;
}

bool destroy_file(const char* file_path) {
    if (std::remove(file_path) == 0) {          // 删除成功返回 0
        return true;
    } else {
        std::cerr << "Failed to remove " << file_path
                  << " : " << std::strerror(errno) << '\n';
        return false;
    }
}

class PreSCCacheController : public CacheController
{
public:
    PreSCCacheController(int32_t train_step, int32_t device_count)
    {
        train_step_ = train_step;
        device_count_ = device_count;
    }

    virtual ~PreSCCacheController() {}

    void Initialize(
        int32_t dev_id,
        int32_t total_num_nodes) override
    {
        device_idx_ = dev_id;
        total_num_nodes_ = total_num_nodes;
        cudaSetDevice(dev_id);
        cudaHostAlloc(&node_access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)), cudaHostAllocMapped);
        cudaMemset(node_access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaCheckError();
        // cudaMalloc(&edge_access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        // cudaMemset(edge_access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        // cudaCheckError();

        iter_ = 0;
        max_ids_ = 0;
        cudaMalloc(&d_global_count_, 4);
        h_global_count_ = (int32_t *)malloc(4);
        find_iter_ = 0;
        h_cache_hit_ = 0;
    }

    void Finalize() override
    {
        // pos_map_->clear();
    }

    void CacheProfiling(
        int32_t *sampled_ids,
        int32_t *agg_src_id,
        int32_t *agg_dst_id,
        int32_t *agg_src_off,
        int32_t *agg_dst_off,
        int32_t *node_counter,
        int32_t *edge_counter,
        bool is_presc,
        void *stream) override
    {
        dim3 block_num(32, 1);
        dim3 thread_num(1024, 1);

        if (is_presc)
        {
            int32_t *h_node_counter = (int32_t *)malloc(16 * sizeof(int32_t));
            cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
            HotnessMeasure<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(sampled_ids, node_counter, node_access_time_);

            if (h_node_counter[INTRABATCH_CON * 2 + 1] > max_ids_)
            {
                max_ids_ = h_node_counter[INTRABATCH_CON * 2 + 1];
            }
            if (iter_ == (train_step_ - 1))
            {
                iter_ = 0;
            }
            free(h_node_counter);
        }
        iter_++;
    }

    /*num candidates = sampled num*/
    void InitializeMap(int node_capacity, int edge_capacity) override
    {
        cudaSetDevice(device_idx_);
        node_capacity_ = node_capacity;
        edge_capacity_ = edge_capacity;

        auto invalid_key = CACHEMISS_FLAG;
        auto invalid_value = CACHEMISS_FLAG;

        node_map_ = new bght::bcht<int32_t, int32_t>(int64_t(node_capacity_ * device_count_) * 2, invalid_key, invalid_value);
        cudaCheckError();

        edge_index_map_ = new bght::bcht<int32_t, char>(int64_t(edge_capacity_ * device_count_) * 2, invalid_key, invalid_value);
        cudaCheckError();

        edge_offset_map_ = new bght::bcht<int32_t, int32_t>(int64_t(edge_capacity_ * device_count_) * 2, invalid_key, invalid_value);
        cudaCheckError();
    }

    void Insert(int32_t *QT, int32_t *QF, int32_t cache_expand, int32_t Kg) override
    {
        cudaSetDevice(device_idx_);
        cudaCheckError();

        cudaMalloc(&pair_, int64_t(int64_t(node_capacity_ * cache_expand) * sizeof(pair_type)));
        cudaCheckError();
        dim3 block_num(80, 1);
        dim3 thread_num(1024, 1);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        InitPair<<<block_num, thread_num>>>(pair_, QF, node_capacity_, cache_expand, Kg);
        cudaCheckError();
        node_map_->insert(pair_, (pair_ + node_capacity_ * cache_expand), stream);
        cudaCheckError();
        // if(success){
        //     std::cout<<"Feature Cache Successfully Initialized\n";
        // }
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaFree(pair_);
        cudaCheckError();
        // cudaFree(cache_ids_);
        // cudaCheckError();
        // cudaFree(cache_offset_);
        // cudaCheckError();

        index_pair_type *index_pair;
        offset_pair_type *offset_pair;
        cudaMalloc(&index_pair, int64_t(int64_t(edge_capacity_ * cache_expand) * sizeof(index_pair_type)));
        cudaCheckError();
        cudaMalloc(&offset_pair, int64_t(int64_t(edge_capacity_ * cache_expand) * sizeof(offset_pair_type)));
        cudaCheckError();

        InitIndexPair<<<block_num, thread_num>>>(index_pair, QT, edge_capacity_, cache_expand, Kg, device_idx_ / Kg);
        InitOffsetPair<<<block_num, thread_num>>>(offset_pair, QT, edge_capacity_, cache_expand, Kg);

        edge_index_map_->insert(index_pair, (index_pair + edge_capacity_ * cache_expand), stream);
        cudaCheckError();

        edge_offset_map_->insert(offset_pair, (offset_pair + edge_capacity_ * cache_expand), stream);

        cudaCheckError();
        cudaDeviceSynchronize();
        cudaFree(index_pair);
        cudaFree(offset_pair);
    }

    void HybridInsert(int32_t *QF, int32_t cpu_cache_capacity, int32_t gpu_cache_capacity) override
    { // only feature now
        cudaSetDevice(device_idx_);
        cudaCheckError();

        cudaMalloc(&pair_, int64_t(int64_t(cpu_cache_capacity + gpu_cache_capacity) * sizeof(pair_type)));
        cudaCheckError();
        dim3 block_num(80, 1);
        dim3 thread_num(1024, 1);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        HybridInitPair<<<block_num, thread_num>>>(pair_, QF, cpu_cache_capacity, gpu_cache_capacity);
        cudaCheckError();
        node_map_->insert(pair_, (pair_ + (cpu_cache_capacity + gpu_cache_capacity)), stream);
        cudaCheckError();
        // if(success){
        //     std::cout<<"Feature Cache Successfully Initialized\n";
        // }
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaFree(pair_);
        cudaCheckError();
        // cudaFree(cache_ids_);
        // cudaCheckError();
        // cudaFree(cache_offset_);
        // cudaCheckError();
    }

    void AccessCount(
        int32_t *d_key,
        int32_t num_keys,
        void *stream) override
    {
    }

    unsigned long long int *GetNodeAccessedMap()
    {
        return node_access_time_;
        // return nullptr;
    }

    unsigned long long int *GetEdgeAccessedMap()
    {
        // return edge_access_time_;
        return nullptr;
    }

    void FindFeat(
        int32_t *sampled_ids,
        int32_t *cache_offset,
        int32_t *node_counter,
        int32_t op_id,
        void *stream) override
    {
        int32_t *h_node_counter = (int32_t *)malloc(64);
        cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);

        int32_t node_off = h_node_counter[(op_id % INTRABATCH_CON) * 2];
        int32_t batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
        if (batch_size == 0)
        {
            std::cout << "invalid batchsize for feature extraction " << h_node_counter[(op_id % INTRABATCH_CON) * 2] << " " << h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1] << "\n";
            return;
        }
        node_map_->find(sampled_ids + node_off, sampled_ids + (node_off + batch_size), cache_offset, static_cast<cudaStream_t>(stream));
        // if(find_iter_ % 500 == 0){
        //     cudaMemsetAsync(d_global_count_, 0, 4, static_cast<cudaStream_t>(stream));
        //     dim3 block_num(48, 1);
        //     dim3 thread_num(1024, 1);
        //     feature_cache_hit<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(cache_offset, batch_size, d_global_count_);
        //     cudaMemcpy(h_global_count_, d_global_count_, 4, cudaMemcpyDeviceToHost);
        //     h_cache_hit_ += h_global_count_[0];
        //     if(op_id == 8){
        //         std::cout<<device_idx_<<" Feature Cache Hit: "<<(h_cache_hit_ * 1.0 / h_node_counter[INTRABATCH_CON * 2 + 1])<<std::endl;
        //         h_cache_hit_ = 0;
        //     }
        // }
        if (op_id == 8)
        {
            // std::cout<<device_idx_<<" Feature Cache Hit: "<<h_cache_hit_<<" "<<(h_cache_hit_ * 1.0 / h_node_counter[9])<<std::endl;
            // h_cache_hit_ = 0;
            find_iter_++;
            // std::cout<<"find_iter "<<find_iter_<<std::endl;
        }
    }

    void FindTopo(int32_t *input_ids,
                  char *partition_index,
                  int32_t *partition_offset,
                  int32_t batch_size,
                  int32_t op_id,
                  void *strm_hdl,
                  int32_t device_id) override
    {
        edge_index_map_->find(input_ids, input_ids + batch_size, partition_index, static_cast<cudaStream_t>(strm_hdl));
        edge_offset_map_->find(input_ids, input_ids + batch_size, partition_offset, static_cast<cudaStream_t>(strm_hdl));

        // if(find_iter_[device_id] % 500 == 0){
        //     cudaMemsetAsync(d_global_count_[device_id], 0, 4, static_cast<cudaStream_t>(strm_hdl));
        //     dim3 block_num(48, 1);
        //     dim3 thread_num(1024, 1);
        //     cache_hit<<<block_num, thread_num, 0, static_cast<cudaStream_t>(strm_hdl)>>>(partition_index, batch_size, d_global_count_[device_id]);
        //     cudaMemcpy(h_global_count_[device_id], d_global_count_[device_id], 4, cudaMemcpyDeviceToHost);
        //     h_cache_hit_[device_id] += ((h_global_count_[device_id])[0]);
        //     h_batch_size_[device_id] += batch_size;
        //     if(op_id == 4){
        //         std::cout<<device_id<<" Topo Cache Hit: "<<h_cache_hit_[device_id]<<" "<<(h_cache_hit_[device_id] * 1.0 / h_batch_size_[device_id])<<std::endl;
        //         h_cache_hit_[device_id] = 0;
        //         h_batch_size_[device_id] = 0;
        //     }
        // }
        // if(op_id == 4){
        //     find_iter_[device_id] += 1;
        // }
    }

    int32_t MaxIdNum() override
    {
        return max_ids_;
    }

private:
    int32_t device_idx_;
    int32_t device_count_;
    int32_t total_num_nodes_;

    unsigned long long int *node_access_time_;
    unsigned long long int *edge_access_time_;
    int32_t train_step_;
    int32_t iter_;

    int32_t max_ids_; // for allocating feature buffer

    bght::bcht<int32_t, int32_t> *node_map_;
    bght::bcht<int32_t, int32_t> *pos_map_;

    bght::bcht<int32_t, char> *edge_index_map_;
    bght::bcht<int32_t, int32_t> *edge_offset_map_;

    int32_t node_capacity_;
    int32_t edge_capacity_;

    int32_t *cache_ids_;
    int32_t *cache_offset_;
    pair_type *pair_;
    pair_type *graph_pair_;

    int32_t *d_global_count_;
    int32_t *h_global_count_;
    int32_t h_cache_hit_;
    int32_t find_iter_;

    int32_t *start_ptr_;
    int32_t *stop_ptr_;
};

CacheController *NewPreSCCacheController(int32_t train_step, int32_t device_count)
{
    return new PreSCCacheController(train_step, device_count);
}

void UnifiedCache::Initialize(
    int64_t cache_memory,
    int32_t float_feature_len,
    int32_t train_step,
    int32_t device_count,
    int32_t cpu_cache_capacity,
    int32_t gpu_cache_capacity,
    std::string dataset_path)
{
    device_count_ = device_count;
    cache_controller_.resize(device_count_);
    for (int32_t i = 0; i < device_count_; i++)
    {
        CacheController *cctl = NewPreSCCacheController(train_step, device_count_);
        cache_controller_[i] = cctl;
    }
    // std::cout << "Cache Controler Initialize\n";

    if (float_feature_len > 0)
    {
        float_feature_cache_.resize(device_count_);
    }
    cudaCheckError();

    cache_memory_ = cache_memory;
    float_feature_len_ = float_feature_len;
    cpu_cache_capacity_ = cpu_cache_capacity;
    gpu_cache_capacity_ = gpu_cache_capacity;
    is_presc_ = true;
    dataset_path_ = dataset_path;
}

void UnifiedCache::InitializeCacheController(
    int32_t dev_id,
    int32_t total_num_nodes)
{
    cache_controller_[dev_id]->Initialize(dev_id, total_num_nodes);
}

void UnifiedCache::Finalize()
{
    if (!destroy_file((dataset_path_ + "accesstimes").c_str())) {
        std::cerr << "Access Time Delete failed\n";
    }
}

void UnifiedCache::FindFeat(
    int32_t *sampled_ids,
    int32_t *cache_offset,
    int32_t *node_counter,
    int32_t op_id,
    void *stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->FindFeat(sampled_ids, cache_offset, node_counter, op_id, stream);
}

void UnifiedCache::FindTopo(
    int32_t *input_ids,
    char *partition_index,
    int32_t *partition_offset,
    int32_t batch_size,
    int32_t op_id,
    void *strm_hdl,
    int32_t dev_id)
{
    cache_controller_[dev_id]->FindTopo(input_ids, partition_index, partition_offset, batch_size, op_id, strm_hdl, dev_id);
}

void UnifiedCache::HybridInit(FeatureStorage *feature, GraphStorage *graph)
{ // single gpu

    cudaHostAlloc(&cpu_float_features_, int64_t(int64_t(cpu_cache_capacity_) * float_feature_len_ * sizeof(float)), cudaHostAllocMapped);

    // std::cout << "Start selecting cache candidates\n";
    std::vector<unsigned long long int *> node_access_time;
    for (int32_t i = 0; i < device_count_; i++)
    {
        cudaSetDevice(i);
        node_access_time.push_back(cache_controller_[i]->GetNodeAccessedMap());
    }
    // 拿到全局热度global_node_access_time
    int32_t total_num_nodes = feature->TotalNodeNum();
    unsigned long long int *global_node_access_time;
    cudaMalloc(&global_node_access_time, total_num_nodes * sizeof(unsigned long long int));
    cudaMemset(global_node_access_time, 0, total_num_nodes * sizeof(unsigned long long int));

    for (int32_t i = 0; i < device_count_; i++)
    {
        cudaSetDevice(i);
        // 假设每个 GPU 的 node_access_time 已经是设备指针
        accumulate<<<(total_num_nodes + 1023) / 1024, 1024>>>(global_node_access_time, node_access_time[i], total_num_nodes);
        cudaDeviceSynchronize();
        cudaCheckError();
    }
    unsigned long long* h_node_access_time = (unsigned long long*)malloc(int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
    cudaMemcpy(h_node_access_time, global_node_access_time, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (!dump_access_times_to_txt((dataset_path_ + "accesstimes").c_str(), h_node_access_time,
                                total_num_nodes)) {
        std::cerr << "Dump failed!\n";
        std::free(h_node_access_time);
        return;
    }

    // std::cout << "Successfully wrote " << total_num_nodes
    //         << " lines to " << dataset_path_.c_str() << '\n';

    std::free(h_node_access_time);

    // for (int i = 0; i < device_count_; i++)
    // {
    //     cudaSetDevice(i);
    //     cudaCheckError();
    //     int32_t total_num_nodes = feature->TotalNodeNum();
    //     total_num_nodes_ = total_num_nodes;
    //     int32_t *node_cache_order;
    //     cudaMalloc(&node_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    //     cudaCheckError();
    //     init_cache_order<<<80, 1024>>>(node_cache_order, total_num_nodes);
    //     thrust::sort_by_key(thrust::device, global_node_access_time, global_node_access_time + total_num_nodes, node_cache_order, thrust::greater<unsigned long long int>());
    //     cudaCheckError();
    //     QF_.push_back(node_cache_order);
    // }

    for(int32_t i = 0; i < device_count_; i++){
        cudaSetDevice(i);
        cudaCheckError();
        int32_t total_num_nodes = feature->TotalNodeNum();
        total_num_nodes_ = total_num_nodes;
        int32_t* node_cache_order;
        // cudaMalloc(&node_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
        cudaHostAlloc(&node_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)),cudaHostAllocMapped);
        cudaCheckError();
        init_cache_order<<<80, 1024>>>(node_cache_order, total_num_nodes);
        thrust::sort_by_key(thrust::device, node_access_time[i], node_access_time[i] + total_num_nodes, node_cache_order, thrust::greater<unsigned long long int>());
        cudaCheckError();
        QF_.push_back(node_cache_order);
    }

    for (int i = 0; i < device_count_; i++)
    {
        cache_controller_[i]->InitializeMap(gpu_cache_capacity_ + cpu_cache_capacity_, 100);
        cache_controller_[i]->HybridInsert(QF_[i], cpu_cache_capacity_, gpu_cache_capacity_);
    }

    d_float_feature_cache_ptr_.resize(device_count_);

    for (int32_t i = 0; i < device_count_; i++)
    {
        cudaSetDevice(i);
        // float** new_ptr;
        // cudaMalloc(&new_ptr, 1 * sizeof(float*));
        // d_float_feature_cache_ptr_[i] = new_ptr;

        if (float_feature_len_ > 0)
        {
            float *new_float_feature_cache;
            cudaMalloc(&new_float_feature_cache, int64_t(int64_t(int64_t(gpu_cache_capacity_) * float_feature_len_) * sizeof(float)));
            // std::cout << "Allocate GPU Feature Cache" << gpu_cache_capacity_ << "\n";
            // FeatFillUp<<<128, 1024>>>(gpu_cache_capacity_, float_feature_len_, new_float_feature_cache, cpu_float_feature, QF_[i], Kg_, j);
            float_feature_cache_[i] = new_float_feature_cache;
            // init_feature_cache<<<1,1>>>(d_float_feature_cache_ptr_[i], new_float_feature_cache, i);          //j: device id in clique
            // cudaCheckError();
        }
    }

    cudaDeviceSynchronize();
    is_presc_ = false;

    // std::cout << "Finish load feature cache\n";
}

int32_t UnifiedCache::NodeCapacity(int32_t dev_id)
{
    return node_capacity_[dev_id / Kg_];
}

int32_t UnifiedCache::CPUCapacity()
{
    return cpu_cache_capacity_;
}

int32_t UnifiedCache::GPUCapacity()
{
    return gpu_cache_capacity_; // single gpu version
}

float *UnifiedCache::Float_Feature_Cache(int32_t dev_id)
{
    return float_feature_cache_[dev_id];
}

float **UnifiedCache::Global_Float_Feature_Cache(int32_t dev_id)
{
    return d_float_feature_cache_ptr_[dev_id];
}

int32_t UnifiedCache::MaxIdNum(int32_t dev_id)
{
    return cache_controller_[dev_id]->MaxIdNum();
}

unsigned long long int *UnifiedCache::GetEdgeAccessedMap(int32_t dev_id)
{
    return cache_controller_[dev_id]->GetEdgeAccessedMap();
}

void UnifiedCache::CacheProfiling(
    int32_t *sampled_ids,
    int32_t *agg_src_id,
    int32_t *agg_dst_id,
    int32_t *agg_src_off,
    int32_t *agg_dst_off,
    int32_t *node_counter,
    int32_t *edge_counter,
    void *stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->CacheProfiling(sampled_ids, agg_src_id, agg_dst_id, agg_src_off, agg_dst_off, node_counter, edge_counter, is_presc_, stream);
}

void UnifiedCache::AccessCount(
    int32_t *d_key,
    int32_t num_keys,
    void *stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->AccessCount(d_key, num_keys, stream);
}

void UnifiedCache::FeatCacheLookup(int32_t *sampled_ids, int32_t *cache_index,
                                   int32_t *node_counter, float *dst_float_buffer,
                                   int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl)
{
    dim3 block_num(64, 1);
    dim3 thread_num(1024, 1);
    // float** gpu_float_feature     = Global_Float_Feature_Cache(dev_id);
    int32_t cpu_cache_capacity = CPUCapacity();
    int32_t gpu_cache_capacity = GPUCapacity();
    // printf("test_lookup\n");
    feat_cache_lookup<<<block_num, thread_num, 0, (strm_hdl)>>>(
        cpu_float_features_, float_feature_cache_[dev_id], float_feature_len_,
        sampled_ids, cache_index,
        cpu_cache_capacity, gpu_cache_capacity,
        node_counter, dst_float_buffer,
        op_id);
}
