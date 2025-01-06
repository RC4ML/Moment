#include "cache.cuh"
#include "cache_impl.cuh"
#include <algorithm>  
class PreSCCacheController : public CacheController {
public:
    PreSCCacheController(int32_t train_step, int32_t device_count){
       train_step_ = train_step;
       device_count_ = device_count;
    }

    virtual ~PreSCCacheController(){}

    void Initialize(
        int32_t dev_id,
        int32_t total_num_nodes) override
    {
        device_idx_ = dev_id;
        total_num_nodes_ = total_num_nodes;
        cudaSetDevice(dev_id);

        cudaMalloc(&node_access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaMemset(node_access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaCheckError();
        cudaMalloc(&edge_access_time_, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaMemset(edge_access_time_, 0, int64_t(int64_t(total_num_nodes) * sizeof(unsigned long long int)));
        cudaCheckError();

        iter_ = 0;
        max_ids_ = 0;
        cudaMalloc(&d_global_count_, 4);
        h_global_count_ = (int32_t*)malloc(4);
        find_iter_ = 0;
        h_cache_hit_ = 0;
    }

    void Finalize() override {
        // pos_map_->clear();
    }

    void CacheProfiling(
                    int32_t* sampled_ids,
                    int32_t* agg_src_id,
                    int32_t* agg_dst_id,
                    int32_t* agg_src_off,
                    int32_t* agg_dst_off,
                    int32_t* node_counter,
                    int32_t* edge_counter,
                    bool     is_presc,
                    void*    stream) override
    {
        dim3 block_num(32, 1);
        dim3 thread_num(1024, 1);

        if(is_presc){
            int32_t* h_node_counter = (int32_t*)malloc(16*sizeof(int32_t));
            cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);
            HotnessMeasure<<<block_num, thread_num, 0, static_cast<cudaStream_t>(stream)>>>(sampled_ids, node_counter, node_access_time_);

            if(h_node_counter[INTRABATCH_CON * 2 + 1] > max_ids_){
                max_ids_ = h_node_counter[INTRABATCH_CON * 2 + 1];
            }
            if(iter_ == (train_step_ - 1)){
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

        edge_index_map_ = new bght::bcht<int32_t, int32_t>(int64_t(edge_capacity_ * device_count_) * 2, invalid_key, -2);
        cudaCheckError();

        edge_offset_map_ = new bght::bcht<int32_t, int32_t>(int64_t(edge_capacity_ * device_count_) * 2, invalid_key, invalid_value);
        cudaCheckError();
    }

    void UnifiedInsert(int32_t* QF, int32_t* QT, int32_t gpu_feat_num, int32_t cpu_feat_num, int32_t gpu_topo_num, int32_t cpu_topo_num) override {//only feature now
        cudaSetDevice(device_idx_);
        cudaCheckError();

        cudaMalloc(&pair_, int64_t(int64_t(cpu_feat_num + gpu_feat_num) * sizeof(pair_type)));
        cudaCheckError();
        dim3 block_num(80, 1);
        dim3 thread_num(1024, 1);
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        HybridInitPair<<<block_num, thread_num>>>(pair_, QF, cpu_feat_num, gpu_feat_num);
        cudaCheckError();
        node_map_->insert(pair_, (pair_ + (gpu_feat_num + cpu_feat_num)), stream);
        cudaCheckError();
        cudaDeviceSynchronize();
        cudaCheckError();
        cudaFree(pair_);
        cudaCheckError();

        index_pair_type* index_pair;
        offset_pair_type* offset_pair;
        cudaMalloc(&index_pair, int64_t(int64_t(gpu_topo_num + cpu_topo_num) * sizeof(index_pair_type)));
        cudaCheckError();
        cudaMalloc(&offset_pair, int64_t(int64_t(gpu_topo_num + cpu_topo_num) * sizeof(offset_pair_type)));
        cudaCheckError();

        HybridInitIndexPair<<<block_num, thread_num>>>(index_pair, QT, cpu_topo_num, gpu_topo_num);
        HybridInitOffsetPair<<<block_num, thread_num>>>(offset_pair, QT, cpu_topo_num, gpu_topo_num);

        edge_index_map_->insert(index_pair, (index_pair + int64_t(gpu_topo_num + cpu_topo_num)), stream);
        cudaCheckError();

        edge_offset_map_->insert(offset_pair, (offset_pair + int64_t(gpu_topo_num + cpu_topo_num)), stream);

        cudaFree(index_pair);
        cudaFree(offset_pair);
    }


    void AccessCount(
        int32_t* d_key,
        int32_t num_keys,
        void* stream) override
    {}

    unsigned long long int* GetNodeAccessedMap() {
        return node_access_time_;
    }

    unsigned long long int* GetEdgeAccessedMap() {
        return edge_access_time_;
    }

    void FindFeat(
        int32_t* sampled_ids,
        int32_t* cache_offset,
        int32_t* node_counter,
        int32_t op_id,
        void* stream) override
    {
        int32_t* h_node_counter = (int32_t*)malloc(64);
        cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);

        int32_t node_off = h_node_counter[(op_id % INTRABATCH_CON) * 2];
        int32_t batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
        if(batch_size == 0){
            std::cout<<"invalid batchsize for feature extraction "<<h_node_counter[(op_id % INTRABATCH_CON) * 2]<<" "<<h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1]<<"\n";
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
        if(op_id == 8){
            // std::cout<<device_idx_<<" Feature Cache Hit: "<<h_cache_hit_<<" "<<(h_cache_hit_ * 1.0 / h_node_counter[9])<<std::endl;    
            // h_cache_hit_ = 0;
            find_iter_++;
            // std::cout<<"find_iter "<<find_iter_<<std::endl;
        }
    }

    void FindTopo(int32_t* input_ids, 
                    int32_t* partition_index, 
                    int32_t* partition_offset, 
                    int32_t batch_size, 
                    int32_t op_id, 
                    void* strm_hdl, 
                    int32_t device_id) override {
        edge_index_map_->find(input_ids, input_ids + batch_size, partition_index, static_cast<cudaStream_t>(strm_hdl));
        edge_offset_map_->find(input_ids, input_ids + batch_size, partition_offset, static_cast<cudaStream_t>(strm_hdl));
    }

    // void FindTopoSSD(int32_t* sampled_ids,
    //                 int32_t* cache_offset,
    //                 int32_t* node_counter,
    //                 int32_t op_id,
    //                 void* stream) override {
    //     int32_t* h_node_counter = (int32_t*)malloc(64);
    //     cudaMemcpy(h_node_counter, node_counter, 64, cudaMemcpyDeviceToHost);

    //     int32_t node_off = h_node_counter[(op_id % INTRABATCH_CON) * 2];
    //     int32_t batch_size = h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1];
    //     if(batch_size == 0){
    //         std::cout<<"invalid batchsize for feature extraction "<<h_node_counter[(op_id % INTRABATCH_CON) * 2]<<" "<<h_node_counter[(op_id % INTRABATCH_CON) * 2 + 1]<<"\n";
    //         return;
    //     }
    //     edge_index_map_->find(sampled_ids + node_off, sampled_ids + (node_off + batch_size), cache_offset, static_cast<cudaStream_t>(stream));
    // }

    int32_t MaxIdNum() override
    {
        return max_ids_;
    }

private:
    int32_t device_idx_;
    int32_t device_count_;
    int32_t total_num_nodes_;

    unsigned long long int* node_access_time_;
    unsigned long long int* edge_access_time_;
    int32_t train_step_;
    int32_t iter_;

    int32_t max_ids_;//count maximal number of samples, for allocating feature buffer

    bght::bcht<int32_t, int32_t>* node_map_;
    bght::bcht<int32_t, int32_t>* edge_index_map_;
    bght::bcht<int32_t, int32_t>* edge_offset_map_;

    int32_t node_capacity_;
    int32_t edge_capacity_;

    pair_type* pair_;
    pair_type* graph_pair_;

    int32_t* d_global_count_;
    int32_t* h_global_count_;
    int32_t  h_cache_hit_;
    int32_t  find_iter_;
};

CacheController* NewPreSCCacheController(int32_t train_step, int32_t device_count)
{
    return new PreSCCacheController(train_step, device_count);
}

void UnifiedCache::Initialize(
    int32_t float_feature_len,
    int32_t train_step, 
    int32_t device_count,
    int64_t cpu_topo_size,
    int64_t gpu_topo_size,
    int64_t cpu_feat_size,
    int64_t gpu_feat_size
    )
{
    device_count_ = device_count;
    cache_controller_.resize(device_count_);
    for(int32_t i = 0; i < device_count_; i++){
        CacheController* cctl = NewPreSCCacheController(train_step, device_count_);
        cache_controller_[i] = cctl;
    }
    std::cout<<"Cache Controler Initialize\n";

    if(float_feature_len > 0){
        float_feature_cache_.resize(device_count_);
    }
    cudaCheckError();

    float_feature_len_ = float_feature_len;
    cpu_topo_size_ = cpu_topo_size;
    gpu_topo_size_ = gpu_topo_size;
    cpu_feat_size_ = cpu_feat_size;
    gpu_feat_size_ = gpu_feat_size;
    is_presc_ = true;
}

void UnifiedCache::InitializeCacheController(
    int32_t dev_id,
    int32_t total_num_nodes)
{
    cache_controller_[dev_id]->Initialize(dev_id, total_num_nodes);
}

void UnifiedCache::Finalize(int32_t dev_id){
    cudaSetDevice(dev_id);
    cache_controller_[dev_id]->Finalize();
}

void UnifiedCache::FindFeat(
    int32_t* sampled_ids,
    int32_t* cache_offset,
    int32_t* node_counter,
    int32_t op_id,
    void* stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->FindFeat(sampled_ids, cache_offset, node_counter, op_id, stream);
}


void UnifiedCache::FindTopo(
    int32_t* input_ids,
    int32_t* partition_index,
    int32_t* partition_offset, 
    int32_t batch_size, 
    int32_t op_id, 
    void* strm_hdl,
    int32_t dev_id)
{
    cache_controller_[dev_id]->FindTopo(input_ids, partition_index, partition_offset, batch_size, op_id, strm_hdl, dev_id);
}

void UnifiedCache::HybridInit(FeatureStorage* feature, GraphStorage* graph){//single gpu 

    std::cout<<"Start selecting cache candidates\n";
    std::vector<unsigned long long int*> node_access_time;
    for(int32_t i = 0; i < 1; i++){
        node_access_time.push_back(cache_controller_[i]->GetNodeAccessedMap());
    }
    std::vector<unsigned long long int*> edge_access_time;
    for(int32_t i = 0; i < 1; i++){
        edge_access_time.push_back(cache_controller_[i]->GetEdgeAccessedMap());
    }
    cudaCheckError();
    int32_t total_num_nodes = feature->TotalNodeNum();
    total_num_nodes_ = total_num_nodes;
    int32_t* node_cache_order;
    cudaMalloc(&node_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    cudaCheckError();

    init_cache_order<<<80, 1024>>>(node_cache_order, total_num_nodes);

    thrust::sort_by_key(thrust::device, node_access_time[0], node_access_time[0] + total_num_nodes, node_cache_order, thrust::greater<unsigned long long int>());
    cudaCheckError();

    QF_.push_back(node_cache_order);
    
    int32_t* edge_cache_order;
    cudaMalloc(&edge_cache_order, int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
    cudaCheckError();

    init_cache_order<<<80, 1024>>>(edge_cache_order, total_num_nodes);

    thrust::sort_by_key(thrust::device, edge_access_time[0], edge_access_time[0] + total_num_nodes, edge_cache_order, thrust::greater<unsigned long long int>());
    cudaCheckError();
    cudaFree(edge_access_time[0]);
    cudaFree(node_access_time[0]);
    QT_.push_back(edge_cache_order);

    int64_t* csr_index = graph->GetCSRNodeIndexCPU();
    uint64_t* d_edge_mem;
    cudaMalloc(&d_edge_mem, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    GetEdgeMem<<<80, 1024>>>(QT_[0], d_edge_mem, total_num_nodes, csr_index);
    cudaCheckError();
    uint64_t* d_edge_mem_prefix;
    cudaMalloc(&d_edge_mem_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t))); 
    thrust::inclusive_scan(thrust::device, d_edge_mem, d_edge_mem + total_num_nodes, d_edge_mem_prefix);
    cudaCheckError();
    uint64_t* h_edge_mem_prefix = (uint64_t*)malloc(int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)));
    cudaMemcpy(h_edge_mem_prefix, d_edge_mem_prefix, int64_t(int64_t(total_num_nodes)*sizeof(uint64_t)), cudaMemcpyDeviceToHost);
    cudaFree(d_edge_mem);
    cudaFree(d_edge_mem_prefix);

    gpu_topo_num_ = std::min((std::lower_bound(h_edge_mem_prefix, h_edge_mem_prefix + total_num_nodes, gpu_topo_size_) - h_edge_mem_prefix), int64_t(total_num_nodes));
    cpu_topo_num_ = std::min((std::lower_bound(h_edge_mem_prefix, h_edge_mem_prefix + total_num_nodes, (gpu_topo_size_ + cpu_topo_size_)) - h_edge_mem_prefix - gpu_topo_num_), int64_t(total_num_nodes));
    
    gpu_feat_num_  = int(gpu_feat_size_ / (float_feature_len_ * sizeof(float)));
    cpu_feat_num_  = int(cpu_feat_size_ / (float_feature_len_ * sizeof(float)));
    std::cout<<"GPU Topo Num: "<<gpu_topo_num_<<" CPU Topo Num: "<<cpu_topo_num_<<"\n";
    std::cout<<"GPU Feat Num: "<<gpu_feat_num_<<" CPU Feat Num: "<<cpu_feat_num_<<"\n";
    graph->HyrbidGraphCache(QT_[0], cpu_topo_num_, gpu_topo_num_);

    cache_controller_[0]->InitializeMap(gpu_feat_num_ + cpu_feat_num_, gpu_topo_num_ + cpu_topo_num_);

    cache_controller_[0]->UnifiedInsert(QF_[0], QT_[0], gpu_feat_num_, cpu_feat_num_, gpu_topo_num_, cpu_topo_num_);

    cudaHostAlloc(&cpu_float_features_, int64_t(int64_t(cpu_feat_num_) * float_feature_len_ * sizeof(float)), cudaHostAllocMapped);
    cudaSetDevice(0);

    d_float_feature_cache_ptr_.resize(1);

    for(int32_t i = 0; i < 1; i++){
        float** new_ptr;
        cudaMalloc(&new_ptr, 1 * sizeof(float*));
        d_float_feature_cache_ptr_[i] = new_ptr;
    }

    if(float_feature_len_ > 0){
        float* new_float_feature_cache;
        cudaMalloc(&new_float_feature_cache, int64_t(int64_t(int64_t(gpu_feat_num_) * float_feature_len_) * sizeof(float)));
        // std::cout<<"Allocate GPU Feature Cache"<<gpu_feat_num_<<"\n";
        // FeatFillUp<<<128, 1024>>>(gpu_cache_capacity_, float_feature_len_, new_float_feature_cache, cpu_float_feature, QF_[i], Kg_, j);
        float_feature_cache_[0] = new_float_feature_cache;
        init_feature_cache<<<1,1>>>(d_float_feature_cache_ptr_[0], new_float_feature_cache, 0);//j: device id in clique
        cudaCheckError();
    }

    cudaDeviceSynchronize();
    is_presc_ = false;

    std::cout<<"Finish load feature cache\n";
}

int32_t UnifiedCache::NodeCapacity(int32_t dev_id){
    return node_capacity_[dev_id / Kg_];
}

int32_t UnifiedCache::CPUCapacity(){
    return cpu_cache_capacity_;
}

int32_t UnifiedCache::GPUCapacity(){
    return gpu_cache_capacity_;//single gpu version
}

float* UnifiedCache::Float_Feature_Cache(int32_t dev_id)
{
    return float_feature_cache_[dev_id];
}

float** UnifiedCache::Global_Float_Feature_Cache(int32_t dev_id)
{
    return d_float_feature_cache_ptr_[dev_id];
}

int32_t UnifiedCache::MaxIdNum(int32_t dev_id){
    return cache_controller_[dev_id]->MaxIdNum();
}

unsigned long long int* UnifiedCache::GetEdgeAccessedMap(int32_t dev_id){
    return cache_controller_[dev_id]->GetEdgeAccessedMap();
}

void UnifiedCache::CacheProfiling(
    int32_t* sampled_ids,
    int32_t* agg_src_id,
    int32_t* agg_dst_id,
    int32_t* agg_src_off,
    int32_t* agg_dst_off,
    int32_t* node_counter,
    int32_t* edge_counter,
    void* stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->CacheProfiling(sampled_ids, agg_src_id, agg_dst_id, agg_src_off, agg_dst_off, node_counter, edge_counter, is_presc_, stream);
}

void UnifiedCache::AccessCount(
    int32_t* d_key,
    int32_t num_keys,
    void* stream,
    int32_t dev_id)
{
    cache_controller_[dev_id]->AccessCount(d_key, num_keys, stream);
}


void UnifiedCache::FeatCacheLookup(int32_t* sampled_ids, int32_t* cache_index,
                                    int32_t* node_counter, float* dst_float_buffer,
                                    int32_t op_id, int32_t dev_id, cudaStream_t strm_hdl){
    dim3 block_num(16, 1);
	dim3 thread_num(1024, 1);
    int32_t cpu_cache_capacity    = cpu_feat_num_;
    int32_t gpu_cache_capacity    = gpu_feat_num_;
    feat_cache_lookup<<<block_num, thread_num, 0, (strm_hdl)>>>(
        cpu_float_features_, float_feature_cache_[0], float_feature_len_,
        sampled_ids, cache_index, 
        cpu_cache_capacity, gpu_cache_capacity,
        node_counter, dst_float_buffer,
        op_id
    );
}