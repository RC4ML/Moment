# Hyperion is an Out-of-core GNN Training System with GPU-initiated Asynchronous Disk IO Stack [ICDE 25]
With the rapid expansion of large models and datasets, GPU memory capacity is increasingly becoming a bottleneck. Out-of-core (disk-based) systems offer a cost-efficient solution to this challenge. Modern NVMe SSDs, with their terabytes of capacity and throughput of up to 7 GB/s (PCIe 4.0), present a promising option for managing large-scale models/data.

We introduce Hyperion, a cost-efficient out-of-core GNN training system designed to handle terabyte-scale graphs, which can achieve in-memory-like GNN training throughput with some cheap NVMe SSDs. At its core is a GPU-initiated asynchronous disk IO stack, optimized to fully leverage the performance of modern NVMe SSDs. Beyond GNN systems, this IO stack is versatile and can be extended to other applications, such as DLRM, KVCache, and RAG systems (if you need GPU to orchestrate disk access directly). To support broader adoption, we also provide the IO stack as a standalone component for independent use.


```
$ git clone https://github.com/RC4ML/Hyperion.git
```

## 1. Preparation of Hardware and Software 
### 1.1 Hardware Recommended
Local bare-metal machine.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | SSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 768GB | PCIe 4.0x16| 80GB-PCIe-A100 | Intel P5510, Sansumg 980 pro |
### 1.2 Tested Software Environments
1. Nvidia Driver Version: 515.43.04
2. CUDA 11.7 - 12.4
3. GCC/G++ 11.4.0
4. OS: Ubuntu 22.04, Linux version 5.15.72 (customized, see BaM's requirements)
5. pytorch (according to your CUDA toolkit version), torchmetrics
```
$ pip install torch-cu124
```
6. dgl 1.1.0 - 2.x (according to pytorch and CUDA version)
```
$ pip install dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
### 1.3 Driver for GPU Direct Storage Access
We reuse the BaM (https://github.com/ZaidQureshi/bam) Kernel Module to enable GPU Direct Storage Access.
#### Step 1 Disable IOMMU in Linux
$ cat /proc/cmdline | grep iommu
If either iommu=on or intel_iommu=on is found by grep, the IOMMU is enabled.
Disable it by removing iommu=on and intel_iommu=on from the CMDLINE variable in /etc/default/grub and then reconfiguring GRUB. The next time you reboot, the IOMMU will be disabled.
#### Step 2 Compiling Nvidia Driver Kernel Symbols
```
$ cd /usr/src/nvidia-515.43.04/
$ sudo make
```
#### Step 3 Building BaM Project
From the project root directory, do the following:
```
$ git clone https://github.com/ZaidQureshi/bam.git
$ cd bam
$ git submodule update --init --recursive
$ mkdir -p build; cd build
$ cmake ..
$ make libnvm                         # builds library
$ make benchmarks                     # builds benchmark program
$ cd build/module
$ make
```
#### Step 4 Loading/Unloading the Kernel Module
Unbind the NVMe drivers according to your needs (customize unload_ssd.py):
```
$ sudo python /path/Hyperion/unload_ssd.py 
$ cd /path/BaM/build/module
$ sudo make load
```
Check whether it's successful
This should create a /dev/libnvm* device file for each controller that isn't bound to the NVMe driver.
```
$ ls /dev/
```
The module can be unloaded from the project root directory with the following:
```
$ cd build/module
$ sudo make unload
```
## 2. Standalone Usage of GPU-Initiated Asynchronous Disk IO Stack
GPU-initiated asynchronous disk IO stack is the key component of Hyperion, maximizing the throughput of GPU-initiated direct disk access with only a few GPU cores.

You can try to integrate it into your own AI system.

<img src="https://github.com/user-attachments/assets/894134f3-c0be-46da-b43d-f0ddd709f604" alt="IO Stack Figure" style="width:50%; height:auto;">

### Quick Start
```
$ cd IOStack
$ make
$ sudo ./test
```
### User Interface
In this part, we introduce the usage of the IO stack. The code of the IO stack is head-only and can be easily integrated into your projects. We regard NVMe SSDs as block-device, i.e., users should know which SSD logic blocks to access. We define `ITEM_SIZE` as the minimal logic block size. `ITEM_SIZE` is usually 512 bytes. `NUM_LBS_PER_SSD` is the total number of logic blocks in each SSD.
#### Initialize IO stack
Initialize the IO stack with configurable parameters. `num_ssds` is the total number of SSDs and `num_queue_per_ssd` is the SQ/CQ number of each SSD. `io_submission_TB_num` and `io_completion_TB_num` are the thread block number of IO submission kernels and IO completion kernels, respectively. In our evaluated platforms, setting `io_submission_TB_num` to **1** and setting `io_completion_TB_num` to **32** is sufficient to maximize SSD throughput (even for 12 * SSDs).
```cpp
IOStack(int num_ssds, int num_queue_per_ssd, int io_submission_TB_num, int io_completion_TB_num)
```
#### IO Submission
The IO stack supports submitting multiple micro-batches of IO requests and only handling their completion once. Call the io_submission with the IO requests array in GPU memory and the number of requests.
You can also assign the asynchronous CUDA stream for the io_submission.
```cpp
void IOStack::io_submission(IOReq *reqs, int num_reqs, cudaStream_t stream);
```
Each IO request needs to be organized in the following structure. `start_lb` represents the logic block index in the SSDs. The `start_lb` of i-th logic block in the j-th SSD should be `NUM_LBS_PER_SSD*j+i`. Each IO request can read `num_items` logically-continuous blocks (`num_items` is not larger than `MAX_ITEMS`). For the k-th logic block in a request, users can assign its output virtual address, i.e., `dest_addr[k]` in the GPU global memory. 
```cpp
struct IOReq
{
    uint64_t start_lb;
    app_addr_t dest_addr[MAX_ITEMS];
    int num_items;
    ...
};
```
#### IO Completion
Call the `io_completion` at a suitable time. It will handle the completion of all previous IO requests.
```cpp
void IOStack::io_completion(cudaStream_t stream);
```
#### Example:
Concurrently initiate the IO requests and GNN computation. The disk access (stream1) can be overlapped with GNN computation (stream2).
```cpp
io_submission(reqs_micro_batch1, num_reqs_batch1, stream1);
GNN_Kernel(..., stream2) ## GNN computation
io_submission(reqs_micro_batch2, num_reqs_batch2, stream1);
GNN_Kernel(..., stream2) ## GNN computation
io_submission(reqs_micro_batch3, num_reqs_batch3, stream1);
GNN_Kernel(..., stream2) ## GNN computation
io_completion(stream1);
```
## 3. Prepare Datasets for GNN Training
Datasets are from OGB (https://ogb.stanford.edu/), Standford-snap (https://snap.stanford.edu/), and Webgraph (https://webgraph.di.unimi.it/).
Here is an example of preparing datasets for Hyperion.

### Uk-Union Datasets
Refer to README in dataset directory for more instructions
```
$ bash prepare_datasets.sh
```
## 4. Run Hyperion
### 4.1 Build Hyperion from Source
```
$ bash build.sh
```
There are two steps to train a GNN model in Hyperion. In these steps, you need to change to **root**/**sudo** user for GPU Direct SSD Access.
### 4.2 Start Hyperion Server

```
$ sudo python Hyperion_server.py --dataset_path 'dataset' --dataset_name ukunion --train_batch_size 8000 --fanout [25,10] --epoch 2 
```

### 4.3 Run Hyperion Training
#### (Optional) Configure SM Utilization of Training Backend:
```
$ export CUDA_VISIBLE_DEVICES=0         # Example using GPU0, adjust for other GPUs as needed
$ sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS  # Set GPU0 to exclusive process mode
$ sudo nvidia-cuda-mps-control -d            # Start the MPS service
# ====== check =========
$ ps -ef | grep mps                     # After starting successfully, the corresponding process can be seen
# ====== stop =========
$ sudo nvidia-smi -i 0 -c DEFAULT       # Restore GPU to default mode
$ echo quit | nvidia-cuda-mps-control   # Stop the MPS service
```
After Hyperion outputs "System is ready for serving", then start training by: 
```
$ sudo python training_backend/Hyperion_graphsage.py --class_num 2  --features_num 128 --hidden_dim 256 --hops_num 2 --epoch 2
```

## Cite this work
If Hyperion and the IO stack are helpful for your research, please cite our work

```
@inproceedings{sun2024hyperion,
  title={Hyperion: Optimizing SSD Access is All You Need to Enable Cost-efficient Out-of-core GNN Training}, 
  author={Jie Sun and Mo Sun and Zheng Zhang and Jun Xie and Zuocheng Shi and Zihan Yang and Jie Zhang and Fei Wu and Zeke Wang},
  year={2025},
  booktitle={ICDE}
}
```

