# Moment [SC 25]
```
$ git clone https://github.com/RC4ML/Moment.git
```

## 1. Preparation of Hardware and Software [Skip this section during SC AE]
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
$ sudo python /path/Moment/unload_ssd.py 
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
## 2. Prepare Datasets for GNN Training
Datasets are from OGB (https://ogb.stanford.edu/), Standford-snap (https://snap.stanford.edu/), and Webgraph (https://webgraph.di.unimi.it/).
Here is an example of preparing datasets for Moment.

### Uk-Union Datasets
Refer to README in dataset directory for more instructions
```
$ bash prepare_datasets.sh
```
## 3. Run Moment
### 3.1 Build Moment from Source
```
$ bash build.sh
```
There are two steps to train a GNN model in Moment. In these steps, you need to change to **root**/**sudo** user for GPU Direct SSD Access.
### 3.2 Run Moment's Automatic Module
```
$ sudo python3 automatic_module.py
```
Customize dataset path, GNN feature dimension, number of GPUs, and number of SSDs in the automatic_module.py:
```
file_path = "/share/gnn_data/igb260m/IGB-Datasets/data/"  # Replace with your file path
feature_dim = 1024
num_gpu = 2
num_ssd = 6
```
The automatic module will execute three main steps:
#### 3.2.1 Get hardware and run profiling
Output:
<img width="347" height="444" alt="6fbc296ecc7eba6ff70cd95a2de75fe" src="https://github.com/user-attachments/assets/ba034c66-6503-4450-aafb-dd90ae14ebff" />

#### 3.2.2 Solve maxflow and recommend the best hardware configuration setting
Output:
<img width="289" height="148" alt="a4c0ceeade58389e58c615a9c51f7a4" src="https://github.com/user-attachments/assets/1797729f-9270-4ff4-a43b-5bbef381c251" />

#### 3.2.3 Run DDAk
<img width="153" height="131" alt="c25aa938a37f4c9fbec6332db895f77" src="https://github.com/user-attachments/assets/ff048152-59cf-462f-b061-27e2d4204550" />


### 3.3 Start Moment Server
Execute the following instruction:
```
$ sudo python3 moment_server.py --dataset_name igb --train_batch_size 8000 --fanout [25,10] --epoch 2 
```
Customize the hyperparameters of the Moment server:
```
    argparser.add_argument('--dataset_path', type=str, default="/share/gnn_data/igb260m/IGB-Datasets/data")
    argparser.add_argument('--dataset_name', type=str, default="igb")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--fanout', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--ssd_number', type=int, default=6)
    argparser.add_argument('--num_queues_per_ssd', type=int, default=128)
```
Note that the dataset_path should be the same as the automatic module.
When the system outputs the following, start training in another session:
<img width="158" height="15" alt="1752558735103" src="https://github.com/user-attachments/assets/bd002711-360b-4af8-9608-aa11cbb3f332" />



### 3.4 Run Moment Training
#### (Optional) Configure SM Utilization of Training Backend:
```
$ export CUDA_VISIBLE_DEVICES=0         # Example using GPU0, adjust for other GPUs as needed
$ sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS  # Set GPU0 to exclusive process mode
$ sudo nvidia-cuda-mps-control -d            # Start the MPS service
# ====== check =========
$ ps -ef | grep mps                     # After starting successfully, the corresponding process can be seen
# ====== configure =========
$ export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 # Assign the percentage of action SMs for the training backend
# ====== stop =========
$ sudo nvidia-smi -i 0 -c DEFAULT       # Restore GPU to default mode
$ echo quit | nvidia-cuda-mps-control   # Stop the MPS service
```
After Moment outputs "System is ready for serving", then start training by: 
```
$ sudo python3 training_backend/moment_graphsage.py --class_num 2  --features_num 1024 --hidden_dim 256 --epoch 2
```


