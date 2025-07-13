import heapq
import random
import numpy as np

def allocate_balanced_samples(samples, k, capacities, targets):
    """ Allocate samples to k bins trying to match target access counts and respecting capacities,
        with enhanced balancing for the number of samples in each bin. Additionally, track the bin ID
        and order of each sample within its bin. """
    samples.sort(key=lambda x: x[1], reverse=True)

    bins = [{'samples': [], 'total_access': 0, 'capacity': cap, 'target': tgt} for cap, tgt in zip(capacities, targets)]
    bin_heap = [(0, i) for i in range(k)]  # (current_access_to_target_ratio, bin_index)
    heapq.heapify(bin_heap)

    # Arrays to store the bin ID and the order of each sample within its bin
    sample_bin_ids = [-1] * len(samples)
    sample_orders = [-1] * len(samples)

    # Attempt to distribute samples while balancing both access count and sample count
    for i, (sample, access) in enumerate(samples):
        tried_bins = []
        while bin_heap:
            _, bin_index = heapq.heappop(bin_heap)
            bin = bins[bin_index]

            # Evaluate future state if sample is added
            future_access_ratio = (bin['total_access'] + access) / bin['target']
            future_sample_count = len(bin['samples']) + 1

            if future_sample_count <= bin['capacity'] and future_access_ratio <= 1.1:  # Allow some overage
                bin['samples'].append(sample)
                bin['total_access'] += access
                sample_bin_ids[sample] = bin_index
                sample_orders[sample] = len(bin['samples']) - 1
                heapq.heappush(bin_heap, (future_access_ratio + (future_sample_count / bin['capacity']), bin_index))
                break
            else:
                tried_bins.append((future_access_ratio + (future_sample_count / bin['capacity']), bin_index))

        if not bin_heap:
            # If no bin could accept the sample, re-add bins and try least loaded
            heapq.heapify(tried_bins)
            _, bin_index = heapq.heappop(tried_bins)
            bin = bins[bin_index]
            if len(bin['samples']) < bin['capacity']:
                bin['samples'].append(sample)
                bin['total_access'] += access
                sample_bin_ids[sample] = bin_index
                sample_orders[sample] = len(bin['samples']) - 1
                for bin_info in tried_bins:
                    heapq.heappush(bin_heap, bin_info)
            else:
                print(f"Warning: No capacity left for Sample {sample} with Access {access}.")

        if (i + 1) % (len(samples) // 10) == 0:
            print(f"Distribution Progress: {(i + 1) / len(samples) * 100:.2f}%")

    # Print final distributions
    for i, bin in enumerate(bins):
        print(f"Bin {i+1}: Total Access = {bin['total_access']} (Target = {bin['target']}), Samples = {len(bin['samples'])}/{bin['capacity']}")

    return bins, sample_bin_ids, sample_orders

def allocate_balanced_samples_prior(samples, k, capacities, targets):
    """ Allocate samples to k bins, ensuring that the hottest samples are prioritized to bins 1 and 2. """
    samples.sort(key=lambda x: x[1], reverse=True)  # Sort samples by access count in descending order

    bins = [{'samples': [], 'total_access': 0, 'capacity': cap, 'target': tgt} for cap, tgt in zip(capacities, targets)]
    # Initialize priority for first two bins for hottest samples
    special_bins_priority = [(0, i) for i in range(2)]  # Use only the first two bins initially
    heapq.heapify(special_bins_priority)

    general_bins_priority = [(0, i) for i in range(2, k)]  # The rest of the bins
    heapq.heapify(general_bins_priority)

    # Arrays to store the bin ID and the order of each sample within its bin
    sample_bin_ids = [-1] * len(samples)
    sample_orders = [-1] * len(samples)

    # Allocate hottest samples preferentially to bins 1 and 2
    for i, (sample, access) in enumerate(samples):
        if i < len(samples) // 100:  # Assume top 1% are the hottest
            target_heap = special_bins_priority
        else:
            target_heap = general_bins_priority if not special_bins_priority else special_bins_priority

        allocated = False
        while not allocated and target_heap:
            _, bin_index = heapq.heappop(target_heap)
            bin = bins[bin_index]

            if len(bin['samples']) < bin['capacity']:
                bin['samples'].append(sample)
                bin['total_access'] += access
                sample_bin_ids[sample] = bin_index
                sample_orders[sample] = len(bin['samples']) - 1
                heapq.heappush(target_heap, ((bin['total_access'] / bin['target']) + (len(bin['samples']) / bin['capacity']), bin_index))
                allocated = True

        if not allocated:  # If still not allocated, try to place in any available bin
            for heap in [special_bins_priority, general_bins_priority]:
                while heap and not allocated:
                    _, bin_index = heapq.heappop(heap)
                    bin = bins[bin_index]
                    if len(bin['samples']) < bin['capacity']:
                        bin['samples'].append(sample)
                        bin['total_access'] += access
                        sample_bin_ids[sample] = bin_index
                        sample_orders[sample] = len(bin['samples']) - 1
                        heapq.heappush(heap, ((bin['total_access'] / bin['target']) + (len(bin['samples']) / bin['capacity']), bin_index))
                        allocated = True

        if (i + 1) % (len(samples) // 10) == 0:
            print(f"Distribution Progress: {(i + 1) / len(samples) * 100:.2f}%")

    # Print final distributions
    for i, bin in enumerate(bins):
        print(f"Bin {i+1}: Total Access = {bin['total_access']} (Target = {bin['target']}), Samples = {len(bin['samples'])}/{bin['capacity']}")

    return bins, sample_bin_ids, sample_orders

# This updated function now gives priority to placing the hottest samples in bins 1 and 2.


def save_arrays_to_binary(file_bin_ids, file_orders, bin_ids, orders):
    """ Save arrays to binary files using numpy with int64 format. """
    np_bin_ids = np.array(bin_ids, dtype=np.int64)  # Use int64 for larger range
    np_orders = np.array(orders, dtype=np.int64)   # Use int64 for larger range
    
    # Save to files
    np_bin_ids.tofile(file_bin_ids)
    np_orders.tofile(file_orders)
    print(f"Saved bin IDs to {file_bin_ids} and orders to {file_orders}.")


def run_ddak(file_path, access_times, hotness, capacity, num_node):
    # Sample execution
    samples = [(i, at) for i, at in enumerate(access_times)]
    k = num_node

    # granularity = 4096
    # SSD_cap = 3.5*1024*1024*1024*1024/granularity
    # CPU1_cap = 0
    # CPU2_cap = 26934617
    # CPU_cap = 26934617 / 2
    # SSD_Hot = 22.32*1024*1024*1024/granularity
    # A_SSD3_0_Hot = 24.18*1024*1024*1024/granularity
    # B_SSD4_0_Hot = 24.18*1024*1024*1024/granularity
    # C_SSD1_0_Hot = 24.18*1024*1024*1024/granularity

    # CPU1_Hot = 141.0508*1024*1024*1024/granularity
    # CPU2_Hot = 196.4505*1024*1024*1024/granularity
    C = capacity #[CPU_cap] * num_cpu + [SSD_cap] * num_ssd + [GPU_cap] * num_gpu
    # C = [CPU_cap, CPU_cap, SSD_cap, SSD_cap, SSD_cap, SSD_cap, SSD_cap, SSD_cap, SSD_cap, SSD_cap]  # 例如：第0和第1个分块容量较小
    # X = [CPU1_Hot, CPU2_Hot, A_SSD3_0_Hot, A_SSD3_0_Hot, A_SSD3_0_Hot, B_SSD4_0_Hot, B_SSD4_0_Hot, B_SSD4_0_Hot, B_SSD4_0_Hot, C_SSD1_0_Hot]
    X = hotness # [CPU1_Hot, CPU2_Hot, SSD_Hot, SSD_Hot, SSD_Hot, SSD_Hot, SSD_Hot, SSD_Hot, SSD_Hot, SSD_Hot]
    # capacities = [30, 20, 25, 15, 10]  # Make sure total is >= len(samples)
    # targets = [2000, 1500, 1800, 1000, 500]

    bins, sample_bin_ids, sample_orders = allocate_balanced_samples_prior(samples, k, C, X)

    # File paths where the arrays will be saved
    file_path_bin_ids = file_path + 'sample_bin_ids'
    file_path_orders = file_path + 'sample_orders'

    # Save the arrays to binary files
    save_arrays_to_binary(file_path_bin_ids, file_path_orders, sample_bin_ids, sample_orders)
