import os 
import argparse 
import subprocess
import re
import networkx as nx
import math
import sys

def Run(args):

    if args.dataset_name == "products":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100m":
        path = args.dataset_path + "/paper100M/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 11105995
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "com-friendster":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "ukunion":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "uk2014":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "igb":
        path = args.dataset_path + "/igb/"
        vertices_num = 269346175
        edges_num = 3870892894
        features_dim = 256
        train_set_num = 2693461
        valid_set_num = 165
        test_set_num = 218
    else:
        print("invalid dataset path")
        exit
    

    with open("meta_config","w") as file:
        file.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} ".format(path, args.train_batch_size, vertices_num, edges_num, \
                                                features_dim, train_set_num, valid_set_num, test_set_num, \
                                            0, args.epoch, 0, args.ssd_number, args.num_queues_per_ssd, 100, 100))
    
    
    os.system("./sampling_server/build/bin/server {} {}".format(args.gpu_number, 1))

if __name__ == "__main__":

    argparser = argparse.ArgumentParser("Server.")
    argparser.add_argument('--dataset_path', type=str, default="/share/gnn_data/igb260m/IGB-Datasets/data")
    argparser.add_argument('--dataset_name', type=str, default="igb")
    argparser.add_argument('--train_batch_size', type=int, default=8000)
    argparser.add_argument('--fanout', type=list, default=[25, 10])
    argparser.add_argument('--gpu_number', type=int, default=2)
    argparser.add_argument('--epoch', type=int, default=2)
    argparser.add_argument('--ssd_number', type=int, default=2)
    argparser.add_argument('--num_queues_per_ssd', type=int, default=128)

    args = argparser.parse_args()

    Run(args)
