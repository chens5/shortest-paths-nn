import numpy as np
import torch, queue
from torch_geometric.data import Data
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import multiprocessing as mp
import time
import itertools

import argparse
import os
from point_sampler import * 
from dataset import * 
import csv

def process_node_features_np(node_data):
    node_idx_dict = {}
    x = node_data['x']/1000
    y = node_data['y']/1000
    z = node_data['z']/1000
    idx = node_data['vertex_ids']
    features = np.vstack((x, y, z)).T

    for i in trange(len(x)):
        node_idx_dict[idx[i]] = features[i]
    print(max(idx))
    return node_idx_dict, features

def process_node_features(filename):

    node_idx_dict = {}
    node_feature_lst = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            idx = int(row[0])
            x = float(row[1])/1000
            y = float(row[2])/1000
            z = float(row[3])/1000
            node_idx_dict[idx] = [x, y, z]
            node_feature_lst.append([x, y, z])
            counter += 1
    return node_idx_dict, node_feature_lst

def process_edges(filename):
    edges = []
    counter = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if counter == 0:
                counter += 1
                continue
            edges.append([int(row[0]), int(row[1])])
    return edges

def construct_nx_graph(edges, node_mappings, scale=False):
    G = nx.Graph()
    for edge in tqdm(edges):
        idx1 = int(edge[0])
        idx2 = int(edge[1])
        p1 = np.array(node_mappings[idx1])
        p2 = np.array(node_mappings[idx2])
        if scale:
            slope = (abs(p1[2] - p2[2]))/(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
            angle_of_elevation = np.abs(np.arctan(p1[2] - p2[2])/np.linalg.norm(p2[:2] - p1[:2], ord=2))
            val = angle_of_elevation
            w = (1 + val) * np.linalg.norm(p1 - p2, ord=2)
        else: 
            w = np.linalg.norm(p1 - p2, ord=2)
        G.add_edge(idx1, idx2, weight=w)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())
    return G


def construct_dataset(G, 
                      node_features, 
                      filename, 
                      num_srcs, 
                      samples_per_source,
                      edges,
                      node_idx_dict,
                      sources = [],
                      sampling_method='random-sampling',
                      predefined_src_tar_sampling = 'random_sampling',  
                      pct=1.0, 
                      threshhold=0.001,
                      mixture=0.5):
    lst_of_edges = edges
    edges, distances = to_pyg_graph(G)
    print("Sampling techinuqe:", sampling_method)
    if sampling_method == 'random-sampling':
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        src_sampling_fn_pairs = [(src_nodes, random_sampling)]
    elif sampling_method == 'distance-based': # random sources, distance based
        src_nodes = np.random.choice(len(node_features), size=num_srcs)
        src_sampling_fn_pairs = [(src_nodes, distance_based_mesh)]
    elif sampling_method == 'from-prev-sources':
        num_predefined = int(num_srcs*mixture)
        num_random = num_srcs - num_predefined
        random_srcs =  np.random.choice(len(node_features), size=num_random, replace=False).astype(int)
        predefined_srcs = np.random.choice(sources, size=num_predefined, replace=False).astype(int)
        fn = globals()[predfined_src_tar_sampling]
        src_sampling_fn_pairs = [(random_srcs, random_sampling), (predefined_srcs, fn)]
    elif sampling_method == 'critical-point-db':
        critical_points, _ = mesh_lower_star_filtration(lst_of_edges, node_idx_dict, threshhold=threshhold)
        num_cp = int(num_srcs*mixture)
        if num_cp > len(critical_points):
            num_random = num_srcs - len(critical_points)
            cp_srcs = critical_points
        else:
            cp_srcs = np.random.choice(critical_points, size=num_cp, replace=False).astype(int)
            num_random = num_srcs - num_cp
        random_srcs =  np.random.choice(len(node_features), size=num_random, replace=False).astype(int)
        print(len(random_srcs), len(cp_srcs))
        src_nodes = np.hstack((random_srcs, cp_srcs))
        src_sampling_fn_pairs = [(src_nodes, distance_based)]
    else:
        raise NotImplementedError("Sampling technique not implemented yet")
    srcs = []
    tars = []
    lengths = []
    print("Number of source nodes:", num_srcs)
    print("Generating shortest path distances.....")
    for pair in src_sampling_fn_pairs:
        src_nodes = pair[0]
        sampling_fn = pair[1]
        for src in tqdm(src_nodes):
            source, target, length = sampling_fn(G, samples_per_source, src=src, pct=pct)
            srcs += source
            tars += target
            lengths += length

    print("Number of lengths in dataset:", len(lengths))
    print("Saved dataset in:", filename)
    np.savez(filename, 
         edge_index = edges, 
         distances=distances, 
         srcs = srcs,
         tars = tars,
         lengths = lengths,
         node_features=node_features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    parser.add_argument('--edge-input-data', type=str)
    parser.add_argument('--node-feature-data', type=str)
    parser.add_argument('--num-sources', type=int)
    parser.add_argument('--dataset-size', type=int)
    parser.add_argument('--sampling-method', type=str, default='random-sampling')
    parser.add_argument('--pct-db-tar', type=float, default=0.30)
    parser.add_argument('--source-node-file', type=str, default=None)
    parser.add_argument('--source-mixture', type=float, default=0.50)
    parser.add_argument('--predefined_src_tar_sampling', type=str, default='random_sampling')

    args = parser.parse_args()
    if '.csv' in args.edge_input_data and '.csv' in args.node_feature_data:
        node_idx_dict, node_features = process_node_features(args.node_feature_data)
        edges = process_edges(args.edge_input_data)
    else:
        edge_data = np.load(args.edge_input_data)
        edges = edge_data['edges']
        print("Number of edges:",len(edges))
        edges = np.unique(edges, axis=0)
        print("Number of unique edges:",len(edges))
        node_data = np.load(args.node_feature_data)
        node_idx_dict, node_features = process_node_features_np(node_data)
    G = construct_nx_graph(edges, node_idx_dict)
    print("Source sampling method:", args.sampling_method, "target sampling method:", args.predefined_src_tar_sampling)
    if args.source_node_file == None:
        sources = []
    else:
        sources = np.load(args.source_node_file)
    construct_dataset(G=G, 
                    node_features=node_features, 
                    filename=filename,
                    num_srcs=args.num_sources, 
                    samples_per_source = args.dataset_size//args.num_sources,
                    sampling_method=args.sampling_method,
                    edges=edges,
                    node_idx_dict = node_idx_dict, 
                    mixture = args.source_mixture,
                    predfined_src_tar_sampling=args.predefined_src_tar_sampling,
                    pct = args.pct_db_tar, 
                    sources= sources)

    return 

if __name__ == '__main__':
    main()