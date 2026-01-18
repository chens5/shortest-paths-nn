# I took this chunk of code from Haoyun Wang's final project 
# from the Topological Data Analysis course from UCSD.

import numpy as np
import torch
import networkx as nx
from ripser import ripser, lower_star_img
from persim import plot_diagrams
import gudhi as gd
from tqdm import tqdm

def random_sampling(graph: nx.Graph, samples_per_source, src=None, **kwargs):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    target = np.random.choice(graph.number_of_nodes(), (samples_per_source, ), replace=False)
    distance = [distance[t.item()] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def distance_based_mesh(graph: nx.Graph, samples_per_source, src=None, pct=1.0, **kwargs):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    _verts_distances = np.array(list(distance.items()))
    vertices = _verts_distances[:, 0].astype(int)
    dist_from_src = (1/_verts_distances[1:, 1])**2
    probs = dist_from_src/np.sum(dist_from_src)

    num_distance_based = int(pct * samples_per_source)
    num_random = int((1.0 - pct)* samples_per_source)
    
    targets_db = np.random.choice(vertices[1:], (num_distance_based, ), p=probs, replace=False)
    targets_random = np.random.choice(vertices[1:], (num_random, ), p=probs, replace=False)

    target = targets_db + targets_random
    distances = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distances

def distance_based(graph: nx.Graph, samples_per_source, src=None):
    if src is None:
        src = np.random.randint(0, graph.number_of_nodes())
    distance = nx.single_source_dijkstra_path_length(graph, src)
    # random
    vertices = np.arange(graph.number_of_nodes())
    row_num = int(np.around(graph.number_of_nodes() ** 0.5))
    hops = abs(src // row_num - vertices // row_num) + abs(src % row_num - vertices % row_num) + 1
    probs = 1 / hops
    probs = probs / probs.sum()
    target = np.random.choice(vertices, (samples_per_source, ), p=probs, replace=False)
    distance = [distance[t] for t in target]
    return [src] * samples_per_source, target.tolist(), distance


def find_critical_points(terrain, threshold):
    # the original terrain has same-height points we must break the tie
    N = terrain.shape[0]
    terrain[:, :, 2] += torch.rand((N, N)) * 1e-5
    lower_dgm = lower_star_img(terrain[:, :, 2])
    upper_dgm = - lower_star_img(- terrain[:, :, 2])
    long_pers_lower_dgm = lower_dgm[lower_dgm[:, 1]- lower_dgm[:, 0] > threshold]
    long_pers_upper_dgm = upper_dgm[upper_dgm[:, 0]- upper_dgm[:, 1] > threshold]
    long_pers_dgm = np.concatenate([long_pers_lower_dgm, long_pers_upper_dgm])
    print(f"{long_pers_dgm.shape[0]} significant critical point pairs at threshhold {threshold}")

    flatten_terrain = terrain.flatten(0, 1)
    critical_idx_0 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 0])) for i in range(long_pers_lower_dgm.shape[0])]
    critical_idx_2 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 0])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = [np.argmin(abs(flatten_terrain[:, 2] - long_pers_lower_dgm[i, 1])) for i in range(long_pers_lower_dgm.shape[0])] + \
                    [np.argmin(abs(flatten_terrain[:, 2] - long_pers_upper_dgm[i, 1])) for i in range(long_pers_upper_dgm.shape[0])]
    critical_idx_1 = list(set(critical_idx_1))

    critical_idx = torch.stack(critical_idx_0 + critical_idx_1 + critical_idx_2)
    # shuffle it
    critical_idx = critical_idx[torch.randperm(critical_idx.shape[0])]
    critical_idx = [src.item() for src in critical_idx]
    return critical_idx

def mesh_lower_star_filtration(edges, node_idx_dict, threshhold):
    print("creating simplex tree")
    st = gd.SimplexTree()
    # build simplex tree
    print("Adding verts to simplex tree")
    for node in node_idx_dict:
        f_val = node_idx_dict[node][2]
        st.insert([node], filtration=f_val)
    print("Adding edges to simplex tree")
    for edge in tqdm(edges):
        v1 = edge[0]
        v2 = edge[1]
        if v1 == v2:
            continue
        f1 = node_idx_dict[v1][2]
        f2 = node_idx_dict[v2][2]
        f_edge = max(f1, f2)
        # print(type(edge), type(edge[0]), type(edge[1]))
        # to_add = 
        st.insert(edge, filtration=f_edge)
    print("computing persistence.....")
    make_non_decreasing = st.make_filtration_non_decreasing()
    print("Make non decreasing:",make_non_decreasing)
    st.persistence()
    high_persistence_points = []
    for (birth, death) in tqdm(st.persistence_pairs()):
        birth_simplex = birth[0]
        if len(death) == 0:
            high_persistence_points.append(birth_simplex)
            continue
        death_simplex = death[0] if node_idx_dict[death[0]][2] > node_idx_dict[death[1]][2] else death[1]
        persistence = abs(node_idx_dict[death_simplex][2] - node_idx_dict[birth_simplex][2])
        if persistence > threshhold:
            high_persistence_points.append(birth_simplex)
            high_persistence_points.append(death_simplex)
    print("Number of high persistence pairs:", len(high_persistence_points)//2)
    return high_persistence_points, st

def reshape_node_features_grid(node_features, rows, cols):
    c1 = node_features[:, 0].reshape(rows, cols)
    c2 = node_features[:, 1].reshape(rows, cols)
    c3 = nodeo_features[:, 2].reshape(rows, cols)
    terrain = torch.tensor(np.stack([c1, c2, c3]), dtype=torch.float)
    terrain = np.transpose(terrain, (1, 2, 0))
    return terrain