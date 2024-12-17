import math
from collections import deque
import struct
import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
from collections import deque
from scipy.special import softmax
from utils.graph_utils import  serialize, get_edge_count


def sample_neg(adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
    pos_edges = edges
    neg_edges = []

    # if max_size is set, randomly sample train links
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    # sample negative links for train/test
    n, r = adj_list[0].shape[0], len(adj_list)

    # distribution of edges across reelations
    theta = 0.001
    edge_count = get_edge_count(adj_list)
    rel_dist = np.zeros(edge_count.shape)
    idx = np.nonzero(edge_count)
    rel_dist[idx] = softmax(theta * edge_count[idx])

    # possible head and tails for each relation
    valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
    valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

    pbar = tqdm(total=len(pos_edges))
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], pos_edges[pbar.n % len(pos_edges)][2]
        if np.random.uniform() < constrained_neg_prob:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(valid_heads[rel])
            else:
                neg_tail = np.random.choice(valid_tails[rel])
        else:
            if np.random.uniform() < 0.5:
                neg_head = np.random.choice(n)
            else:
                neg_tail = np.random.choice(n)

        if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
            neg_edges.append([neg_head, neg_tail, rel])
            pbar.update(1)

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, max_label_value=None):
    '''
    extract enclosing subgraphs, write map mode + named dbs
    '''
    max_n_label = {'value': np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    BYTES_PER_DATUM = get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
    links_length = 0
    for split_name, split in graphs.items():
        links_length += (len(split['pos']) + len(split['neg'])) * 2
        
    map_size = math.ceil(links_length * BYTES_PER_DATUM) * 2
    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def extraction_helper(A, links, g_labels, split_env):

        with env.begin(write=True, db=split_env) as txn:
            txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

        with mp.Pool(processes=None, initializer=intialize_worker, initargs=(A, params, max_label_value)) as p:
            args_ = zip(range(len(links)), links, g_labels)
            for (str_id, datum) in tqdm(p.imap(extract_save_subgraph, args_), total=len(links)):
                max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                subgraph_sizes.append(datum['subgraph_size'])
                enc_ratios.append(datum['enc_ratio'])
                num_pruned_nodes.append(datum['num_pruned_nodes'])

                with env.begin(write=True, db=split_env) as txn:
                    txn.put(str_id, serialize(datum))

    for split_name, split in graphs.items():
        logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
        labels = np.ones(len(split['pos']))
        db_name_pos = split_name + '_pos'
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split['pos'], labels, split_env)

        logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
        labels = np.zeros(len(split['neg']))
        db_name_neg = split_name + '_neg'
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split['neg'], labels, split_env)

    max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
        txn.put('max_n_label_sub'.encode(), (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
        txn.put('max_n_label_obj'.encode(), (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

        txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
        txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
        txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
        txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

        txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
        txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
        txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
        txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

        txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
        txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
        txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
        txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for (n1, n2, r_label) in tqdm(links[np.random.choice(len(links), sample_size)]):
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling((n1, n2), r_label, A, params.hop, params.enclosing_sub_graph, params.max_nodes_per_hop)
        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_subgraph(args_):
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling(
        (n1, n2), r_label, A_, params_.hop, params_.enclosing_sub_graph, params_.max_nodes_per_hop
    )

    # Ensure n_labels is valid
    if n_labels.size == 0:
        n_labels = np.array([[0, 1], [1, 0]])  # Default labels for fallback case
        nodes = [n1, n2]
        subgraph_size = 2
        enc_ratio = 1.0
        num_pruned_nodes = 0

    # Cap node labels if max_label_value_ is set
    if max_label_value_ is not None:
        n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])
    datum = {
        'nodes': nodes,
        'r_label': r_label,
        'g_label': g_label,
        'n_labels': n_labels,
        'subgraph_size': subgraph_size,
        'enc_ratio': enc_ratio,
        'num_pruned_nodes': num_pruned_nodes
    }
    # from pprint import pprint
    # print(idx)
    # pprint(datum['n_labels'])
    str_id = '{:08}'.format(idx).encode('ascii')

    return (str_id, datum)


def subgraph_extraction_labeling(ind, rel, A_list, h=1, enclosing_sub_graph=False,
                                 max_nodes_per_hop=None, max_node_label_value=None):
    """
    Extract the h-hop enclosing subgraph around link 'ind' and filter unreachable nodes.

    Parameters:
        ind (tuple): Target node pair (u, v).
        rel (int): Relation index.
        A_list (list): List of adjacency matrices in CSR format.
        h (int): Maximum number of hops.
        enclosing_sub_graph (bool): Whether to use intersection (True).
        max_nodes_per_hop (int, optional): Not used but kept for compatibility.
        max_node_label_value (int, optional): Maximum node label value.

    Returns:
        tuple: (pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes)
    """
    def bfs_with_labels(start_node, adj_matrix, max_hops):
        """Perform BFS from a start node to collect neighbors and label distances."""
        visited = {}
        queue = deque([(start_node, 0)])

        while queue:
            node, dist = queue.popleft()
            if node in visited or dist > max_hops:
                continue
            visited[node] = dist

            for neighbor in adj_matrix[node].indices:
                if neighbor not in visited:
                    queue.append((neighbor, dist + 1))

        return visited

    u, v = ind
    adj_matrix = sum(A_list)  # Combine adjacency matrices into a single undirected graph

    # Step 1: BFS neighbors and distances from u and v
    distances_u = bfs_with_labels(u, adj_matrix, h)
    distances_v = bfs_with_labels(v, adj_matrix, h)

    # Add v to u's distances and vice versa
    distances_u[v] = 1
    distances_v[u] = 1

    # Step 2: Intersection of reachable nodes
    reachable_nodes = set(distances_u.keys()) & set(distances_v.keys())
    num_pruned_nodes = len(set(distances_u.keys()) | set(distances_v.keys())) - len(reachable_nodes)

    # Step 3: Build subgraph nodes
    subgraph_nodes = sorted(reachable_nodes)

    # Step 4: Assign labels explicitly for u and v
    labels = np.array([[distances_u[node], distances_v[node]] for node in subgraph_nodes], dtype=int)

    # Cap labels if max_node_label_value is specified
    if max_node_label_value is not None:
        labels = np.minimum(labels, max_node_label_value)

    # Step 5: Metrics
    subgraph_size = len(subgraph_nodes)
    enc_ratio = len(reachable_nodes) / (len(set(distances_u.keys()) | set(distances_v.keys())) + 1e-3)

    return subgraph_nodes, labels, subgraph_size, enc_ratio, num_pruned_nodes
