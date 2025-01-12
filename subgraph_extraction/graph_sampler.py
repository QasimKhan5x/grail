import math
from collections import deque
import struct
import logging
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import numpy as np
from collections import deque
from scipy.sparse import coo_matrix
from utils.graph_utils import serialize
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib


def sample_neg(
    adj_list,
    edges,
    num_neg_samples_per_link=1,
    max_size=1000000,
    constrained_neg_prob=0,
):
    """
    Fully vectorized negative sampling for train/test datasets.

    Parameters:
        adj_list (list): List of adjacency matrices (one per relation).
        edges (ndarray): Positive edges [head, tail, relation].
        num_neg_samples_per_link (int): Number of negative samples per positive edge.
        max_size (int): Max number of edges to process.
        constrained_neg_prob (float): Probability of sampling constrained negatives.

    Returns:
        pos_edges, neg_edges: Positive and negative edges.
    """
    pos_edges = edges

    # Step 1: Limit the number of positive edges
    if max_size < len(pos_edges):
        perm = np.random.permutation(len(pos_edges))[:max_size]
        pos_edges = pos_edges[perm]

    n = adj_list[0].shape[0]  # Number of nodes
    r = len(adj_list)  # Number of relations

    # Step 2: Convert the adjacency list to a single block diagonal matrix for efficiency
    adj_block = coo_matrix(([], ([], [])), shape=(n * r, n))  # Initialize empty block
    for rel_idx, adj in enumerate(adj_list):
        row, col, data = adj.tocoo().row, adj.tocoo().col, adj.tocoo().data
        adj_block += coo_matrix((data, (row + rel_idx * n, col)), shape=(n * r, n))

    adj_csr = adj_block.tocsr()  # Convert to CSR for fast slicing

    neg_edges = []
    pbar = tqdm(
        total=num_neg_samples_per_link * len(pos_edges), desc="Negative Sampling"
    )

    # Step 3: Fully vectorized sampling
    while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
        batch_size = min(
            10000, num_neg_samples_per_link * len(pos_edges) - len(neg_edges)
        )
        neg_heads = np.random.choice(n, size=batch_size)
        neg_tails = np.random.choice(n, size=batch_size)
        rels = np.random.choice(r, size=batch_size)

        # Map relation indices to block offsets in the adjacency matrix
        row_offsets = rels * n
        rows = row_offsets + neg_heads

        # Check if the sampled edges are valid (do not exist in the adjacency matrix)
        valid_mask = (neg_heads != neg_tails) & (
            np.asarray(adj_csr[rows, neg_tails]).ravel() == 0
        )

        # Collect valid edges
        valid_neg_edges = np.stack(
            [neg_heads[valid_mask], neg_tails[valid_mask], rels[valid_mask]], axis=1
        )
        neg_edges.extend(valid_neg_edges.tolist())
        pbar.update(len(valid_neg_edges))

    pbar.close()

    neg_edges = np.array(neg_edges)
    return pos_edges, neg_edges


def links2subgraphs(A, graphs, params, max_label_value=None):
    """
    Extract enclosing subgraphs, write in batch mode to LMDB, and handle parallel processing efficiently.

    Parameters:
        A (list): List of adjacency matrices for the graph.
        graphs (dict): Dictionary containing positive and negative links for each split.
        params (Namespace): Configuration parameters.
        max_label_value (int, optional): Maximum node label value.

    Returns:
        None: Writes results to LMDB.
    """
    max_n_label = {"value": np.array([0, 0])}
    subgraph_sizes = []
    enc_ratios = []
    num_pruned_nodes = []

    # Estimate the LMDB map size
    BYTES_PER_DATUM = (
        get_average_subgraph_size(100, list(graphs.values())[0]["pos"], A, params) * 1.5
    )
    links_length = sum(
        (len(split["pos"]) + len(split["neg"])) * 2 for split in graphs.values()
    )
    map_size = math.ceil(links_length * BYTES_PER_DATUM) * 2

    env = lmdb.open(params.db_path, map_size=map_size, max_dbs=6)

    def batch_write_lmdb(txn, data_batch):
        """Batch writes data to LMDB."""
        for key, value in data_batch:
            txn.put(key, serialize(value))

    def extraction_helper(A, links, g_labels, split_env):
        """Helper function to parallelize subgraph extraction and write results with a real-time progress bar."""
        batch_size = 1000
        data_batch = []

        # Write the number of graphs to LMDB before starting extraction
        with env.begin(write=True, db=split_env) as txn:
            txn.put(
                "num_graphs".encode(),
                len(links).to_bytes(int.bit_length(len(links)), byteorder="little"),
            )

        with tqdm_joblib(tqdm(total=len(links), desc="Extracting Subgraphs")):
            # Extract subgraphs in parallel with real-time progress
            results = Parallel(n_jobs=mp.cpu_count(), backend="loky")(
                delayed(extract_save_subgraph)(
                    (idx, link, g_label), A, params, max_label_value
                )
                for idx, (link, g_label) in enumerate(zip(links, g_labels))
            )

        # Batch write results to LMDB
        for str_id, datum in results:
            max_n_label["value"] = np.maximum(
                np.max(datum["n_labels"], axis=0), max_n_label["value"]
            )
            subgraph_sizes.append(datum["subgraph_size"])
            enc_ratios.append(datum["enc_ratio"])
            num_pruned_nodes.append(datum["num_pruned_nodes"])

            data_batch.append((str_id, datum))
            if len(data_batch) >= batch_size:
                with env.begin(write=True, db=split_env) as txn:
                    batch_write_lmdb(txn, data_batch)
                data_batch = []

        # Write any remaining data
        if data_batch:
            with env.begin(write=True, db=split_env) as txn:
                batch_write_lmdb(txn, data_batch)

    # Process each split (train/valid/test)
    for split_name, split in graphs.items():
        logging.info(
            f"Extracting enclosing subgraphs for positive links in {split_name} set"
        )
        labels = np.ones(len(split["pos"]), dtype=np.int8)
        db_name_pos = split_name + "_pos"
        split_env = env.open_db(db_name_pos.encode())
        extraction_helper(A, split["pos"], labels, split_env)

        logging.info(
            f"Extracting enclosing subgraphs for negative links in {split_name} set"
        )
        labels = np.zeros(len(split["neg"]), dtype=np.int8)
        db_name_neg = split_name + "_neg"
        split_env = env.open_db(db_name_neg.encode())
        extraction_helper(A, split["neg"], labels, split_env)

    # Store overall statistics in LMDB
    max_n_label["value"] = (
        max_label_value if max_label_value is not None else max_n_label["value"]
    )
    with env.begin(write=True) as txn:
        bit_len_label_sub = int.bit_length(int(max_n_label["value"][0]))
        bit_len_label_obj = int.bit_length(int(max_n_label["value"][1]))
        txn.put(
            "max_n_label_sub".encode(),
            (int(max_n_label["value"][0])).to_bytes(
                bit_len_label_sub, byteorder="little"
            ),
        )
        txn.put(
            "max_n_label_obj".encode(),
            (int(max_n_label["value"][1])).to_bytes(
                bit_len_label_obj, byteorder="little"
            ),
        )

        txn.put(
            "avg_subgraph_size".encode(),
            struct.pack("f", float(np.mean(subgraph_sizes))),
        )
        txn.put(
            "min_subgraph_size".encode(),
            struct.pack("f", float(np.min(subgraph_sizes))),
        )
        txn.put(
            "max_subgraph_size".encode(),
            struct.pack("f", float(np.max(subgraph_sizes))),
        )
        txn.put(
            "std_subgraph_size".encode(),
            struct.pack("f", float(np.std(subgraph_sizes))),
        )

        txn.put("avg_enc_ratio".encode(), struct.pack("f", float(np.mean(enc_ratios))))
        txn.put("min_enc_ratio".encode(), struct.pack("f", float(np.min(enc_ratios))))
        txn.put("max_enc_ratio".encode(), struct.pack("f", float(np.max(enc_ratios))))
        txn.put("std_enc_ratio".encode(), struct.pack("f", float(np.std(enc_ratios))))

        txn.put(
            "avg_num_pruned_nodes".encode(),
            struct.pack("f", float(np.mean(num_pruned_nodes))),
        )
        txn.put(
            "min_num_pruned_nodes".encode(),
            struct.pack("f", float(np.min(num_pruned_nodes))),
        )
        txn.put(
            "max_num_pruned_nodes".encode(),
            struct.pack("f", float(np.max(num_pruned_nodes))),
        )
        txn.put(
            "std_num_pruned_nodes".encode(),
            struct.pack("f", float(np.std(num_pruned_nodes))),
        )


def get_average_subgraph_size(sample_size, links, A, params):
    total_size = 0
    for n1, n2, r_label in tqdm(links[np.random.choice(len(links), sample_size)]):
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = (
            subgraph_extraction_labeling(
                (n1, n2),
                r_label,
                A,
                params.hop,
                params.enclosing_sub_graph,
                params.max_nodes_per_hop,
            )
        )
        datum = {
            "nodes": nodes,
            "r_label": r_label,
            "g_label": 0,
            "n_labels": n_labels,
            "subgraph_size": subgraph_size,
            "enc_ratio": enc_ratio,
            "num_pruned_nodes": num_pruned_nodes,
        }
        total_size += len(serialize(datum))
    return total_size / sample_size


def intialize_worker(A, params, max_label_value):
    global A_, params_, max_label_value_
    A_, params_, max_label_value_ = A, params, max_label_value


def extract_save_subgraph(args_, A, params, max_label_value):
    """
    Extract and save a subgraph for a given edge.
    """
    idx, (n1, n2, r_label), g_label = args_
    nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = (
        subgraph_extraction_labeling(
            (n1, n2),
            r_label,
            A,
            params.hop,
            params.enclosing_sub_graph,
            params.max_nodes_per_hop,
        )
    )
    # hop = 1
    # subgraph_size = 0
    # nodes, n_labels, enc_ratio, num_pruned_nodes = [], [], 0, 0

    # while subgraph_size < 100 and hop <= params.hop:
    #     nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes = subgraph_extraction_labeling(
    #         (n1, n2), r_label, A, hop, params.enclosing_sub_graph, params.max_nodes_per_hop
    #     )
    #     hop += 1

    # Ensure n_labels is valid
    if n_labels.size == 0:
        n_labels = np.array([[0, 1], [1, 0]])  # Default labels for fallback case
        nodes = [n1, n2]
        subgraph_size = 2
        enc_ratio = 1.0
        num_pruned_nodes = 0

    # Cap node labels if max_label_value is set
    if max_label_value is not None:
        n_labels = np.array(
            [np.minimum(label, max_label_value).tolist() for label in n_labels]
        )

    datum = {
        "nodes": nodes,
        "r_label": r_label,
        "g_label": g_label,
        "n_labels": n_labels,
        "subgraph_size": subgraph_size,
        "enc_ratio": enc_ratio,
        "num_pruned_nodes": num_pruned_nodes,
    }
    str_id = "{:08}".format(idx).encode("ascii")
    return str_id, datum


def subgraph_extraction_labeling(
    ind,
    rel,
    A_list,
    h=1,
    enclosing_sub_graph=False,
    max_nodes_per_hop=None,
    max_node_label_value=None,
):
    """
    Extract the k-hop enclosing subgraph around link 'ind' (u, v), i.e.,
    all nodes that occur on any path of length <= (k+1) between u and v.

    Parameters:
        ind (tuple): Target node pair (u, v).
        rel (int): Relation index (unused here except for placeholders).
        A_list (list): List of adjacency matrices in CSR format.
        h (int): 'k' in the problem statement — the maximum number of hops
                 we allow from u to v is (k+1).
        enclosing_sub_graph (bool): Whether to use intersection (legacy, not used).
        max_nodes_per_hop (int, optional): Not used but kept for compatibility.
        max_node_label_value (int, optional): If set, caps the distance labels.

    Returns:
        tuple: (pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes)
    """

    # ---------------------------------
    # 1) Define a BFS that records distances up to max_hops
    # ---------------------------------
    def bfs_with_distances(start, adj_matrix, max_hops):
        """
        Standard BFS that tracks distances from 'start' up to 'max_hops'.
        Returns a dict: node -> distance from 'start'.
        """
        visited = {}
        queue = deque([(start, 0)])
        while queue:
            node, dist = queue.popleft()
            if node not in visited and dist <= max_hops:
                visited[node] = dist
                # Enqueue neighbors if within max_hops
                for neighbor in adj_matrix[node].indices:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        return visited

    # ---------------------------------
    # 2) Preprocess the graph
    # ---------------------------------
    u, v = ind

    # Combine adjacency matrices
    adj_matrix = sum(A_list)
    adj_matrix = adj_matrix + adj_matrix.T  # ensure undirected

    # Optionally remove self-loops if you want
    # adj_matrix.setdiag(0)
    # adj_matrix.eliminate_zeros()

    # ---------------------------------
    # 3) Get BFS distances from u and from v, up to h
    # ---------------------------------
    distances_u = bfs_with_distances(u, adj_matrix, h)
    distances_v = bfs_with_distances(v, adj_matrix, h)
    # distance between u and v is 1 (0 if both are the same)
    distances_u[v] = int(u != v)
    distances_v[u] = int(u != v)

    # ---------------------------------
    # 4) Determine which nodes lie on a path <= (k+1) edges from u to v
    #    Condition: dist_u(x) + dist_v(x) <= (k+1)
    # ---------------------------------
    # Union of all nodes we encountered in BFS from either side
    union_nodes = set(distances_u.keys()) | set(distances_v.keys())

    # Keep only those that
    # 1. appear on a path of length <= k+1
    # 2. are within h hops from u and v
    subgraph_nodes = []
    for x in union_nodes:
        # if x not visited, treat as infinite
        dist_u_x = distances_u.get(x, 1e9)  
        dist_v_x = distances_v.get(x, 1e9)
        if dist_u_x + dist_v_x <= (h + 1) and dist_u_x <= h and dist_v_x <= h:
            subgraph_nodes.append(x)

    subgraph_nodes = sorted(subgraph_nodes)

    # Count how many we pruned
    num_pruned_nodes = len(union_nodes) - len(subgraph_nodes)

    # ---------------------------------
    # 5) Build label array: (dist_from_u, dist_from_v) for each node
    # ---------------------------------
    labels = np.array(
        [(distances_u[node], distances_v[node]) for node in subgraph_nodes], dtype=int
    )

    # If capping labels is required:
    if max_node_label_value is not None:
        labels = np.minimum(labels, max_node_label_value)

    # ---------------------------------
    # 6) Metrics
    # ---------------------------------
    subgraph_size = len(subgraph_nodes)
    enc_ratio = len(subgraph_nodes) / (len(union_nodes) + 1e-9)

    return subgraph_nodes, labels, subgraph_size, enc_ratio, num_pruned_nodes
