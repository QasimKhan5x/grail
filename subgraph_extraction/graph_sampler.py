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
    # Merge adjacency matrices and make undirected
    A = sum(A)
    A = A + A.T

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
    full_adj,
    h=1,
    enclosing_sub_graph=False,
    max_nodes_per_hop=None,
    max_node_label_value=None,
):
    """
    Extract the k-hop enclosing subgraph around link 'ind' (u, v),
    then label the nodes with Double Radius Node Labeling (DRNL).

    Parameters:
        ind (tuple): Target node pair (u, v).
        rel (int): Relation index (not used here, kept for compatibility).
        A_list (list): List of adjacency matrices in CSR format.
        h (int): 'k' in the problem statement — the maximum number of hops
                 we allow from u to v is (k+1).
        enclosing_sub_graph (bool): (Unused in BFS method; can remain for API).
        max_nodes_per_hop (int): (Unused here, but kept for signature).
        max_node_label_value (int): If set, caps the distance labels.

    Returns:
        tuple:
            pruned_subgraph_nodes (list[int]): Node IDs in the final subgraph,
                                               in sorted order.
            drnl_labels (np.ndarray): Array of shape (num_nodes, 2),
                                      the DRNL labels (distances ignoring the other root).
            subgraph_size (int): Number of nodes in final subgraph.
            enc_ratio (float): Ratio of final subgraph size to BFS-union size (for logging).
            num_pruned_nodes (int): How many nodes got pruned out from BFS union.
    """
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
                for neighbor in adj_matrix[node].indices:
                    if neighbor not in visited:
                        queue.append((neighbor, dist + 1))
        return visited

    # 1) Get the source and target nodes
    u, v = ind

    # 2) Get BFS distances up to h from u and from v
    distances_u = bfs_with_distances(u, full_adj, h)
    distances_v = bfs_with_distances(v, full_adj, h)

    # 3) Union of BFS nodes
    union_nodes = set(distances_u.keys()) | set(distances_v.keys())

    # Keep only nodes x where dist_u(x) + dist_v(x) <= (h+1)
    # and each distance <= h
    subgraph_nodes = []
    for x in union_nodes:
        du = distances_u.get(x, 1e9)
        dv = distances_v.get(x, 1e9)
        if du <= h and dv <= h and (du + dv) <= (h + 1):
            subgraph_nodes.append(x)

    subgraph_nodes = sorted(subgraph_nodes)
    num_pruned_nodes = len(union_nodes) - len(subgraph_nodes)

    # 4) Build sub-adjacency for these subgraph_nodes
    #    We'll reindex them from 0..len(subgraph_nodes)-1
    idx_map = {node_id: i for i, node_id in enumerate(subgraph_nodes)}
    sub_n = len(subgraph_nodes)
    # Extract the adjacency sub-matrix in CSR form
    sub_adj = full_adj[subgraph_nodes, :][:, subgraph_nodes].tocsr()

    # 5) DRNL labeling:
    #    We'll remove v from adjacency, BFS from u in that 'reduced' graph,
    #    then remove u, BFS from v.
    #    dist_u_ignore_v[i] = BFS distance from subgraph index of u to i (with v removed).
    #    dist_v_ignore_u[i] = BFS distance from subgraph index of v to i (with u removed).

    # -----------------------------------------------------------
    #  If u or v not in subgraph, force a minimal subgraph of {u, v}
    # -----------------------------------------------------------
    if u not in idx_map or v not in idx_map:
        # Instead of returning an empty subgraph, we create a 2-node subgraph:
        pruned_subgraph_nodes = [u, v]  # just these two
        # DRNL labels: by convention, (0,1) for u, (1,0) for v
        drnl_labels = np.array([[0, 1], [1, 0]], dtype=int)

        subgraph_size = 2
        enc_ratio = 1.0
        num_pruned_nodes_total = 0

        return pruned_subgraph_nodes, drnl_labels, subgraph_size, enc_ratio, num_pruned_nodes_total


    u_idx = idx_map[u]
    v_idx = idx_map[v]

    def bfs_distances_ignoring(adj_csr, start_idx, ignore_idx):
        """
        Return array of BFS distances from start_idx in adjacency 'adj_csr',
        after removing 'ignore_idx' (no edges from/to that node).
        If unreachable, distance = large sentinel (999999).
        """
        n = adj_csr.shape[0]
        visited = np.full(n, False, dtype=bool)
        dist = np.full(n, 999999, dtype=int)

        # We effectively "remove" ignore_idx by skipping edges from/to it.
        # We'll just treat it as visited or no edges from it.
        # Easiest hack: treat ignore_idx as visited from the start.
        visited[ignore_idx] = True  # so we never traverse from or to 'ignore_idx'

        # BFS
        from collections import deque
        queue = deque([start_idx])
        dist[start_idx] = 0
        visited[start_idx] = True

        while queue:
            curr = queue.popleft()
            curr_dist = dist[curr]

            # look at neighbors
            row_start = adj_csr.indptr[curr]
            row_end = adj_csr.indptr[curr+1]
            neighbors = adj_csr.indices[row_start:row_end]

            for nbr in neighbors:
                if not visited[nbr]:
                    visited[nbr] = True
                    dist[nbr] = curr_dist + 1
                    queue.append(nbr)
        return dist

    # Distances from u ignoring v:
    dist_u_ignore_v = bfs_distances_ignoring(sub_adj, u_idx, v_idx)
    # Distances from v ignoring u:
    dist_v_ignore_u = bfs_distances_ignoring(sub_adj, v_idx, u_idx)

    # Combine into a 2D label array:
    drnl_labels = np.vstack([dist_u_ignore_v, dist_v_ignore_u]).T  # shape (num_nodes, 2)

    # Special label for the roots:
    # By many DRNL conventions, we set label(u) = (0,1), label(v) = (1,0)
    drnl_labels[u_idx] = [0, 1]
    drnl_labels[v_idx] = [1, 0]

    # 6) Optional filter: keep only nodes with max distance <= h
    #    (This enforces the "enclosing subgraph" of radius h in DRNL sense)
    keep_mask = (drnl_labels.max(axis=1) <= h)
    # If you want this "DRNL radius" filter, do:
    pruned_indices = np.where(keep_mask)[0]
    # Else, if you prefer to keep them all, just use:
    # pruned_indices = np.arange(sub_n)

    # 7) Apply the filter
    pruned_subgraph_nodes = [subgraph_nodes[i] for i in pruned_indices]
    drnl_labels = drnl_labels[pruned_indices]

    # 8) Cap labels if max_node_label_value is set
    if max_node_label_value is not None:
        drnl_labels = np.minimum(drnl_labels, max_node_label_value)

    # 9) Final metrics
    subgraph_size = len(pruned_subgraph_nodes)
    enc_ratio = subgraph_size / (len(subgraph_nodes) + 1e-9)
    num_pruned_after_label = len(subgraph_nodes) - subgraph_size
    num_pruned_nodes_total = num_pruned_nodes + num_pruned_after_label

    return (
        pruned_subgraph_nodes,
        drnl_labels,
        subgraph_size,
        enc_ratio,
        num_pruned_nodes_total
    )
