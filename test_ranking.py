import os
import argparse
import logging
import json
import time

from scipy.sparse import issparse
import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import networkx as nx
import torch
import numpy as np
import dgl

from subgraph_extraction.graph_sampler import subgraph_extraction_labeling


def ssp_multigraph_to_dgl(graph, n_feats=None):
    """
    Converting ssp multigraph (i.e. list of adjs) to dgl multigraph.
    """

    g_nx = nx.MultiDiGraph()
    g_nx.add_nodes_from(list(range(graph[0].shape[0])))
    # Add edges
    for rel, adj in enumerate(graph):
        # Convert adjacency matrix to tuples for nx0
        nx_triplets = []
        for src, dst in list(zip(adj.tocoo().row, adj.tocoo().col)):
            nx_triplets.append((src, dst, {"type": rel}))
        g_nx.add_edges_from(nx_triplets)

    # make dgl graph
    g_dgl = dgl.from_networkx(g_nx, edge_attrs=["type"])
    # add node features
    if n_feats is not None:
        g_dgl.ndata["feat"] = torch.tensor(n_feats)

    return g_dgl


def process_files(files, saved_relation2id, add_traspose_rels):
    """
    files: Dictionary map of file paths to read the triplets from.
    saved_relation2id: Saved relation2id (mostly passed from a trained model) which can be used to map relations to pre-defined indices and filter out the unknown ones.
    """
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}

    ent = 0
    rel = 0

    for file_type, file_path in files.items():

        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n")[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Save the triplets corresponding to only the known relations
            if triplet[1] in saved_relation2id:
                data.append(
                    [
                        entity2id[triplet[0]],
                        entity2id[triplet[2]],
                        saved_relation2id[triplet[1]],
                    ]
                )

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct the list of adjacency matrix each corresponding to eeach relation. Note that this is constructed only from the train data.
    adj_list = []
    for i in range(len(saved_relation2id)):
        idx = np.argwhere(triplets["graph"][:, 2] == i)
        adj_list.append(
            ssp.csc_matrix(
                (
                    np.ones(len(idx), dtype=np.uint8),
                    (
                        triplets["graph"][:, 0][idx].squeeze(1),
                        triplets["graph"][:, 1][idx].squeeze(1),
                    ),
                ),
                shape=(len(entity2id), len(entity2id)),
            )
        )

    # Add transpose matrices to handle both directions of relations.
    adj_list_aug = adj_list
    if add_traspose_rels:
        adj_list_t = [adj.T for adj in adj_list]
        adj_list_aug = adj_list + adj_list_t

    dgl_adj_list = ssp_multigraph_to_dgl(adj_list_aug)

    return (
        adj_list,
        dgl_adj_list,
        triplets,
        entity2id,
        relation2id,
        id2entity,
        id2relation,
    )


def intialize_worker(
    model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id
):
    global model_, adj_list_, dgl_adj_list_, id2entity_, params_, node_features_, kge_entity2id_
    (
        model_,
        adj_list_,
        dgl_adj_list_,
        id2entity_,
        params_,
        node_features_,
        kge_entity2id_,
    ) = (model, adj_list, dgl_adj_list, id2entity, params, node_features, kge_entity2id)


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):
    """
    Generate negative samples by replacing heads and tails in test_links.

    Args:
        test_links (np.ndarray): Array of shape (num_links, 3) containing test triplets (head, tail, rel).
        adj_list (list of np.ndarray or list of scipy.sparse matrices):
            List where each element is an adjacency matrix for a relation.
        num_samples (int): Number of negative samples to generate for each test link.

    Returns:
        list of dict: Each dictionary contains 'head' and 'tail' keys with corresponding negative samples.
    """
    # Validate and prepare adjacency matrices
    processed_adj_list = []
    for idx, rel_adj in enumerate(adj_list):
        if isinstance(rel_adj, np.ndarray):
            if rel_adj.ndim != 2:
                raise ValueError(
                    f"Adjacency matrix for relation {idx} is not 2-dimensional."
                )
            processed_adj_list.append(rel_adj)
        else:
            # Attempt to convert sparse matrices to dense
            try:
                dense_adj = rel_adj.toarray()
                if dense_adj.ndim != 2:
                    raise ValueError(
                        f"Adjacency matrix for relation {idx} could not be converted to 2D."
                    )
                processed_adj_list.append(dense_adj)
            except AttributeError:
                raise TypeError(
                    f"Adjacency matrix for relation {idx} is neither a NumPy array nor a sparse matrix."
                )

    # Stack adjacency matrices into a 3D NumPy array
    try:
        adj_stack = np.stack(processed_adj_list)  # Shape: (r, n, n)
    except ValueError as e:
        raise ValueError(f"Failed to stack adjacency matrices: {e}")

    # Verify the shape of adj_stack
    if adj_stack.ndim != 3:
        raise ValueError(
            f"Expected adj_stack to be 3-dimensional, but got shape {adj_stack.shape}"
        )

    r, n, n_check = adj_stack.shape
    if n != n_check:
        raise ValueError(
            f"Adjacency matrices must be square. Found shape {adj_stack.shape}"
        )

    # Extract heads, tails, and relations from test_links
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    num_links = len(test_links)

    neg_triplets = []

    for i in range(num_links):
        head, tail, rel = heads[i], tails[i], rels[i]

        # Validate relation index
        if rel < 0 or rel >= r:
            raise IndexError(
                f"Relation index {rel} out of bounds for adj_stack with {r} relations."
            )

        # Initialize negative samples with the original triplet
        head_neg = [[head, tail, rel]]
        tail_neg = [[head, tail, rel]]

        # Number of samples needed excluding the original triplet
        needed_head = num_samples - 1
        needed_tail = num_samples - 1

        # Generate negative samples by replacing the tail
        while needed_head > 0:
            # Sample in batches to reduce the number of iterations
            batch_size = max(needed_head * 2, 100)
            sampled_tails = np.random.randint(0, n, size=batch_size)

            # Apply conditions: neg_tail != head and adj_list[rel][head, neg_tail] == 0
            condition_tail = sampled_tails != head
            condition_no_edge_tail = adj_stack[rel, head, sampled_tails] == 0
            valid_mask = condition_tail & condition_no_edge_tail
            valid_tails = sampled_tails[valid_mask]

            # Append valid samples
            for neg_tail_val in valid_tails:
                head_neg.append([head, neg_tail_val, rel])
                needed_head -= 1
                if needed_head == 0:
                    break

            # Safety check to prevent infinite loops
            if batch_size > n * 2:
                break  # Assuming insufficient valid samples

        # Generate negative samples by replacing the head
        while needed_tail > 0:
            # Sample in batches to reduce the number of iterations
            batch_size = max(needed_tail * 2, 100)
            sampled_heads = np.random.randint(0, n, size=batch_size)

            # Apply conditions: neg_head != tail and adj_list[rel][neg_head, tail] == 0
            condition_head = sampled_heads != tail
            condition_no_edge_head = adj_stack[rel, sampled_heads, tail] == 0
            valid_mask = condition_head & condition_no_edge_head
            valid_heads = sampled_heads[valid_mask]

            # Append valid samples
            for neg_head_val in valid_heads:
                tail_neg.append([neg_head_val, tail, rel])
                needed_tail -= 1
                if needed_tail == 0:
                    break

            # Safety check to prevent infinite loops
            if batch_size > n * 2:
                break  # Assuming insufficient valid samples

        # Convert lists to NumPy arrays
        head_neg_array = np.array(head_neg)
        tail_neg_array = np.array(tail_neg)

        # Append to the result list
        neg_triplets.append({"head": [head_neg_array, 0], "tail": [tail_neg_array, 0]})

    return neg_triplets


def get_neg_samples_replacing_head_tail_all(test_links, adj_list):
    """
    Generate all possible negative samples by replacing heads and tails in test_links.

    Args:
        test_links (np.ndarray): Array of shape (num_links, 3) containing test triplets (head, tail, rel).
        adj_list (list of np.ndarray or list of scipy.sparse matrices):
            List where each element is an adjacency matrix for a relation.

    Returns:
        list of dict: Each dictionary contains 'head' and 'tail' keys with corresponding negative samples.
                      The structure is {'head': [array_of_neg_samples, 0], 'tail': [array_of_neg_samples, 0]}.
    """
    # Validate and prepare adjacency matrices
    processed_adj_list = []
    for idx, rel_adj in enumerate(adj_list):
        if isinstance(rel_adj, np.ndarray):
            if rel_adj.ndim != 2:
                raise ValueError(
                    f"Adjacency matrix for relation {idx} is not 2-dimensional."
                )
            processed_adj_list.append(rel_adj)
        elif issparse(rel_adj):
            # Convert sparse matrices to dense
            dense_adj = rel_adj.toarray()
            if dense_adj.ndim != 2:
                raise ValueError(
                    f"Adjacency matrix for relation {idx} could not be converted to 2D."
                )
            processed_adj_list.append(dense_adj)
        else:
            raise TypeError(
                f"Adjacency matrix for relation {idx} is neither a NumPy array nor a sparse matrix."
            )

    # Stack adjacency matrices into a 3D NumPy array
    try:
        adj_stack = np.stack(processed_adj_list)  # Shape: (r, n, n)
    except ValueError as e:
        raise ValueError(f"Failed to stack adjacency matrices: {e}")

    # Verify the shape of adj_stack
    if adj_stack.ndim != 3:
        raise ValueError(
            f"Expected adj_stack to be 3-dimensional, but got shape {adj_stack.shape}"
        )

    r, n, n_check = adj_stack.shape
    if n != n_check:
        raise ValueError(
            f"Adjacency matrices must be square. Found shape {adj_stack.shape}"
        )

    # Extract heads, tails, and relations from test_links
    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    num_links = len(test_links)

    neg_triplets = []

    for i in tqdm(range(num_links), total=num_links, desc="Sampling negative triplets"):
        head, tail, rel = heads[i], tails[i], rels[i]

        # Validate relation index
        if rel < 0 or rel >= r:
            raise IndexError(
                f"Relation index {rel} out of bounds for adj_stack with {r} relations."
            )

        # Get the adjacency matrix for the current relation
        rel_adj = adj_stack[rel]  # Shape: (n, n)

        # **Generating Negative Samples by Replacing Tails**
        # Condition 1: neg_tail != head
        # Condition 2: (head, neg_tail, rel) does not exist in adj_list[rel]
        # Equivalent to: rel_adj[head, neg_tail] == 0 and neg_tail != head

        # Create a boolean mask for valid neg_tails
        mask_neg_tails = (rel_adj[head] == 0) & (np.arange(n) != head)
        valid_neg_tails = np.where(mask_neg_tails)[0]

        # **Generating Negative Samples by Replacing Heads**
        # Condition 1: neg_head != tail
        # Condition 2: (neg_head, tail, rel) does not exist in adj_list[rel}
        # Equivalent to: rel_adj[neg_head, tail] == 0 and neg_head != tail

        # Create a boolean mask for valid neg_heads
        mask_neg_heads = (rel_adj[:, tail] == 0) & (np.arange(n) != tail)
        valid_neg_heads = np.where(mask_neg_heads)[0]

        # **Constructing the Negative Triplet Dictionary**
        # Initialize with the original triplet
        head_neg = [[head, tail, rel]]
        tail_neg = [[head, tail, rel]]

        # Append all valid negative tail replacements
        if valid_neg_tails.size > 0:
            head_neg.extend([[head, neg_tail, rel] for neg_tail in valid_neg_tails])

        # Append all valid negative head replacements
        if valid_neg_heads.size > 0:
            tail_neg.extend([[neg_head, tail, rel] for neg_head in valid_neg_heads])

        # Convert lists to NumPy arrays
        head_neg_array = np.array(head_neg)
        tail_neg_array = np.array(tail_neg)

        # Append to the result list
        neg_triplets.append({"head": [head_neg_array, 0], "tail": [tail_neg_array, 0]})

    return neg_triplets


def get_neg_samples_replacing_head_tail_from_ruleN(
    ruleN_pred_path, entity2id, saved_relation2id
):
    with open(ruleN_pred_path) as f:
        pred_data = [line.split() for line in f.read().split("\n")[:-1]]

    neg_triplets = []
    for i in range(len(pred_data) // 3):
        neg_triplet = {"head": [[], 10000], "tail": [[], 10000]}
        if pred_data[3 * i][1] in saved_relation2id:
            head, rel, tail = (
                entity2id[pred_data[3 * i][0]],
                saved_relation2id[pred_data[3 * i][1]],
                entity2id[pred_data[3 * i][2]],
            )
            for j, new_head in enumerate(pred_data[3 * i + 1][1::2]):
                neg_triplet["head"][0].append([entity2id[new_head], tail, rel])
                if entity2id[new_head] == head:
                    neg_triplet["head"][1] = j
            for j, new_tail in enumerate(pred_data[3 * i + 2][1::2]):
                neg_triplet["tail"][0].append([head, entity2id[new_tail], rel])
                if entity2id[new_tail] == tail:
                    neg_triplet["tail"][1] = j

            neg_triplet["head"][0] = np.array(neg_triplet["head"][0])
            neg_triplet["tail"][0] = np.array(neg_triplet["tail"][0])

            neg_triplets.append(neg_triplet)

    return neg_triplets


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    # One hot encode the node label feature and concat to node features
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1

    n_feats = (
        np.concatenate((label_feats, n_feats), axis=1)
        if n_feats is not None
        else label_feats
    )
    subgraph.ndata["feat"] = torch.FloatTensor(n_feats)

    head_id = np.argwhere(
        [label[0] == 0 and label[1] == 1 for label in n_labels]
    ).flatten()
    tail_id = np.argwhere(
        [label[0] == 1 and label[1] == 0 for label in n_labels]
    ).flatten()
    n_ids = np.zeros(n_nodes)
    # If head_id and tail_id are empty, that means head == tail
    if head_id.size == 0 and tail_id.size == 0:
        head_id = np.argwhere(
            [label[0] == 0 and label[1] == 0 for label in n_labels]
        ).flatten()
        n_ids[head_id] = 1
    else:
        n_ids[head_id] = 1
        n_ids[tail_id] = 2
    subgraph.ndata["id"] = torch.FloatTensor(n_ids)

    return subgraph


def get_subgraphs(
    all_links,
    adj_list,
    dgl_adj_list,
    max_node_label_value,
    id2entity,
    node_features=None,
    kge_entity2id=None,
):
    """
    Extracts subgraphs for each link in all_links.

    Parameters:
    - all_links: Iterable of triplets (head, tail, relation)
    - adj_list: List of adjacency matrices
    - dgl_adj_list: DGLGraph corresponding to adj_list
    - max_node_label_value: Maximum node label value for features
    - id2entity: Mapping from node IDs to entity names
    - node_features: Optional node feature matrix
    - kge_entity2id: Optional mapping from entities to KGE IDs

    Returns:
    - batched_graph: Batched DGLGraph containing all subgraphs
    - r_labels: Tensor of relation labels
    """
    subgraphs = []
    r_labels = []

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]

        # Extract nodes and labels for the subgraph
        nodes, node_labels = subgraph_extraction_labeling(
            (head, tail),
            rel,
            adj_list,
            h=params_.hop,
            enclosing_sub_graph=params_.enclosing_sub_graph,
            max_node_label_value=max_node_label_value,
        )[:2]

        # Create subgraph using the list of nodes
        subgraph = dgl_adj_list.subgraph(nodes)

        # Assign 'type' and 'label' edge attributes based on parent edge IDs
        parent_eids = subgraph.edata[dgl.EID]
        subgraph.edata["type"] = dgl_adj_list.edata["type"][parent_eids]
        subgraph.edata["label"] = torch.full(
            (subgraph.number_of_edges(),),
            rel,
            dtype=torch.long,
            device=subgraph.device,  # Ensure the tensor is on the same device as the graph
        )

        # Check if there is an edge between node 0 and node 1 with relation 'rel'
        if subgraph.number_of_nodes() > 1:
            if subgraph.has_edges_between(0, 1):
                try:
                    # Get edge IDs from node 0 to node 1
                    eids_between_roots = subgraph.edge_ids(0, 1, return_uv=False)
                    # Check if any of these edges have 'type' == rel
                    rel_link = (
                        (subgraph.edata["type"][eids_between_roots] == rel)
                        .nonzero(as_tuple=False)
                        .squeeze()
                    )
                except KeyError:
                    # No valid edge exists
                    rel_link = torch.tensor([], device=subgraph.device)
            else:
                rel_link = torch.tensor([], device=subgraph.device)

            # If no such edge exists, add it with the appropriate attributes
            if rel_link.numel() == 0:
                # Add edge from node 0 to node 1 with 'type' and 'label' equal to 'rel'
                subgraph = dgl.add_edges(
                    subgraph,
                    [0],
                    [1],
                    {
                        "type": torch.tensor(
                            [rel], dtype=torch.long, device=subgraph.device
                        ),
                        "label": torch.tensor(
                            [rel], dtype=torch.long, device=subgraph.device
                        ),
                    },
                )

        # Handle KGE nodes and features if applicable
        if kge_entity2id is not None:
            kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes]
        else:
            kge_nodes = None

        if node_features is not None:
            if kge_nodes is not None:
                n_feats = node_features[kge_nodes]
            else:
                n_feats = None
        else:
            n_feats = None

        # Prepare node features using the provided function
        subgraph = prepare_features(
            subgraph, node_labels, max_node_label_value, n_feats
        )

        # Append the processed subgraph and its label
        subgraphs.append(subgraph)
        r_labels.append(rel)

    # Batch all subgraphs into a single graph
    batched_graph = dgl.batch(subgraphs)
    num_nodes_per_graph = torch.tensor(
        [subgraph.number_of_nodes() for subgraph in subgraphs]
    )
    r_labels = torch.LongTensor(r_labels)

    return batched_graph, num_nodes_per_graph, r_labels


def save_to_file(neg_triplets, id2entity, id2relation, dataset):
    with open(os.path.join("./data", dataset, "ranking_head.txt"), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet["head"][0].reshape(-1, 3):
                f.write("\t".join([id2entity[s], id2relation[r], id2entity[o]]) + "\n")

    with open(os.path.join("./data", dataset, "ranking_tail.txt"), "w") as f:
        for neg_triplet in neg_triplets:
            for s, o, r in neg_triplet["tail"][0].reshape(-1, 3):
                f.write("\t".join([id2entity[s], id2relation[r], id2entity[o]]) + "\n")


def save_score_to_file(
    neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, dataset
):
    with open(
        os.path.join("./data", dataset, "grail_ranking_head_predictions.txt"), "w"
    ) as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(
                neg_triplet["head"][0], all_head_scores[50 * i : 50 * (i + 1)]
            ):
                f.write(
                    "\t".join(
                        [id2entity[s], id2relation[r], id2entity[o], str(head_score)]
                    )
                    + "\n"
                )

    with open(
        os.path.join("./data", dataset, "grail_ranking_tail_predictions.txt"), "w"
    ) as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(
                neg_triplet["tail"][0], all_tail_scores[50 * i : 50 * (i + 1)]
            ):
                f.write(
                    "\t".join(
                        [id2entity[s], id2relation[r], id2entity[o], str(tail_score)]
                    )
                    + "\n"
                )


def save_score_to_file_from_ruleN(
    neg_triplets, all_head_scores, all_tail_scores, id2entity, id2relation, dataset
):
    with open(
        os.path.join("./data", dataset, "grail_ruleN_ranking_head_predictions.txt"), "w"
    ) as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], head_score in zip(
                neg_triplet["head"][0], all_head_scores[50 * i : 50 * (i + 1)]
            ):
                f.write(
                    "\t".join(
                        [id2entity[s], id2relation[r], id2entity[o], str(head_score)]
                    )
                    + "\n"
                )

    with open(
        os.path.join("./data", dataset, "grail_ruleN_ranking_tail_predictions.txt"), "w"
    ) as f:
        for i, neg_triplet in enumerate(neg_triplets):
            for [s, o, r], tail_score in zip(
                neg_triplet["tail"][0], all_tail_scores[50 * i : 50 * (i + 1)]
            ):
                f.write(
                    "\t".join(
                        [id2entity[s], id2relation[r], id2entity[o], str(tail_score)]
                    )
                    + "\n"
                )


def get_rank(neg_links):
    head_neg_links = neg_links["head"][0]
    head_target_id = neg_links["head"][1]

    if head_target_id != 10000:
        data = get_subgraphs(
            head_neg_links,
            adj_list_,
            dgl_adj_list_,
            model_.gnn.max_label_value,
            id2entity_,
            node_features_,
            kge_entity2id_,
        )
        head_scores = model_(data).squeeze(1).detach().numpy()
        head_rank = np.argwhere(np.argsort(head_scores)[::-1] == head_target_id) + 1
    else:
        head_scores = np.array([])
        head_rank = 10000

    tail_neg_links = neg_links["tail"][0]
    tail_target_id = neg_links["tail"][1]

    if tail_target_id != 10000:
        data = get_subgraphs(
            tail_neg_links,
            adj_list_,
            dgl_adj_list_,
            model_.gnn.max_label_value,
            id2entity_,
            node_features_,
            kge_entity2id_,
        )
        tail_scores = model_(data).squeeze(1).detach().numpy()
        tail_rank = np.argwhere(np.argsort(tail_scores)[::-1] == tail_target_id) + 1
    else:
        tail_scores = np.array([])
        tail_rank = 10000

    return head_scores, head_rank, tail_scores, tail_rank


def get_kge_embeddings(dataset, kge_model):

    path = "./experiments/kge_baselines/{}_{}".format(kge_model, dataset)
    node_features = np.load(os.path.join(path, "entity_embedding.npy"))
    with open(os.path.join(path, "id2entity.json")) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}

    return node_features, kge_entity2id


def main(params):
    model = torch.load(params.model_path, map_location="cpu")

    adj_list, dgl_adj_list, triplets, entity2id, relation2id, id2entity, id2relation = (
        process_files(params.file_paths, model.relation2id, params.add_traspose_rels)
    )

    node_features, kge_entity2id = (
        get_kge_embeddings(params.dataset, params.kge_model)
        if params.use_kge_embeddings
        else (None, None)
    )

    if params.mode == "sample":
        neg_triplets = get_neg_samples_replacing_head_tail(triplets["links"], adj_list)
        save_to_file(
            neg_triplets, id2entity, id2relation, params.dataset
        )  # Passed dataset
    elif params.mode == "all":
        neg_triplets = get_neg_samples_replacing_head_tail_all(
            triplets["links"], adj_list
        )
    elif params.mode == "ruleN":
        neg_triplets = get_neg_samples_replacing_head_tail_from_ruleN(
            params.ruleN_pred_path, entity2id, relation2id
        )

    ranks = []
    all_head_scores = []
    all_tail_scores = []
    with mp.Pool(
        processes=None,
        initializer=intialize_worker,
        initargs=(
            model,
            adj_list,
            dgl_adj_list,
            id2entity,
            params,
            node_features,
            kge_entity2id,
        ),
    ) as p:
        for head_scores, head_rank, tail_scores, tail_rank in tqdm(
            p.imap(get_rank, neg_triplets), total=len(neg_triplets)
        ):
            ranks.append(head_rank)
            ranks.append(tail_rank)

            all_head_scores += head_scores.tolist()
            all_tail_scores += tail_scores.tolist()

    if params.mode == "ruleN":
        save_score_to_file_from_ruleN(
            neg_triplets,
            all_head_scores,
            all_tail_scores,
            id2entity,
            id2relation,
            params.dataset,
        )
    else:
        save_score_to_file(
            neg_triplets,
            all_head_scores,
            all_tail_scores,
            id2entity,
            id2relation,
            params.dataset,
        )

    isHit1List = [x for x in ranks if x <= 1]
    isHit5List = [x for x in ranks if x <= 5]
    isHit10List = [x for x in ranks if x <= 10]
    hits_1 = len(isHit1List) / len(ranks)
    hits_5 = len(isHit5List) / len(ranks)
    hits_10 = len(isHit10List) / len(ranks)

    mrr = np.mean(1 / np.array(ranks))

    logger.info(
        f"MRR | Hits@1 | Hits@5 | Hits@10 : {mrr} | {hits_1} | {hits_5} | {hits_10}"
    )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Testing script for hits@10")

    # Experiment setup params
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="fb_v2_margin_loss",
        help="Experiment name. Log file with this name will be created",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="FB237_v2", help="Path to dataset"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="sample",
        choices=["sample", "all", "ruleN"],
        help="Negative sampling mode",
    )
    parser.add_argument(
        "--use_kge_embeddings",
        "-kge",
        type=bool,
        default=False,
        help="whether to use pretrained KGE embeddings",
    )
    parser.add_argument(
        "--kge_model",
        type=str,
        default="TransE",
        help="Which KGE model to load entity embeddings from",
    )
    parser.add_argument(
        "--enclosing_sub_graph",
        "-en",
        type=bool,
        default=True,
        help="whether to only consider enclosing subgraph",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=3,
        help="How many hops to go while eextracting subgraphs?",
    )
    parser.add_argument(
        "--add_traspose_rels",
        "-tr",
        type=bool,
        default=False,
        help="Whether to append adj matrix list with symmetric relations?",
    )
    params = parser.parse_args()

    params.file_paths = {
        "graph": os.path.join("./data", params.dataset, "train.txt"),
        "links": os.path.join("./data", params.dataset, "test.txt"),
    }

    params.ruleN_pred_path = os.path.join(
        "./data", params.dataset, "pos_predictions.txt"
    )
    params.model_path = os.path.join(
        "experiments", params.experiment_name, "best_graph_classifier.pth"
    )

    file_handler = logging.FileHandler(
        os.path.join(
            "experiments", params.experiment_name, f"log_rank_test_{time.time()}.txt"
        )
    )
    logger = logging.getLogger()
    logger.addHandler(file_handler)

    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("============================================")

    main(params)
