import os
import argparse
import logging
import json
import time
import random

from scipy.sparse import issparse
import multiprocessing as mp
import scipy.sparse as ssp
from tqdm import tqdm
import torch
import numpy as np
import dgl
import matplotlib.pyplot as plt

from subgraph_extraction.graph_sampler import subgraph_extraction_labeling
from utils.graph_utils import ssp_multigraph_to_dgl

import psutil
import os


def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 2):.2f} MB")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def process_in_batches(test_links, adj_list, batch_size, num_samples=50):
    """
    Process test_links in batches to avoid memory issues.
    """
    num_links = len(test_links)
    for i in range(0, num_links, batch_size):
        batch_links = test_links[i : i + batch_size]
        yield get_neg_samples_replacing_head_tail(batch_links, adj_list, num_samples)


def get_neg_samples_replacing_head_tail(test_links, adj_list, num_samples=50):
    """
    Generate negative samples by replacing heads and tails in test_links,
    returning exactly `num_samples` negative examples for head and tail each.
    """
    # adj_list[r] is a sparse adjacency matrix for relation r
    n = adj_list[0].shape[0]  # number of entities
    r = len(adj_list)

    heads, tails, rels = test_links[:, 0], test_links[:, 1], test_links[:, 2]
    neg_triplets = []

    for head, tail, rel in zip(heads, tails, rels):
        neg_triplet = {"head": [[], 0], "tail": [[], 0]}

        # 1) Head replaced negative samples
        #    We keep the original triplet as the first element:
        head_list = [[head, tail, rel]]
        while len(head_list) < num_samples:
            neg_tail = np.random.randint(0, n)
            if neg_tail != head and adj_list[rel][head, neg_tail] == 0:
                head_list.append([head, neg_tail, rel])
        neg_triplet["head"][0] = np.array(head_list)

        # 2) Tail replaced negative samples
        tail_list = [[head, tail, rel]]
        while len(tail_list) < num_samples:
            neg_head = np.random.randint(0, n)
            if neg_head != tail and adj_list[rel][neg_head, tail] == 0:
                tail_list.append([neg_head, tail, rel])
        neg_triplet["tail"][0] = np.array(tail_list)

        neg_triplets.append(neg_triplet)

    return neg_triplets


def process_files(files, saved_relation2id, add_traspose_rels):
    """
    Build sparse adjacency lists for each known relation, etc.
    """
    entity2id = {}
    relation2id = saved_relation2id

    triplets = {}
    ent = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split("\n") if line.strip()]

        for triplet in file_data:
            if len(triplet) < 3:
                continue
            if triplet[0] == triplet[2]:
                # skip self-loops, e.g. (a, r, a)
                continue
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1

            # Only save triplets for known relations
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

    # Construct adjacency list from 'graph' portion (train set)
    num_entities = len(entity2id)
    adj_list = []
    for i in range(len(saved_relation2id)):
        # indices where relation == i
        idx = np.where(triplets["graph"][:, 2] == i)[0]
        rows = triplets["graph"][idx, 0]
        cols = triplets["graph"][idx, 1]
        data = np.ones(len(idx), dtype=np.uint8)

        adj = ssp.csc_matrix((data, (rows, cols)), shape=(num_entities, num_entities))
        adj_list.append(adj)

    # Optionally add transpose relations
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


########################
### SUBGRAPH UTILS #####
########################


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


def get_kge_embeddings(dataset, kge_model):
    """
    Load pretrained KGE embeddings if needed.
    """
    path = f"./experiments/kge_baselines/{kge_model}_{dataset}"
    node_features = np.load(os.path.join(path, "entity_embedding.npy"))
    with open(os.path.join(path, "id2entity.json")) as json_file:
        kge_id2entity = json.load(json_file)
        kge_entity2id = {v: int(k) for k, v in kge_id2entity.items()}
    return node_features, kge_entity2id


#############################
### SUBGRAPH EXTRACTION  ####
#############################


def prepare_features(subgraph, n_labels, max_n_label, n_feats=None):
    """
    Prepare the node features and attach them to subgraph.ndata['feat'].
    """
    n_nodes = subgraph.number_of_nodes()
    label_feats = np.zeros((n_nodes, max_n_label[0] + 1 + max_n_label[1] + 1))
    label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
    label_feats[np.arange(n_nodes), max_n_label[0] + 1 + n_labels[:, 1]] = 1

    if n_feats is not None:
        n_feats = np.concatenate((label_feats, n_feats), axis=1)
    else:
        n_feats = label_feats

    subgraph.ndata["feat"] = torch.FloatTensor(n_feats)

    # Mark head/tail
    head_id = np.argwhere((n_labels[:, 0] == 0) & (n_labels[:, 1] == 1))
    tail_id = np.argwhere((n_labels[:, 0] == 1) & (n_labels[:, 1] == 0))
    n_ids = np.zeros(n_nodes)
    n_ids[head_id] = 1  # head
    n_ids[tail_id] = 2  # tail
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
    Extract subgraphs for each link in `all_links`.
    """

    subgraphs = []
    r_labels = []

    # We combine all adjacency for BFS-based extraction
    # (adj_list has multiple relations, so sum them to get an undirected adjacency)
    adj_matrix = sum(adj_list)
    adj_matrix = adj_matrix + adj_matrix.T

    for link in all_links:
        head, tail, rel = link[0], link[1], link[2]
        # BFS subgraph extraction + labeling
        nodes, node_labels = subgraph_extraction_labeling(
            (head, tail),
            rel,
            adj_matrix,
            h=params_.hop,
            enclosing_sub_graph=params_.enclosing_sub_graph,
            max_node_label_value=max_node_label_value,
            max_nodes_per_hop=params_.max_nodes_per_hop
        )[:2]

        # Subgraph
        sg = dgl_adj_list.subgraph(nodes)
        # In DGL, edges keep a reference to the original graph's EIDs
        parent_eids = sg.edata[dgl.EID]
        sg.edata["type"] = dgl_adj_list.edata["type"][parent_eids]
        sg.edata["label"] = torch.full((sg.number_of_edges(),), rel, dtype=torch.long)

        # Possibly add missing edge between node 0 and 1 if it doesn't exist
        if sg.number_of_nodes() > 1:
            if not sg.has_edges_between(0, 1):
                # Add the correct edge for (head->tail)
                sg = dgl.add_edges(
                    sg,
                    [0],
                    [1],
                    {
                        "type": torch.tensor([rel], dtype=torch.long),
                        "label": torch.tensor([rel], dtype=torch.long),
                    },
                )

        # If we have KGE embeddings, map the subgraph's nodes to KGE IDs
        if kge_entity2id is not None:
            kge_nodes = [kge_entity2id[id2entity[n]] for n in nodes]
        else:
            kge_nodes = None

        # Prepare node features
        if node_features is not None and kge_nodes is not None:
            n_feats = node_features[kge_nodes]
        else:
            n_feats = None

        sg = prepare_features(sg, node_labels, max_node_label_value, n_feats)
        subgraphs.append(sg)
        r_labels.append(rel)

    batched_graph = dgl.batch(subgraphs)
    r_labels = torch.LongTensor(r_labels)
    return batched_graph, r_labels


###################################
### SCORING & RANK COMPUTATIONS ###
###################################


def save_score_to_file(
    neg_triplets, head_scores_batch, tail_scores_batch, id2entity, id2relation, dataset
):
    """
    Append the newly computed scores to disk immediately (batch by batch).
    """
    # Each `neg_triplets` is a list of size = batch_size, each item is a dict:
    #    {'head': [array_of_triplets, 0], 'tail': [array_of_triplets, 0]}
    # The arrays have shape (num_samples, 3).

    # For each item in `neg_triplets`, we have `num_samples` for head, and `num_samples` for tail
    # so total is `len(neg_triplets) * num_samples` scores in each of head_scores_batch / tail_scores_batch
    # We expect them to line up in the order we appended them.

    # We'll open two files in append mode:
    head_path = os.path.join("/gpfs/workdir/yutaoc/grail/data", dataset, "grail_ranking_head_predictions.txt")
    tail_path = os.path.join("/gpfs/workdir/yutaoc/grail/data", dataset, "grail_ranking_tail_predictions.txt")

    # Be sure the number of negative triplets matches the number of scores
    # By default, we used 50 samples per triple => 50 * len(neg_triplets).
    # But let's not hardcode 50. We'll do it dynamically below.
    idx_head_scores = 0
    idx_tail_scores = 0

    with open(head_path, "a") as f_head, open(tail_path, "a") as f_tail:
        for i, neg_trip in enumerate(neg_triplets):
            head_array = neg_trip["head"][0]  # shape (#neg_samples, 3)
            tail_array = neg_trip["tail"][0]
            n_h = head_array.shape[0]
            n_t = tail_array.shape[0]

            # The slice of head_scores_batch relevant to this test link
            local_head_scores = head_scores_batch[
                idx_head_scores : idx_head_scores + n_h
            ]
            local_tail_scores = tail_scores_batch[
                idx_tail_scores : idx_tail_scores + n_t
            ]

            idx_head_scores += n_h
            idx_tail_scores += n_t

            # Write head predictions
            for [s, o, r], sc in zip(head_array, local_head_scores):
                f_head.write(
                    "\t".join([id2entity[s], id2relation[r], id2entity[o], str(sc)])
                    + "\n"
                )

            # Write tail predictions
            for [s, o, r], sc in zip(tail_array, local_tail_scores):
                f_tail.write(
                    "\t".join([id2entity[s], id2relation[r], id2entity[o], str(sc)])
                    + "\n"
                )


def get_rank(neg_links):
    """
    For a single test instance in neg_links (which is a dict with 'head' and 'tail'),
    compute the ranking metrics.
    """
    # This is called inside the multiprocessing pool.
    # neg_links["head"][0] => array of shape (num_samples, 3)
    # neg_links["head"][1] => index of the *true* triple among them, or 0 if it's the first?
    # In your snippet, it looks like we always put the gold triple as the first example (index 0).
    # So head_target_id = 0, tail_target_id = 0  in your revised sampling code.

    head_neg_links = neg_links["head"][0]
    tail_neg_links = neg_links["tail"][0]

    # By default, let's assume the "target" is at index=0
    head_target_id = 0
    tail_target_id = 0

    # If there's some case where it's 10000, skip
    if head_target_id >= head_neg_links.shape[0]:
        head_scores = np.array([])
        head_rank = 10000
    else:
        # Evaluate subgraphs
        batched_graph, r_labels = get_subgraphs(
            head_neg_links,
            adj_list_,
            dgl_adj_list_,
            model_.gnn.max_label_value,
            id2entity_,
            node_features_,
            kge_entity2id_,
        )
        with torch.no_grad():
            logits = model_((batched_graph, r_labels)).squeeze(1).cpu().numpy()
        # rank the gold triple
        # The gold triple was appended at index=0, so let's see where that score ends up among all
        sorted_indices = np.argsort(logits)[::-1]  # descending
        head_rank = int(np.where(sorted_indices == head_target_id)[0][0] + 1)
        head_scores = logits  # store them all

    # Tail side
    if tail_target_id >= tail_neg_links.shape[0]:
        tail_scores = np.array([])
        tail_rank = 10000
    else:
        batched_graph, r_labels = get_subgraphs(
            tail_neg_links,
            adj_list_,
            dgl_adj_list_,
            model_.gnn.max_label_value,
            id2entity_,
            node_features_,
            kge_entity2id_,
        )
        with torch.no_grad():
            logits = model_((batched_graph, r_labels)).squeeze(1).cpu().numpy()
        sorted_indices = np.argsort(logits)[::-1]  # descending
        tail_rank = int(np.where(sorted_indices == tail_target_id)[0][0] + 1)
        tail_scores = logits

    return head_scores, head_rank, tail_scores, tail_rank


#####################
####   MAIN()   #####
#####################


def main(params):
    seed_everything(42)

    logger.info("Loading model...")
    model = torch.load(params.model_path, map_location="cpu")

    logger.info("Processing files and building adjacency lists...")
    (
        adj_list,
        dgl_adj_list,
        triplets,
        entity2id,
        relation2id,
        id2entity,
        id2relation,
    ) = process_files(params.file_paths, model.relation2id, params.add_traspose_rels)

    node_features, kge_entity2id = (
        get_kge_embeddings(params.dataset, params.kge_model)
        if params.use_kge_embeddings
        else (None, None)
    )

    # We create a single multiprocessing pool upfront:
    pool = mp.Pool(
        processes=mp.cpu_count(),
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
    )

    ranks = []  # To store all ranks (both head+tail)
    sum_rr = 0.0  # For MRR on the fly
    count_ranks = 0  # total number of (head+tail) ranks

    # For the final hits@k, you either store all ranks or compute partial counts
    hits_count = {1: 0, 5: 0, 10: 0}
    thresholds_to_track = [1, 5, 10]

    logger.info("Starting negative sampling + ranking in batches...")
    batch_size = 200  # You can tune this
    total_test_links = len(triplets["links"])
    n_batches = (total_test_links + batch_size - 1) // batch_size

    # Before writing anything, clear (or create) your output files
    head_path = os.path.join(
        "/gpfs/workdir/yutaoc/grail/data",
        params.dataset,
        "grail_ranking_head_predictions.txt",
    )
    tail_path = os.path.join(
        "/gpfs/workdir/yutaoc/grail/data",
        params.dataset,
        "grail_ranking_tail_predictions.txt",
    )
    open(head_path, "w").close()  # clear file
    open(tail_path, "w").close()  # clear file

    for neg_triplets_batch in tqdm(
        process_in_batches(triplets["links"], adj_list, batch_size, params.num_samples),
        total=n_batches,
    ):
        # `neg_triplets_batch` is a list of length ~ batch_size, each item is the negative samples for 1 test triple
        results_iter = pool.imap(get_rank, neg_triplets_batch)

        # We'll gather the scores for writing to disk
        batch_head_scores = []
        batch_tail_scores = []

        # Extract ranks for MRR/Hits
        for head_scores, head_rank, tail_scores, tail_rank in results_iter:
            # accumulate final ranks if needed:
            ranks.append(head_rank)
            ranks.append(tail_rank)

            # on-the-fly MRR:
            if head_rank != 10000:
                sum_rr += 1.0 / head_rank
                count_ranks += 1
                for thr in thresholds_to_track:
                    if head_rank <= thr:
                        hits_count[thr] += 1

            if tail_rank != 10000:
                sum_rr += 1.0 / tail_rank
                count_ranks += 1
                for thr in thresholds_to_track:
                    if tail_rank <= thr:
                        hits_count[thr] += 1

            # Collect scores for saving
            batch_head_scores.extend(head_scores.tolist())
            batch_tail_scores.extend(tail_scores.tolist())

        # Now write out the batch's scores to disk (append mode)
        save_score_to_file(
            neg_triplets_batch,
            batch_head_scores,
            batch_tail_scores,
            id2entity,
            id2relation,
            params.dataset,
        )

        # Clear references to free memory
        del neg_triplets_batch
        del batch_head_scores
        del batch_tail_scores

        print_memory_usage()

    # Done with the pool
    pool.close()
    pool.join()

    # Final metrics
    # If you need all ranks for a detailed threshold curve, you have them in `ranks`.
    # Otherwise you could rely on partial sums.  We'll show both ways:

    # 1) Using partial sums:
    if count_ranks > 0:
        mrr = sum_rr / count_ranks
        hits_1 = hits_count[1] / count_ranks
        hits_5 = hits_count[5] / count_ranks
        hits_10 = hits_count[10] / count_ranks
        logger.info(
            f"[Partial sums] MRR={mrr:.4f}, Hits@1={hits_1:.4f}, Hits@5={hits_5:.4f}, Hits@10={hits_10:.4f}"
        )
    else:
        logger.info("No valid ranks computed (count_ranks=0).")

    # 2) If we want to do it from the raw list `ranks` (the final approach):
    #    This might be easier if you need hits@all thresholds.
    #    `ranks` contains 2 ranks per test triple (head, tail).
    ranks = [r for r in ranks if r != 10000]  # filter out invalid
    if len(ranks) == 0:
        logger.info("No valid ranks, cannot compute final metrics!")
        return

    ranks = np.array(ranks)
    mrr_final = np.mean(1.0 / ranks)
    hits1_final = np.mean(ranks <= 1)
    hits5_final = np.mean(ranks <= 5)
    hits10_final = np.mean(ranks <= 10)

    logger.info(
        f"[From final ranks array] MRR={mrr_final:.4f}, Hits@1={hits1_final:.4f}, "
        f"Hits@5={hits5_final:.4f}, Hits@10={hits10_final:.4f}"
    )

    # Optionally plot hits up to 50
    thresholds = range(1, 51)
    hits_curve = [np.mean(ranks <= t) for t in thresholds]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, hits_curve, marker="o", label="Hits@Threshold")
    plt.axhline(y=mrr_final, color="r", linestyle="--", label=f"MRR={mrr_final:.4f}")
    plt.title("Hits at Thresholds (1 to 50)")
    plt.xlabel("Threshold")
    plt.ylabel("Proportion of test links hitting at or below threshold")
    plt.grid(True)
    plt.legend()
    plt.savefig(params.dataset, dpi=300)
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Testing script for hits@10")

    parser.add_argument("--experiment_name", "-e", type=str, default="default")
    parser.add_argument("--dataset", "-d", type=str, default="WN18RR")
    parser.add_argument(
        "--mode", "-m", type=str, default="sample", choices=["sample", "all", "ruleN"]
    )
    parser.add_argument("--use_kge_embeddings", "-kge", type=bool, default=False)
    parser.add_argument("--kge_model", type=str, default="TransE")
    parser.add_argument("--enclosing_sub_graph", "-en", type=bool, default=True)
    parser.add_argument("--hop", type=int, default=3)
    parser.add_argument("--add_traspose_rels", "-tr", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default="best_graph_classifier.pth")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument(
        "--max_nodes_per_hop",
        "-mnph",
        type=int,
        default=None,
        help="if > 0, upper bound the # nodes per hop by subsampling",
    )

    params = parser.parse_args()
    params.file_paths = {
        "graph": os.path.join(
            "/gpfs/workdir/yutaoc/grail/data", params.dataset, "train.txt"
        ),
        "links": os.path.join(
            "/gpfs/workdir/yutaoc/grail/data", params.dataset, "test.txt"
        ),
    }
    params.ruleN_pred_path = os.path.join(
        "/gpfs/workdir/yutaoc/grail/data", params.dataset, "pos_predictions.txt"
    )
    params.model_path = os.path.join(
        "experiments", params.experiment_name, params.model_path
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
