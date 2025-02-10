import os
import shutil
import argparse
import logging
import random
import logging
import numpy as np
import torch
from scipy.sparse import SparseEfficiencyWarning

from subgraph_extraction.datasets import SubgraphDataset, generate_subgraph_datasets
from utils.initialization_utils import initialize_experiment, initialize_model
from utils.graph_utils import collate_dgl, move_batch_to_device_dgl

from model.dgl.graph_classifier import GraphClassifier as dgl_model

from managers.evaluator import Evaluator
from managers.trainer import Trainer

from warnings import simplefilter


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


seed_everything(42)


def main(params):

    simplefilter(action="ignore", category=UserWarning)
    simplefilter(action="ignore", category=SparseEfficiencyWarning)

    params.db_path = f"{params.data_dir}/data/{params.dataset}/subgraphs_en_{params.enclosing_sub_graph}_neg_{params.num_neg_samples_per_link}_hop_{params.hop}"

    if params.preprocess and os.path.isdir(params.db_path):
        shutil.rmtree(params.db_path)
    if not os.path.isdir(params.db_path):
        generate_subgraph_datasets(params)

    train = SubgraphDataset(
        params.db_path,
        "train_pos",
        "train_neg",
        params.file_paths,
        add_traspose_rels=params.add_traspose_rels,
        num_neg_samples_per_link=params.num_neg_samples_per_link,
        use_kge_embeddings=params.use_kge_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        file_name=params.train_file,
    )
    valid = SubgraphDataset(
        params.db_path,
        "valid_pos",
        "valid_neg",
        params.file_paths,
        add_traspose_rels=params.add_traspose_rels,
        num_neg_samples_per_link=params.num_neg_samples_per_link,
        use_kge_embeddings=params.use_kge_embeddings,
        dataset=params.dataset,
        kge_model=params.kge_model,
        file_name=params.valid_file,
    )

    params.num_rels = train.num_rels
    params.aug_num_rels = train.aug_num_rels
    params.inp_dim = train.n_feat_dim

    # Log the max label value to save it in the model. This will be used to cap the labels generated on test set.
    params.max_label_value = train.max_n_label

    graph_classifier = initialize_model(params, dgl_model, params.load_model)

    logging.info(f"Device: {params.device}")
    logging.info(
        f"Input dim : {params.inp_dim}, # Relations : {params.num_rels}, # Augmented relations : {params.aug_num_rels}"
    )

    valid_evaluator = Evaluator(params, graph_classifier, valid)

    trainer = Trainer(params, graph_classifier, train, valid_evaluator)

    logging.info("Starting training with full batch...")

    trainer.train()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="TransE model")

    # Experiment setup params
    parser.add_argument(
        "--experiment_name",
        "-e",
        type=str,
        default="default",
        help="A folder with this name would be created to dump saved models and log files",
    )
    parser.add_argument(
        "--dataset", "-d", type=str, default="WN18RR", help="Dataset string"
    )
    parser.add_argument("--gpu", type=int, default=0, help="Which GPU to use?")
    parser.add_argument("--disable_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument(
        "--load_model", action="store_true", help="Load existing model?"
    )
    parser.add_argument(
        "--train_file",
        "-tf",
        type=str,
        default="train",
        help="Name of file containing training triplets",
    )
    parser.add_argument(
        "--valid_file",
        "-vf",
        type=str,
        default="valid",
        help="Name of file containing validation triplets",
    )
    parser.add_argument(
        "--preprocess", action="store_true", help="Recreate preprocessed subgraphs?"
    )

    # Training regime params
    parser.add_argument(
        "--num_epochs",
        "-ne",
        type=int,
        default=100,
        help="Learning rate of the optimizer",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=3,
        help="Interval of epochs to evaluate the model?",
    )
    parser.add_argument(
        "--eval_every_iter",
        type=int,
        default=455,
        help="Interval of iterations to evaluate the model?",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Interval of epochs to save a checkpoint of the model?",
    )
    parser.add_argument(
        "--early_stop", type=int, default=100, help="Early stopping patience"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Which optimizer to use?"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Learning rate of the optimizer"
    )
    parser.add_argument(
        "--clip", type=int, default=1000, help="Maximum gradient norm allowed"
    )
    parser.add_argument(
        "--l2", type=float, default=5e-4, help="Regularization constant for GNN weights"
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=10,
        help="The margin between positive and negative samples in the max-margin loss",
    )

    # Data processing pipeline params
    parser.add_argument(
        "--max_links",
        type=int,
        default=1000000,
        help="Set maximum number of train links (to fit into memory)",
    )
    parser.add_argument(
        "--hop", type=int, default=3, help="Enclosing subgraph hop number"
    )
    parser.add_argument(
        "--max_nodes_per_hop",
        "-mnph",
        type=int,
        default=None,
        help="if > 0, upper bound the # nodes per hop by subsampling",
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
        "--model_type",
        "-m",
        type=str,
        choices=["ssp", "dgl"],
        default="dgl",
        help="what format to store subgraphs in for model",
    )
    parser.add_argument(
        "--constrained_neg_prob",
        "-cn",
        type=float,
        default=0.0,
        help="with what probability to sample constrained heads/tails while neg sampling",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--num_neg_samples_per_link",
        "-neg",
        type=int,
        default=1,
        help="Number of negative examples to sample per positive link",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloading processes"
    )
    parser.add_argument(
        "--add_traspose_rels",
        "-tr",
        type=bool,
        default=False,
        help="whether to append adj matrix list with symmetric relations",
    )
    parser.add_argument(
        "--enclosing_sub_graph",
        "-en",
        type=bool,
        default=True,
        help="whether to only consider enclosing subgraph",
    )

    # Model params
    parser.add_argument(
        "--rel_emb_dim", "-r_dim", type=int, default=32, help="Relation embedding size"
    )
    parser.add_argument(
        "--attn_rel_emb_dim",
        "-ar_dim",
        type=int,
        default=32,
        help="Relation embedding size for attention",
    )
    parser.add_argument(
        "--emb_dim", "-dim", type=int, default=32, help="Entity embedding size"
    )
    parser.add_argument(
        "--num_gcn_layers", "-l", type=int, default=3, help="Number of GCN layers"
    )
    parser.add_argument(
        "--num_bases",
        "-b",
        type=int,
        default=4,
        help="Number of basis functions to use for GCN weights",
    )
    parser.add_argument(
        "--dropout", type=float, default=0, help="Dropout rate in GNN layers"
    )
    parser.add_argument(
        "--edge_dropout",
        type=float,
        default=0.5,
        help="Dropout rate in edges of the subgraphs",
    )
    parser.add_argument(
        "--gnn_agg_type",
        "-a",
        type=str,
        choices=["sum", "mlp", "gru"],
        default="sum",
        help="what type of aggregation to do in gnn msg passing",
    )
    parser.add_argument(
        "--add_ht_emb",
        "-ht",
        type=bool,
        default=True,
        help="whether to concatenate head/tail embedding with pooled graph representation",
    )
    parser.add_argument(
        "--has_attn",
        "-attn",
        type=bool,
        default=True,
        help="whether to have attn in model or not",
    )

    params = parser.parse_args()
    initialize_experiment(params, __file__)

    params.data_dir = "/gpfs/workdir/yutaoc/grail"

    params.file_paths = {
        "train": os.path.join(
            params.data_dir, "data/{}/{}.txt".format(params.dataset, params.train_file)
        ),
        "valid": os.path.join(
            params.data_dir, "data/{}/{}.txt".format(params.dataset, params.valid_file)
        ),
    }



    relation2id_file = os.path.join(params.data_dir, "data", params.dataset, "relation2id.txt")
    if os.path.exists(relation2id_file):
        saved_relation2id = {}
        with open(relation2id_file, "r") as f:
            next(f)  # Skip First Line
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    relation = parts[0]
                    rel_id = int(parts[1])
                    saved_relation2id[relation] = rel_id
        params.saved_relation2id = saved_relation2id
        logging.info(f"Loaded saved_relation2id from {relation2id_file}")
    else:
        params.saved_relation2id = None
        logging.info(f"File saved_relation2id from {relation2id_file} not found")



    entity2id_file = os.path.join(params.data_dir, "data", params.dataset, "entity2id.txt")
    if os.path.exists(entity2id_file):
        saved_entity2id = {}
        with open(entity2id_file, "r") as f:
            next(f)  # Skip First Line
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    entity = parts[0]
                    rel_id = int(parts[1])
                    saved_entity2id[entity] = rel_id
        params.saved_entity2id = saved_entity2id
        logging.info(f"Loaded saved_entity2id from {entity2id_file}")
    else:
        params.saved_entity2id = None
        logging.info(f"File saved_entity2id from {entity2id_file} not found")


    disjoint_file = os.path.join(params.data_dir, "data", params.dataset, "DisjointWith_axioms.txt")
    if os.path.exists(disjoint_file):
        disjoint_ontology = {}
        with open(disjoint_file, "r") as f:
            for line in f:
                # remove comma from the end
                line = line.strip()[:-1]
                if not line:  # Skip empty lines or lines with trailing commas
                    continue
                parts = line.split(",")
                if len(parts) > 1:  # Ensure valid format
                    key = int(parts[0])  # Convert the key to integer
                    values = list(map(int, parts[1:]))  # Convert disjoint values to integers
                    disjoint_ontology[key] = values
        params.disjoint_ontology = disjoint_ontology
        logging.info(f"Loaded disjoint_ontology from {disjoint_file} ith the length {len(disjoint_ontology)}")
    else:
        params.disjoint_ontology = None
        logging.info(f"File disjoint_ontology from {disjoint_file} not found")

    range_file = os.path.join(params.data_dir, "data", params.dataset, "Domain_axioms.txt")
    if os.path.exists(range_file):
        range_ontology = {}
        with open(range_file, "r") as f:
            for line in f:
                # remove comma from the end
                line = line.strip()[:-1]
                if not line:  # Skip empty lines or lines with trailing commas
                    continue
                parts = line.split(",")
                if len(parts) > 1:  # Ensure valid format
                    key = int(parts[0])  # Convert the key to integer
                    values = list(map(int, parts[1:]))  # Convert disjoint values to integers
                    range_ontology[key] = values
        params.range_ontology = range_ontology
        logging.info(f"Loaded range_file from {range_file} ith the length {len(range_ontology)}")
    else:
        params.range_ontology = None
        logging.info(f"File range_file from {range_file} not found")

    domain_file = os.path.join(params.data_dir, "data", params.dataset, "Range_axioms.txt")
    if os.path.exists(disjoint_file):
        domain_ontology = {}
        with open(domain_file, "r") as f:
            for line in f:
                # remove comma from the end
                line = line.strip()[:-1]
                if not line:  # Skip empty lines or lines with trailing commas
                    continue
                parts = line.split(",")
                if len(parts) > 1:  # Ensure valid format
                    key = int(parts[0])  # Convert the key to integer
                    values = list(map(int, parts[1:]))  # Convert disjoint values to integers
                    domain_ontology[key] = values
        params.domain_ontology = domain_ontology
        logging.info(f"Loaded domain_file from {domain_file} ith the length {len(domain_ontology)}")
    else:
        params.domain_ontology = None
        logging.info(f"File domain_file from {domain_file} not found")

    # Read asymmetric properties axioms
    asymmetric_file = os.path.join(params.data_dir, "data", params.dataset, "AsymmetricProperties_axioms.txt")
    if os.path.exists(asymmetric_file):
        with open(asymmetric_file, "r") as f:
            asymmetric_ontology = [line.strip() for line in f if line.strip()]
        params.asymmetric_ontology = asymmetric_ontology
        logging.info(f"Loaded asymmetric_ontology from {asymmetric_file} with the length {len(asymmetric_ontology)}")
    else:
        params.asymmetric_ontology = None
        logging.info(f"File asymmetric_ontology from {asymmetric_file} not found")

    # Read irreflexive properties axioms
    irreflexive_file = os.path.join(params.data_dir, "data", params.dataset, "IrreflexiveProperties_axioms.txt")
    if os.path.exists(irreflexive_file):
        with open(irreflexive_file, "r") as f:
            irreflexive_ontology = [line.strip() for line in f if line.strip()]
        params.irreflexive_ontology = irreflexive_ontology
        logging.info(f"Loaded irreflexive_ontology from {irreflexive_file} with the length {len(irreflexive_ontology)}")
    else:
        params.irreflexive_ontology = None
        logging.info(f"File irreflexive_ontology from {irreflexive_file} not found")


    if not params.disable_cuda and torch.cuda.is_available():
        params.device = torch.device("cuda:%d" % params.gpu)
    else:
        params.device = torch.device("cpu")

    params.collate_fn = collate_dgl
    params.move_batch_to_device = move_batch_to_device_dgl

    main(params)