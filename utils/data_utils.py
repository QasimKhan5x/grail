import os
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import logging


def plot_rel_dist(adj_list: list, output_filename: str) -> None:
    """
    Plot the distribution of relations based on adjacency matrices.

    Parameters:
        adj_list (list): List of adjacency matrices for each relation.
        output_filename (str): File path to save the plot.
    """
    rel_counts = [adj.count_nonzero() for adj in adj_list]

    plt.figure(figsize=(12, 8))
    plt.plot(rel_counts)
    plt.title("Relation Distribution")
    plt.xlabel("Relation Index")
    plt.ylabel("Count of Non-zero Entries")
    plt.savefig(output_filename, dpi=300)
    plt.close()


def parse_triplets(
    file_path: str,
    entity2id: dict,
    relation2id: dict,
    saved_relation2id: dict = None,
    saved_entity2id: dict = None
):
    """
    Parse triplets from a file and update entity and relation mappings.

    Parameters:
        file_path (str): Path to the file containing triplets.
        entity2id (dict): Entity-to-ID mapping.
        relation2id (dict): Relation-to-ID mapping.
        saved_relation2id (dict): Predefined relation-to-ID mapping (optional).

    Returns:
        list: Processed triplets as [subject, object, relation] indices.
    """
    triplets = []
    if saved_entity2id is None:
        entity_counter = len(entity2id)
    if saved_relation2id is None:
        relation_counter = len(relation2id)

    with open(file_path, "r") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                subject, relation, obj = line.split()
            except ValueError:
                logging.warning("Line format error in %s: %s", file_path, line.strip())
                continue

            if subject == obj:
                continue

            if saved_entity2id is None:
                if subject not in entity2id:
                    entity2id[subject] = entity_counter
                    entity_counter += 1
                if obj not in entity2id:
                    entity2id[obj] = entity_counter
                    entity_counter += 1
            else:
                if subject not in entity2id:
                    logging.warning("Subject %s not found in saved_entity2id; skipping triplet.", subject)
                    continue
                if obj not in entity2id:
                    logging.warning("Object %s not found in saved_entity2id; skipping triplet.", obj)
                    continue

            if saved_relation2id is None:
                if relation not in relation2id:
                    relation2id[relation] = relation_counter
                    relation_counter += 1
            else:
                if relation not in relation2id:
                    logging.warning("Relation %s not found in saved_relation2id; skipping triplet.", relation)
                    continue

            if relation in relation2id:
                triplets.append([entity2id[subject], entity2id[obj], relation2id[relation]])

    return np.array(triplets)


def build_adjacency_list(
    train_triplets: np.ndarray, relation_count: int, entity_count: int
):
    """
    Build adjacency matrices for each relation from the training data.

    Parameters:
        train_triplets (np.ndarray): Array of training triplets [subject, object, relation].
        relation_count (int): Total number of relations.
        entity_count (int): Total number of entities.

    Returns:
        list: List of sparse adjacency matrices (CSR format) for each relation.
    """
    adj_list = []
    for relation_id in range(relation_count):
        # Find all triplets corresponding to the current relation
        indices = np.argwhere(train_triplets[:, 2] == relation_id).flatten()
        rows = train_triplets[:, 0][indices]
        cols = train_triplets[:, 1][indices]
        data = np.ones(len(indices), dtype=np.uint8)

        # Create the adjacency matrix in CSR format
        adj_matrix = csr_matrix(
            (data, (rows, cols)), shape=(entity_count, entity_count)
        )
        adj_list.append(adj_matrix)

    return adj_list



def process_files(file_paths: dict, saved_relation2id: dict = None, saved_entity2id: dict = None):
    entity2id = saved_entity2id.copy() if saved_entity2id is not None else {}
    relation2id = saved_relation2id.copy() if saved_relation2id is not None else {}
    triplets = {}

    for file_type, file_path in file_paths.items():
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        triplets[file_type] = parse_triplets(
            file_path, entity2id, relation2id, saved_relation2id, saved_entity2id
        )

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}
    adj_list = build_adjacency_list(triplets["train"], len(relation2id), len(entity2id))
    
    # Test Code: Log the loaded mappings to verify they are as expected.
    sample_entity2id = dict(list(entity2id.items())[:5]) 
    sample_relation2id = dict(list(relation2id.items())[:5])
    logging.info("Test Code: Loaded entity2id mapping (%d entries): %s", len(entity2id), sample_entity2id)
    logging.info("Test Code: Loaded relation2id mapping (%d entries): %s", len(relation2id), sample_relation2id)
    
    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation


def save_to_file(
    directory: str, file_name: str, triplets: list, id2entity: dict, id2relation: dict
):
    """
    Save triplets to a file in human-readable format.

    Parameters:
        directory (str): Directory to save the file.
        file_name (str): Name of the output file.
        triplets (list): List of triplets [subject, object, relation].
        id2entity (dict): Mapping from entity IDs to entity names.
        id2relation (dict): Mapping from relation IDs to relation names.
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, file_name)

    with open(file_path, "w") as file:
        for subject, obj, relation in triplets:
            file.write(
                f"{id2entity[subject]}\t{id2relation[relation]}\t{id2entity[obj]}\n"
            )
