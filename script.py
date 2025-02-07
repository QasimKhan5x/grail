folder = "/gpfs/workdir/yutaoc/grail/data/NELL-Elisa"

# Load entity2id and relation2id mappings
def load_mapping(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        mapping = {}
        for line in lines[1:]:  # Skip the first line (number of lines)
            tag, id_ = line.strip().split("\t")
            mapping[id_] = tag
        return mapping

entity2id_path = folder + "/entity2id.txt"
relation2id_path = folder + "/relation2id.txt"

entity_mapping = load_mapping(entity2id_path)
relation_mapping = load_mapping(relation2id_path)

# Process train, valid, and test files
for file in ["train", "valid", "test"]:
    if file == "train":
        input_file = folder + "/train2id_Consistent_withAugmentation.txt"
    else:
        input_file = folder + f"/{file}2id_Consistent.txt"
    output_file = folder + "/" + file + ".txt"

    # Open the input file for reading and the output file for writing
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        # Read all lines from the input file
        lines = infile.readlines()

        # Skip the first line and process the rest
        for line in lines[1:]:
            if file == "test":
                # test file is tab-separated for the first column
                line = line.split("\t")[1]
            # Split the line into parts (assuming whitespace-separated values)
            parts = line.strip().split()

            # Ensure the line has exactly 3 elements (H, T, P)
            if len(parts) == 3:
                # Replace H, T, and P IDs with their tags
                h_tag = entity_mapping.get(parts[0], parts[0])  # Default to ID if not found
                t_tag = entity_mapping.get(parts[1], parts[1])
                p_tag = relation_mapping.get(parts[2], parts[2])

                # Reorder to H, P, T and write to the output file
                reordered_line = f"{h_tag} {p_tag} {t_tag}\n"
                outfile.write(reordered_line)
