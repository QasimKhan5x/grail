import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn import metrics
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Evaluator():
    def __init__(self, params, graph_classifier, data):
        self.params = params
        self.graph_classifier = graph_classifier
        self.data = data

    def eval(self, save=False):
        pos_scores = []
        pos_labels = []
        neg_scores = []
        neg_labels = []
        dataloader = DataLoader(self.data, batch_size=self.params.batch_size, shuffle=False, num_workers=self.params.num_workers, collate_fn=self.params.collate_fn)

        self.graph_classifier.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):

                data_pos, targets_pos, data_neg, targets_neg = self.params.move_batch_to_device(batch, self.params.device)
                score_pos = self.graph_classifier(data_pos)
                score_neg = self.graph_classifier(data_neg)

                pos_scores += score_pos.squeeze(1).detach().cpu().tolist()
                neg_scores += score_neg.squeeze(1).detach().cpu().tolist()
                pos_labels += targets_pos.tolist()
                neg_labels += targets_neg.tolist()

        # acc = metrics.accuracy_score(labels, preds)
        scores = np.array(pos_scores + neg_scores)
        scores_prob = 1 / (1 + np.exp(-scores))
        auc = metrics.roc_auc_score(pos_labels + neg_labels, scores_prob)
        auc_pr = metrics.average_precision_score(pos_labels + neg_labels, scores_prob)

        precision, recall, thresholds = metrics.precision_recall_curve(pos_labels + neg_labels, scores_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_index]
        print("Best Threshold: ", best_threshold)
        # Threshold for classification
        y_pred = (scores_prob >= best_threshold).astype(int)
        report = metrics.classification_report(pos_labels + neg_labels, y_pred, target_names=['Negative', 'Positive'])

        print("Validation Classification Report: \n", report)

        # # Combine the labels
        # all_labels = np.array(pos_labels + neg_labels)
        # # Identify positive examples
        # positive_indices = np.where(all_labels == 1)[0]  # Indices of positive labels
        # # Identify negative examples
        # negative_indices = np.where(all_labels == 0)[0]  # Indices of negative labels
        # wrongly_classified_positives = positive_indices[y_pred[positive_indices] != all_labels[positive_indices]]
        # wrongly_classified_negatives = negative_indices[y_pred[negative_indices] != all_labels[negative_indices]] - len(pos_labels)
        # # # Print the IDs (indices)
        # # print("IDs of wrongly classified positive examples:")
        # # print(wrongly_classified_positives.tolist())
        # # print("IDs of wrongly classified negative examples:")
        # # print(wrongly_classified_negatives.tolist())

        if save:
            pos_test_triplets_path = os.path.join(self.params.data_dir, 'data/{}/{}.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_test_triplets_path) as f:
                pos_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            pos_file_path = os.path.join(self.params.data_dir, 'data/{}/grail_{}_predictions.txt'.format(self.params.dataset, self.data.file_name))
            with open(pos_file_path, "w") as f:
                for ([s, r, o], score) in zip(pos_triplets, pos_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

            neg_test_triplets_path = os.path.join(self.params.data_dir, 'data/{}/neg_{}_0.txt'.format(self.params.dataset, self.data.file_name))
            with open(neg_test_triplets_path) as f:
                neg_triplets = [line.split() for line in f.read().split('\n')[:-1]]
            neg_file_path = os.path.join(self.params.data_dir, 'data/{}/grail_neg_{}_{}_predictions.txt'.format(self.params.dataset, self.data.file_name, self.params.constrained_neg_prob))
            with open(neg_file_path, "w") as f:
                for ([s, r, o], score) in zip(neg_triplets, neg_scores):
                    f.write('\t'.join([s, r, o, str(score)]) + '\n')

        return {'auc': auc, 'auc_pr': auc_pr}
