import statistics
import timeit
import os
import logging
import pdb
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn import metrics
from tqdm import tqdm


class Trainer:
    def __init__(self, params, graph_classifier, train, valid_evaluator=None):
        self.graph_classifier = graph_classifier
        self.valid_evaluator = valid_evaluator
        self.params = params
        self.train_data = train

        self.updates_counter = 0

        model_params = list(self.graph_classifier.parameters())
        logging.info(
            "Total number of parameters: %d"
            % sum(map(lambda x: x.numel(), model_params))
        )

        if params.optimizer == "SGD":
            self.optimizer = optim.SGD(
                model_params,
                lr=params.lr,
                momentum=params.momentum,
                weight_decay=self.params.l2,
            )
        if params.optimizer == "Adam":
            self.optimizer = optim.Adam(
                model_params, lr=params.lr, weight_decay=self.params.l2
            )

        self.criterion = nn.MarginRankingLoss(self.params.margin, reduction="sum")

        self.reset_training_state()

    def reset_training_state(self):
        self.best_metric = 0
        self.last_metric = 0
        self.not_improved_count = 0

    def train_epoch(self):
        total_loss = 0
        all_preds = []
        all_labels = []
        all_scores = []

        dataloader = DataLoader(
            self.train_data,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers,
            collate_fn=self.params.collate_fn,
        )
        self.graph_classifier.train()
        model_params = list(self.graph_classifier.parameters())

        if not self.params.eval_every_iter:
            self.params.eval_every_iter = len(dataloader) // 4


        if not self.params.eval_every_iter:
            self.params.eval_every_iter = len(dataloader) // 4

        for b_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            data_pos, targets_pos, data_neg, targets_neg = (
                self.params.move_batch_to_device(batch, self.params.device)
            )
            self.optimizer.zero_grad()
            score_pos = self.graph_classifier(data_pos)
            score_neg = self.graph_classifier(data_neg)
            batch_size = score_pos.size(0)

            # Squeeze `score_pos` to make it [16] instead of [16, 1], if necessary
            score_pos = score_pos.view(-1)
            score_neg_mean = score_neg.view(batch_size, -1).mean(dim=1)

            # Now, both score_pos and score_neg_mean should have shape [16]
            loss = self.criterion(
                score_pos,
                score_neg_mean,
                torch.ones(batch_size, device=self.params.device),
            )

            loss.backward()
            self.optimizer.step()
            self.updates_counter += 1

            with torch.no_grad():
                all_scores += (
                    score_pos.squeeze().detach().cpu().tolist()
                    + score_neg.squeeze().detach().cpu().tolist()
                )
                all_labels += targets_pos.tolist() + targets_neg.tolist()
                total_loss += loss

            if (
                self.valid_evaluator
                and self.params.eval_every_iter
                and self.updates_counter % self.params.eval_every_iter == 0
            ):
                tic = time.time()
                result = self.valid_evaluator.eval()
                logging.info(
                    "\nPerformance:" + str(result) + "in " + str(time.time() - tic)
                )

                if result["auc"] >= self.best_metric:
                    self.save_classifier()
                    self.best_metric = result["auc"]
                    self.not_improved_count = 0

                else:
                    self.not_improved_count += 1
                    if self.not_improved_count > self.params.early_stop:
                        logging.info(f"Validation performance didn\'t improve for {self.params.early_stop} epochs. Training stops.")
                        break
                self.last_metric = result["auc"]

        auc = metrics.roc_auc_score(all_labels, all_scores)
        auc_pr = metrics.average_precision_score(all_labels, all_scores)

        weight_norm = sum(map(lambda x: torch.norm(x), model_params))
        all_preds = [int(score >= 0.5) for score in all_scores]
        classification_report = metrics.classification_report(all_labels, all_preds)
        logging.info(f"Training Classification report: \n{classification_report}")

        return total_loss, auc, auc_pr, weight_norm

    def train(self):
        self.do_train = True
        self.reset_training_state()

        for epoch in range(1, self.params.num_epochs + 1):
            time_start = time.time()
            loss, auc, auc_pr, weight_norm = self.train_epoch()
            time_elapsed = time.time() - time_start
            logging.info(
                f"Epoch {epoch} with loss: {loss}, training auc: {auc}, training auc_pr: {auc_pr}, best validation AUC: {self.best_metric}, weight_norm: {weight_norm} in {time_elapsed}"
            )

            if epoch % self.params.save_every == 0:
                torch.save(self.graph_classifier, os.path.join(self.params.exp_dir, 'graph_classifier_chk.pth'))

    def save_classifier(self):
        torch.save(
            self.graph_classifier,
            os.path.join(self.params.exp_dir, "best_graph_classifier.pth"),
        ) 
        logging.info("Better models found w.r.t auc. Saved it!")
