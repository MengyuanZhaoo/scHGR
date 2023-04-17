from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.contrib.sampling import NeighborSampler
from tqdm.contrib import tenumerate
# self-defined
from utils import load_data_integrate
from models import GNN,EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
import csv
import time
import numpy as np


class Trainer:
    def __init__(self, params):
        self.params = params
        self.save_path=Path(self.params.save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.num_cells, self.pre_cells, self.num_genes, self.genes, self.num_labels, self.graph, self.train_ids, self.val_ids, self.test_ids, self.labels, self.id2label = load_data_integrate(
            params)
        self.labels = self.labels.to(self.device)
        self.model = GNN(in_feats=self.params.dense_dim,
                         n_hidden=self.params.hidden_dim,
                         n_classes=self.num_labels,
                         n_layers=self.params.n_layers,
                         gene_num=self.num_genes,
                         activation=F.relu,
                         dropout=self.params.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.lr,
                                          weight_decay=self.params.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.early_stopping = EarlyStopping(params.patience)
        if self.params.num_neighbors == 0:
            self.num_neighbors = self.num_cells + self.num_genes
        else:
            self.num_neighbors = self.params.num_neighbors
        self.early_stopping = EarlyStopping(params.patience)
        print(f"Train Number: {len(self.train_ids)}, Validation Number: {len(self.val_ids)}, Test Number: {len(self.test_ids)}")

    # ---------------------------------------------train-----------------------------------#
    def fit(self):
        max_test_acc, _train_acc, _epoch = 0, 0, 0
        record={}
        record['loss'] = []
        record['train_acc']=[]
        record['test_acc']=[]

        for epoch in range(self.params.n_epochs):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            print('In epoch: '+str(epoch))
            loss = self.train()
            print('out train')
            train_correct= self.evaluate(self.train_ids)
            test_correct= self.evaluate(self.val_ids)
            train_acc = train_correct / len(self.train_ids)
            test_acc = test_correct / len(self.val_ids)
            if max_test_acc <= test_acc:
                final_test_correct_num = test_correct
                _train_acc = train_acc
                _epoch = epoch
                max_test_acc = test_acc
                self.save_model()
            print(f">>>>Epoch {epoch:04d}: Train Acc {train_acc:.4f}, Loss {loss / len(self.train_ids):.4f}, Test correct {test_correct}, Test Acc {test_acc:.4f}")

            record['loss'].append(loss)
            record['train_acc'].append(train_acc)
            record['test_acc'].append(test_acc)
            if train_acc == 1:
                break
            self.early_stopping(test_acc)
            # when early stopping
            if self.early_stopping.early_stop:
                print("Early stopping at Epoch " + str(epoch))
                break
        print(f"---{self.params.species} {self.params.tissue} Best test result:---")
        print(f"Saving model at Epoch {_epoch:04d}, Train Acc {_train_acc:.4f}, Test Correct Num {final_test_correct_num}, Test Total Num {len(self.val_ids)}, Test Acc {final_test_correct_num / len(self.val_ids):.4f}")

        # plot loss and acc
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot([epoch for epoch in range(len(record['loss']))], record['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot([epoch for epoch in range(len(record['loss']))], record['train_acc'])
        plt.plot([epoch for epoch in range(len(record['loss']))], record['test_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.grid()
        plt.savefig(self.save_path / f"model" / f"train_record.pdf")
        plt.close()

    def train(self):
        self.model.train()
        total_loss = 0
        # self.graph.ndata['id'] == -1 means gene node
        for batch, nf in tenumerate(NeighborSampler(g=self.graph,
                                                    batch_size=self.params.batch_size,
                                                    expand_factor=self.num_neighbors,
                                                    num_hops=self.params.n_layers,
                                                    neighbor_type='in',
                                                    shuffle=True,
                                                    num_workers=8,
                                                    seed_nodes=self.train_ids.type(torch.int64))):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            logits,_ = self.model(nf)  # GNN
            batch_nids = nf.layer_parent_nid(-1).type(torch.long).to(device=self.device)
            loss = self.loss_fn(logits, self.labels[batch_nids])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss

    def evaluate(self, ids):
        self.model.eval()
        total_correct, total_unsure = 0, 0
        for nf in NeighborSampler(g=self.graph,
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=self.params.n_layers,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_workers=8,
                                  seed_nodes=ids.type(torch.int64)):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits,_ = self.model(nf)
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            logits = nn.functional.softmax(logits, dim=1).cpu().numpy()
            label_list = self.labels.cpu()[batch_nids]
            for pred, label in zip(logits, label_list):
                max_prob = pred.max().item()
                if pred.argmax().item() == label:
                    total_correct += 1
        return total_correct

    def save_model(self):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        model_save_path=self.save_path /f"model"
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True)
        torch.save(state, model_save_path/f"{self.params.species}-{self.params.tissue}.pt")

    # ---------------------------------------------predict-----------------------------------#
    def load_gold(self, file):
        with open(file, 'r') as f_gold:
            f_gold.readline()
            celltype_true = []
            cells=[]
            for line in f_gold:
                [_, cell, cell_type] = line.strip().split(',')
                cell = cell.replace('"', '')
                cell_type = cell_type.replace('"', '')
                cells.append(cell)
                celltype_true.append(cell_type)
        print(str(len(celltype_true)) + ' cells in gold file.')
        return cells, celltype_true

    def save_pred(self, pred, cells=[]):
        print('Saving predictions...')
        save_path = Path(self.params.save_path) / 'predict'
        if not save_path.exists():
            save_path.mkdir()
        if len(cells) > 0:
            df = pd.DataFrame({
                'cell': cells,
                'cell_type': pred})
        else:
            df = pd.DataFrame({'cell_type': pred})
        df.to_csv(
            save_path / (self.params.species + f"_{self.params.tissue}_predict.csv"),
            index=False)
        print(f"output has been stored in {self.params.species}_{self.params.tissue}_predict.csv")

    def metrices(self, celltype_true, celltype_pre):
        celltype_unique = list(set(celltype_true))
        celltype_unique.sort()
        # calculate TP,TF,FP,FN each type
        confusion = [['Type', 'TP', 'TN', 'FP', 'FN', 'Accuracy', 'Precision', 'Recall', 'F1-score']]
        for type in celltype_unique:
            TP, TN, FP, FN, accuracy, precision, recall, f1 = 0, 0, 0, 0, 0, 0, 0, 0
            for index in range(len(celltype_true)):
                if celltype_true[index] == type:
                    if celltype_pre[index] == celltype_true[index]:
                        TP += 1
                    else:
                        FN += 1
                elif celltype_pre[index] == type:
                    FP += 1
                else:
                    TN += 1
                accuracy = round((TP + TN) / len(celltype_true), 6)
                if TP + FP > 0:
                    precision = round(TP / (TP + FP), 6)
                if TP + FN > 0:
                    recall = round(TP / (TP + FN), 6)
                if precision + recall > 0:
                    f1 = round(2 * precision * recall / (precision + recall), 6)
            confusion.append([type, TP, TN, FP, FN, accuracy, precision, recall, f1])
        # a = confusion_matrix(celltype_true, celltype_pre)
        acc = metrics.accuracy_score(celltype_true, celltype_pre)  # All prediction==label samples/all samples
        pre_macro = metrics.precision_score(celltype_true, celltype_pre, average='macro', zero_division=0,
                                            labels=celltype_unique)  # 宏平均，精确率
        recall_macro = metrics.recall_score(celltype_true, celltype_pre, average='macro', zero_division=0,
                                            labels=celltype_unique)
        f1_macro = metrics.f1_score(celltype_true, celltype_pre, average='macro', zero_division=0,
                                    labels=celltype_unique)
        jcd = metrics.jaccard_score(celltype_true, celltype_pre, average='macro')  # 衡量2个集合的相似度，数值越大越好
        mcc = metrics.matthews_corrcoef(celltype_true, celltype_pre)
        print(celltype_unique)
        print('ACC:' + str(acc))
        print('precision-macro:' + str(pre_macro))
        print('recall-macro:' + str(recall_macro))
        print('F1-score-macro:' + str(f1_macro))
        print('jcd:' + str(jcd))
        print('mcc:' + str(mcc))
        with open(self.params.save_path + '/predict/evaluate.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(confusion)
            writer.writerow([' '])
            writer.writerow(['Accuracy: ', acc])
            writer.writerow(['Precision: ', pre_macro])
            writer.writerow(['Recall: ', recall_macro])
            writer.writerow(['F1-score: ', f1_macro])
            writer.writerow(['JCD: ', jcd])
            writer.writerow(['MCC: ', mcc])

    def load_model(self):
        model_path = Path(self.params.save_path) / 'model' / f'{self.params.species}-{self.params.tissue}.pt'
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state['model'])

    def save_weight(self, weight):
        print('Saving predicted weights...')
        save_path = Path(self.params.save_path) / 'predict'
        if not save_path.exists():
            save_path.mkdir()
        df = pd.DataFrame()
        for i in range(len(self.id2label)):
            type = self.id2label[i]
            data = weight[:,i]
            df[type] = data
        df.to_csv(
            save_path / (self.params.species + f"_{self.params.tissue}_weight.csv"),
            index=False)
        print(f"output has been stored in {self.params.species}_{self.params.tissue}_weight.csv")

    def save_alpha(self, alpha):
        print('Saving alpha...')
        save_path = Path(self.params.save_path) / 'predict'
        if not save_path.exists():
            save_path.mkdir()
        genes = list(self.genes.copy())
        genes.append('gene-self-loop')
        genes.append('cell-self-loop')
        df = pd.DataFrame({
            'genes': genes,
            'confidence': alpha.squeeze()})
        df.to_csv(
            save_path / 'gene_weight.csv',
            index=False)
        print("output has been stored in gene_weight.csv")

    def predict(self, gold_file=''):
        self.load_model()
        alpha = self.model._parameters['alpha'].data.cpu().numpy()  # [gene_num] is alpha of gene-gene, [gene_num+1] is alpha of cell-cell self loop
        self.save_alpha(alpha)
        self.model.eval()
        new_logits = torch.zeros((self.graph.number_of_nodes(), self.num_labels))
        cell_feat = None
        for nf in NeighborSampler(g=self.graph,
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=self.params.n_layers,
                                  neighbor_type='in',
                                  shuffle=False,
                                  num_workers=8,
                                  seed_nodes=self.test_ids.type(torch.int64),):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits, cellemb = self.model(nf)
                logits = logits.cpu()
                cellemb = cellemb.cpu()
            batch_nids = nf.layer_parent_nid(-1).type(torch.long)
            new_logits[batch_nids] = logits
            if cell_feat is None:
                cell_feat = cellemb
            else:
                cell_feat = torch.cat([cell_feat,cellemb],0)
        print('logits shape = '+str(new_logits.shape))
        new_logits = new_logits[self.test_ids.cpu().numpy()]
        new_logits = nn.functional.softmax(new_logits, dim=1).numpy()
        # save cell embeddings
        df_cellemb = pd.DataFrame(np.array(cell_feat))
        df_cellemb.index = self.pre_cells
        df_cellemb.to_csv(self.params.save_path + '/predict/cell_embedding.csv')

        predict_label = []
        for pred in new_logits:
            pred_label = self.id2label[pred.argmax().item()]
            pred_label = pred_label.replace('"','')
            predict_label.append(pred_label)

        if gold_file != '':
            cells = ''
            cells, celltype_true = self.load_gold(gold_file)
            self.save_pred(predict_label,cells)
            self.metrices(celltype_true, predict_label)
            print('Predict and evaluate done.')
            return predict_label
        else:
            self.save_weight(new_logits)
            self.save_pred(predict_label, list(self.pre_cells))
            print('Predict done.')
            return predict_label
