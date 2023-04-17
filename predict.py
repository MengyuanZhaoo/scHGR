import argparse
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.contrib.sampling import NeighborSampler
from utils import load_data
from models import GNN
from pprint import pprint
from sklearn import metrics
import csv
import os
import pdb


class Runner:
    def __init__(self, params):
        self.params = params
        self.postfix = time.strftime('%d_%m_%Y') + '_' + time.strftime('%H:%M:%S')
        self.device = torch.device('cpu' if self.params.gpu == -1 else f'cuda:{params.gpu}')
        self.pre_cells, self.num_cells, self.genes, self.num_genes, self.num_labels, self.graph, self.gene_ids, self.train_ids, self.val_ids, self.test_ids, self.labels, self.id2label = load_data(params)

        self.model = GNN(in_feats=self.params.dense_dim,
                         n_hidden=self.params.hidden_dim,
                         n_classes=self.num_labels,
                         n_layers=2,
                         gene_num=self.num_genes,
                         activation=F.relu,
                         dropout=self.params.dropout).to(self.device)
        self.load_model()
        self.num_neighbors = self.num_cells + self.num_genes
        self.model.to(self.device)

    def run(self):
        tic = time.time()
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
                                  seed_nodes=self.test_ids.type(torch.int64), ):
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
        print('logits shape = ' + str(new_logits.shape))
        new_logits = new_logits[self.test_ids.cpu().numpy()]
        new_logits = nn.functional.softmax(new_logits, dim=1).numpy()
        # save cell embeddings
        df_cellemb = pd.DataFrame(np.array(cell_feat))
        df_cellemb.index = self.pre_cells
        df_cellemb.to_csv(self.params.save_path + '/predict/cell_embedding.csv')
        print('Saving cell embedding done.')

        predict_label = []
        for pred in new_logits:
            pred_label = self.id2label[pred.argmax().item()]
            pred_label = pred_label.replace('"', '')
            predict_label.append(pred_label)

        if gold_file != '':
            cells = ''
            cells, celltype_true = self.load_gold(gold_file)
            assert np.array_equal(self.pre_cells,np.array(cells))
            # self.save_pred(predict_label, self.pre_cells)
            self.metrices(celltype_true, predict_label)
            print('Predict and evaluate done.')
        else:
            # self.save_pred(predict_label, self.pre_cells)
            print('Predict done.')

        # save gene embeddings
        new_logits = torch.zeros((self.graph.number_of_nodes(), self.num_labels))
        gene_feat = None
        for nf in NeighborSampler(g=self.graph,
                                  batch_size=self.params.batch_size,
                                  expand_factor=self.num_cells + self.num_genes,
                                  num_hops=self.params.n_layers,
                                  neighbor_type='in',
                                  shuffle=False,
                                  num_workers=8,
                                  seed_nodes=self.gene_ids.type(torch.int64), ):
            nf.copy_from_parent()  # Copy node/edge features from the parent graph.
            with torch.no_grad():
                logits, geneemb = self.model(nf)
                geneemb = geneemb.cpu()
            if gene_feat is None:
                gene_feat = geneemb
            else:
                gene_feat = torch.cat([gene_feat, geneemb], 0)
        df_geneemb = pd.DataFrame(np.array(gene_feat))
        df_geneemb.index = self.genes
        df_geneemb.to_csv(self.params.save_path + '/predict/gene_embedding.csv')
        print('Saving gene embedding done.')



    def load_model(self):
        model_path = Path(self.params.save_path) / 'model' / f'{self.params.species}-{self.params.tissue}.pt'
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state['model'])

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

    def save_pred(self, pred, cells=''):
        print('Saving predictions...')
        save_path = Path(self.params.save_path) / 'predict'
        if not save_path.exists():
            save_path.mkdir()
        if cells != '':
            df = pd.DataFrame({
                'cell': cells,
                'cell_type': pred})
        else:
            df = pd.DataFrame({'cell_type': pred})
        df.to_csv(
            save_path / (self.params.species + f"_{self.params.tissue}_predict.csv"),
            index=False)
        print(f"output has been stored in {self.params.species}_{self.params.tissue}_predict.csv")

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default='human', type=str)
    parser.add_argument("--tissue", default='PBMC', type=str)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id, -1 for cpu")
    parser.add_argument("--n_epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--dense_dim", type=int, default=400,
                        help="number of PCA units")
    parser.add_argument("--hidden_dim", type=int, default=200,
                        help="number of hidden gcn units")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="number of hidden gcn layers")
    parser.add_argument("--random_seed", type=int, default=10086)
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--threshold", type=float, default=0,
                        help="the threshold to connect edges between cells and genes")
    parser.add_argument("--num_neighbors", type=int, default=0,
                        help="number of neighbors to sample in message passing process. 0 means all neighbors")
    parser.add_argument("--exclude_rate", type=float, default=0.005,
                        help="exclude some cells less than this rate.")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--test_rate", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=50,
                        help="the window size of early stopping")

    # input and output setting 
    parser.add_argument("--files", type=list, default=[('datasets/pbmc2_10X_v2_data.csv',''),
                                                       ('datasets/pbmc1_10X_v2_data.csv','datasets/pbmc1_10X_v2_label.csv'),
                                                       ('datasets/pbmc1_10X_v3_data.csv','datasets/pbmc1_10X_v3_label.csv')],
                        help = "the file list of query expression profile, reference expression profile and reference label, note that the query file should be placed first.")
    parser.add_argument("--grn_file", type=str, default='statistics/regulations.txt',
                        help = "the gene regulatory network file")
    parser.add_argument("--emb_file", type=str, default='statistics/genenodes.npy',
                        help = "the embedding file of gene which is generated by pre-process.py")
    parser.add_argument("--order_file", type=str, default='statistics/gene_statistics.csv',
                        help = "the order file of genes sorted by dispersion descending order which is generated by pre-process.py")  
    parser.add_argument("--topk", type=int, default=5000,
                        help = "the number of genes filtered based on order_file, if topk < 0 then all genes in files are included")
    parser.add_argument("--reads", type=str, default='log',
                        help = "data type of scRNA-seq data, log (scHGR do not perform log-nomalization) or reads (scHGR perform log-nomalization)")
    parser.add_argument("--oritation", type=str, default='gc',
                        help = "the orientation of scRNA-seq data, gc (rows represent genes, columns represent cells) or cg (rows represent cells, columns represent genes)")
    parser.add_argument("--save_path", type=str, default='output/',
                        help = "the save path of models and outputs")
    params = parser.parse_args()

    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed(params.random_seed)
    pprint(vars(params))

    # annotating query file by existing scHGR model
    trainer = Runner(params)
    gold_file = 'datasets/pbmc2_10X_v2_label.csv'
    trainer.run(gold_file)

