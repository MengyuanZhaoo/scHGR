import argparse
import pandas as pd
import dgl
import torch
import torch.nn.functional as F
import collections
from scipy.sparse import csr_matrix, vstack, save_npz
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import csv
from pprint import pprint
import json
import pdb

#---------------------------------------------train-----------------------------------#
def load_gene_emb(file):
    temp = np.load(file, allow_pickle=True)
    gene_emb = temp.item()
    return gene_emb
def load_grn_edges(file,id2gene=[]):
    tfs=[]
    targets=[]
    with open(file,'r') as f:
        if len(id2gene) > 0: #only genes overlap with id2gene
            print('Loading GRN genes with overlap.')
            for line in f.readlines():
                [gene1,gene2] = line.strip().split(',')
                if (gene1 in id2gene) and (gene2 in id2gene):
                    tfs.append(gene1)
                    targets.append(gene2)
        else: #all genes in file
            print('Loading GRN genes without overlap.')
            for line in f.readlines():
                [gene1,gene2] = line.strip().split(',')
                tfs.append(gene1)
                targets.append(gene2)
    return tfs, targets

def z_nomalize(data):
    scaler = StandardScaler().fit(data)
    z_data = scaler.transform(data)
    return z_data

def gene_select(file,k):# select top k genes from file
    df_gene = pd.read_csv(file)
    sel_gene=df_gene['gene'][0:k]
    return sel_gene

def normalize_weight(graph: dgl.DGLGraph):
    # normalize weight & add self-loop
    in_degrees = graph.in_degrees()
    for i in range(graph.number_of_nodes()):
        src, dst, in_edge_id = graph.in_edges(i, form='all')
        if src.shape[0] == 0:
            continue
        edge_w = graph.edata['weight'][in_edge_id]
        graph.edata['weight'][in_edge_id] = in_degrees[i] * edge_w / torch.sum(edge_w)

def get_id_2_gene(files, order_file, topk):
    data_files = files
    genes = None
    if order_file == '' or topk < 0:
        for file in data_files:
            data = pd.read_csv(file[0], header=0).values[:, 0]
            if genes is None:
                genes = set(data)
            else:
                genes = genes | set(data)
    else:
        genes = gene_select(order_file, topk)
        print('Select '+str(len(genes))+' genes based on gene order file.')
    id2gene = list(genes)
    id2gene.sort()
    for i in range(len(id2gene)):
        id2gene[i] = id2gene[i].replace('"','')
    return id2gene

def get_id_2_label_and_label_statistics(files):
    cell_files = files
    cell_types = set()
    cell_type_list = list()
    for file in cell_files:
        if file[1] !='':
            df = pd.read_csv(file[1], dtype=np.str, header=0)
            df['Cell_type'] = df['Cell_type'].map(str.strip)
            cell_types = set(df.values[:, df.shape[1]-1]) | cell_types
            cell_type_list.extend(df.values[:, df.shape[1]-1].tolist())
    id2label = list(cell_types)
    id2label.sort()
    for i in range(len(id2label)):
        id2label[i] = id2label[i].replace('"', '')
    for i in range(len(cell_type_list)):
        cell_type_list[i] = cell_type_list[i].replace('"', '')
    label_statistics = dict(collections.Counter(cell_type_list))
    return id2label, label_statistics


def save_statistics(statistics_path, id2label, id2gene, tissue):
    gene_path = statistics_path / f'{tissue}_genes.txt'
    label_path = statistics_path / f'{tissue}_cell_type.txt'
    with open(gene_path, 'w', encoding='utf-8',newline='') as f:
        for gene in id2gene:
            f.write(gene + '\r\n')
    with open(label_path, 'w', encoding='utf-8',newline='') as f:
        for label in id2label:
            f.write(label + '\r\n')


def load_data_integrate(params):
    random_seed = params.random_seed
    dense_dim = params.dense_dim
    tissue = params.tissue
    device = torch.device('cpu' if params.gpu == -1 else f'cuda:{params.gpu}')

    graph_path = Path(params.save_path+'/graphs')
    statistics_path = Path(params.save_path+'/statistics')
    if not statistics_path.exists():
        statistics_path.mkdir(parents=True)
    if not graph_path.exists():
        graph_path.mkdir(parents=True)

    # generate gene statistics file
    id2gene = get_id_2_gene(params.files, params.order_file, params.topk)

    # generate cell label statistics file
    id2label, label_statistics = get_id_2_label_and_label_statistics(params.files)
    assert len(id2label) > 0
    gene2id = {gene: idx for idx, gene in enumerate(id2gene)}
    num_genes = len(id2gene)
    # prepare unified labels
    num_labels = len(id2label)
    label2id = {label: idx for idx, label in enumerate(id2label)}
    save_statistics(statistics_path, id2label, id2gene,tissue)
    print(f"The build graph contains {num_genes} genes with {num_labels} labels supported.")

    graph = dgl.DGLGraph()

    gene_ids = torch.arange(num_genes, dtype=torch.int32, device=device).unsqueeze(-1)
    graph.add_nodes(num_genes, {'id': gene_ids})

    all_labels = []
    matrices = []
    num_cells = 0 # all cell nodes in graph
    num_test_cells = 0 # cell nodes in test set
    test_ids = []
    pre_cells = []

    for (file_data, file_label) in params.files:
        file_data_path = Path(file_data)
        file_label_path = Path(file_label)

        # load data file then update graph
        df = pd.read_csv(file_data_path, index_col=0)  # with ""
        if params.oritation == 'gc':
            df = df.transpose(copy=True)  # (cell, gene)
        # load label file to update all_labels
        if file_label != '':
            # load training celltype file then update labels accordingly
            cell2type = pd.read_csv(file_label_path, index_col=0)
            cell2type.columns = ['cell', 'type']
            cell2type['type'] = cell2type['type'].map(str.strip)
            for i in range(len(cell2type['type'])):
                cell2type['type'][i] = cell2type['type'][i].replace('"', '')
            cell2type['id'] = cell2type['type'].map(label2id)
            # filter out cells not in label-text
            filter_cell = np.where(pd.isnull(cell2type['id']) == False)[0]
            cell2type = cell2type.iloc[filter_cell]
            assert not cell2type['id'].isnull().any(), 'something wrong about celltype file.'
            all_labels += cell2type['id'].tolist()
        else:
            mask = [-2]*df.shape[0]
            all_labels += mask
            cell2type = pd.DataFrame({'cell':mask})
            num_test_cells = df.shape[0]
            pre_cells = np.array(df.index)

        assert len(cell2type['cell']) == len(df.index)

        print(str(len(df.columns)) + ' genes before filtering.')
        df.columns = [col.replace('"', '') for col in df.columns]
        df = df.rename(columns=gene2id)
        # filter out useless columns if exists (when using gene intersection)
        col = [c for c in df.columns if c in gene2id.values()]
        df = df[col]
        print(str(len(df.columns))+' genes left after filtering based on order file.')
        print(f"{file_data} -> Nonzero Ratio: {df.fillna(0).astype(bool).sum().sum() / df.size * 100:.2f}%")

        # maintain inter-datasets index for graph and RNA-seq values
        arr = df.to_numpy()
        # do normalization
        if params.reads == 'reads':
            arr = np.log2(arr + 1)
        row_idx, col_idx = np.nonzero(arr > params.threshold)  # intra-dataset index
        non_zeros = arr[(row_idx, col_idx)]  # non-zero values
        cell_idx = row_idx + graph.number_of_nodes()  # cell_index
        gene_idx = df.columns[col_idx].astype(int).tolist()  # gene_index
        info_shape = (len(df), num_genes)
        info = csr_matrix((non_zeros, (row_idx, gene_idx)), shape=info_shape)
        matrices.append(info)
        num_cells += len(df)
        ids = torch.tensor([-1] * len(df), dtype=torch.int32, device=device).unsqueeze(-1)
        graph.add_nodes(len(df), {'id': ids})
        graph.add_edges(cell_idx, gene_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
        graph.add_edges(gene_idx, cell_idx,
                        {'weight': torch.tensor(non_zeros, dtype=torch.float32, device=device).unsqueeze(1)})
        assert graph.number_of_edges() > 0
        print(f'#Nodes in Graph: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
    # add GRN and update graph
    if params.grn_file != '':
        tfs, targets = load_grn_edges(params.grn_file, id2gene)
        tf_idx = [gene2id[tf] for tf in tfs]
        target_idx = [gene2id[tar] for tar in targets]
        graph.add_edges(tf_idx, target_idx,
                        {'weight': torch.tensor(np.full(len(tf_idx), np.max(non_zeros)), dtype=torch.float32,
                                                device=device).unsqueeze(1)})
        print(f'Added {len(tf_idx)} edges from GRN.')
        print(f'#Nodes in Graph: {graph.number_of_nodes()}, #Edges: {graph.number_of_edges()}.')
        assert len(all_labels) == num_cells
    save_npz(graph_path / f'{params.species}_data', vstack(matrices))

    # 2. create features
    sparse_feat = vstack(matrices).toarray()  # cell-wise  (cell, gene)
    assert sparse_feat.shape[0] == num_cells

    # transpose to gene-wise
    gene_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat.T)
    gene_feat = gene_pca.transform(sparse_feat.T)
    gene_evr = sum(gene_pca.explained_variance_ratio_) * 100   #sum of variance in each component after PCA
    print(f'[PCA] Gene EVR: {gene_evr:.2f} %.')
    # load gene embedding from GRN
    if params.emb_file != '':
        gene_dict = load_gene_emb(params.emb_file)
        record = 0
        for gene in gene_dict.keys():
            if gene in id2gene:
                id = gene2id[gene]
                if id in df.columns:
                    index = df.columns[id]
                    gene_feat[index] = gene_dict[gene]
                    record+=1
        print(str(record)+'/'+str(gene_feat.shape[0])+' gene feature from '+ params.emb_file)
    print('Gene feature shape:' + str(gene_feat.shape))

    print('------Train label statistics------')
    for i, label in enumerate(id2label, start=1):
        print(f"#{i} [{label}]: {label_statistics[label]}")

    # use PCA to form cell_feat
    cell_pca = PCA(dense_dim, random_state=random_seed).fit(sparse_feat)
    cell_feat = cell_pca.transform(sparse_feat)



    gene_feat = torch.from_numpy(gene_feat)  # use shared storage
    cell_feat = torch.from_numpy(cell_feat)

    graph.ndata['features'] = torch.cat([gene_feat, cell_feat], dim=0).type(torch.float).to(device)
    labels = torch.tensor([-1] * num_genes + all_labels, dtype=torch.long, device=device)  # [gene_num+test_num+train_num]

    # split train set and test set
    per = np.random.permutation(range(num_genes+num_test_cells, num_genes + num_cells))# array order random
    val_ids = torch.tensor(per[:int(len(per) * params.test_rate)]).to(device)
    train_ids = torch.tensor(per[int(len(per) * params.test_rate):]).to(device)
    test_ids = torch.tensor(np.arange(num_genes, num_genes + num_test_cells, 1)).to(device)
    # check ids without overlap
    assert len(np.intersect1d(val_ids.cpu(), train_ids.cpu())) == 0
    assert len(np.intersect1d(val_ids.cpu(), test_ids.cpu())) == 0
    assert len(np.intersect1d(train_ids.cpu(), test_ids.cpu())) == 0
    # normalize weight
    normalize_weight(graph)
    # add self-loop
    graph.add_edges(graph.nodes(), graph.nodes(),
                    {'weight': torch.ones(graph.number_of_nodes(), dtype=torch.float, device=device).unsqueeze(1)})
    graph.readonly()
    return num_cells, pre_cells, num_genes, np.array(id2gene), num_labels, graph, train_ids, val_ids, test_ids, labels, id2label


