import pandas as pd
from encoder.models.sdne import SDNE
from encoder.models.basemodel import MultiClassifier
from sklearn.linear_model import LogisticRegression
import networkx as nx
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


def gen_GeneOrderFile(files, outdir): # file orientation is (gene, cell)
    # loading scRNA-seq data
    df_exp = pd.DataFrame()
    for file in files:
        if df_exp.empty:
            df_exp = pd.read_csv(file,index_col = 0).T
        else:
            new_df_exp = pd.read_csv(file,index_col = 0).T
            df_exp = pd.concat([df_exp,new_df_exp], axis=0)
    print('The scRNA-seq data includes '+str(df_exp.shape[0])+' cells and '+str(df_exp.shape[1])+' genes.')
    # calculating statistics
    gene_list = df_exp.columns.tolist()
    num_gene = len(gene_list)
    df_statistics = pd.DataFrame(columns={'gene', 'mean', 'variance', 'dispersion'})
    for gene in gene_list:
        print('Processing ' + str(gene_list.index(gene) + 1) + '/' + str(num_gene))
        new = pd.DataFrame(data={'gene': gene, 'mean': df_exp[gene].mean(0), 'variance': df_exp[gene].var(0),
                                 'dispersion': df_exp[gene].var(0) / df_exp[gene].mean(0)}, index=[0])
        df_statistics = pd.concat([df_statistics,new], axis=0)
    # Order by dispersion
    df_statistics = df_statistics.sort_values(by=['dispersion'], ascending=False)
    df_statistics.rename(columns={'dispersion': 'dispersion=variance/mean'}, inplace=True)
    df_statistics = df_statistics[['gene', 'mean', 'variance', 'dispersion=variance/mean']]
    df_statistics.to_csv(outdir + 'gene_statistics.csv',index = False)
    print('Successfully generate '+ outdir + 'gene_statistics.csv.')


def load_gene_file(file,k=0): # k is the number of filter genes
    with open(file, 'r') as f_gene:
        f_gene.readline()
        genes = []
        for line in f_gene:
            [gene, _,_,_] = line.strip().split(',')
            gene = gene.replace('"','')
            genes.append(gene)
    if k >0:
        genes = genes[0:k]
    print('Successfully load ' + str(len(genes)) + ' genes from ' + file)
    return set(genes)


def load_network_trrust(file,need_genes):
    with open(file, 'r') as f_gene:
        f_gene.readline()
        genes = []
        regulations = []
        for line in f_gene:
            [gene1, gene2] = line.strip().split('\t')[0:2]
            gene1.replace('"','')
            gene2.replace('"', '')
            if (gene1 in need_genes) and (gene2 in need_genes):
                genes.append(gene1)
                genes.append(gene2)
                regulations.append(gene1+','+gene2)
        genes = set(genes)
        regulations = set(regulations)
    print('Successfully load '+str(len(genes))+' genes and '+str(len(regulations))+' regulations from '+file)
    return genes,regulations


def load_network_reg(file,need_genes):
    with open(file, 'r') as f_gene:
        f_gene.readline()
        genes = []
        regulations = []
        for line in f_gene:
            [gene1, _, gene2] = line.strip().split(',')[0:3]
            gene1 = gene1.replace('"', '')
            gene2 = gene2.replace('"', '')
            if (gene1 in need_genes) and (gene2 in need_genes):
                genes.append(gene1)
                genes.append(gene2)
                regulations.append(gene1+','+gene2)
        genes = set(genes)
        regulations = set(regulations)
    print('Successfully load ' + str(len(genes)) + ' genes and ' + str(len(regulations)) + ' regulations from ' + file)
    return genes, regulations


def load_network_biogrid(file,need_genes):
    with open(file, 'r') as f_gene:
        for i in range(0,36): # some illustration of file
            f_gene.readline()
        genes = []
        regulations = []
        for line in f_gene:
            temp = line.strip().split('\t')
            offcial1 = temp[2].replace('"', '')
            offcial2 = temp[3].replace('"', '')
            symbol1 = temp[4].replace('"', '')
            symbol2 = temp[5].replace('"', '')
            if offcial1 in need_genes:
                if offcial2 in need_genes:
                    genes.append(offcial1)
                    genes.append(offcial2)
                    regulations.append(offcial1+','+offcial2)
                else:
                    for name2 in symbol2.split('|'):
                        if name2 in need_genes:
                            genes.append(offcial1)
                            genes.append(name2)
                            regulations.append(offcial1 + ',' + name2)
            elif offcial2 in need_genes:
                for name1 in symbol1.split('|'):
                    if name1 in need_genes:
                        genes.append(name1)
                        genes.append(offcial2)
                        regulations.append(name1 + ',' + offcial2)
                        
        genes = set(genes)
        regulations = set(regulations)
    print('Successfully load ' + str(len(genes)) + ' genes and ' + str(len(regulations)) + ' regulations from ' + file)
    return genes, regulations


def load_network_gredb(file,need_genes):
    with open(file, 'r') as f_gene:
        f_gene.readline()
        genes = []
        regulations = []
        for line in f_gene:
            temp = line.strip().split('\t')
            gene1 = temp[1].replace('"', '')
            gene2 = temp[3].replace('"', '')
            if (gene1 in need_genes) and (gene2 in need_genes):
                genes.append(gene1)
                genes.append(gene2)
                regulations.append(gene1+','+gene2)
        genes = set(genes)
        regulations = set(regulations)
    print('Successfully load ' + str(len(genes)) + ' genes and ' + str(len(regulations)) + ' regulations from ' + file)
    return genes, regulations


def load_network_chip(file,need_genes):
    lower_genes = [gene.lower() for gene in need_genes] #convert need_genes to lowercase
    temp_genes = list(need_genes)
    with open(file, 'r') as f_gene:
        f_gene.readline()
        genes = []
        regulations = []
        row=0
        for line in f_gene:
            if row%10000 == 0:
                print('Loading row '+str(row))
            row += 1
            temp = line.strip().split(',')
            gene1 = temp[0].replace('"', '').lower()
            gene2 = temp[1].replace('"', '').lower()
            if (gene1 in lower_genes) and (gene2 in lower_genes):
                gene1_idx = lower_genes.index(gene1)
                gene2_idx = lower_genes.index(gene2)
                g1 = temp_genes[gene1_idx]
                g2 = temp_genes[gene2_idx]
                genes.append(g1)
                genes.append(g2)
                regulations.append(g1+','+g2)

        genes = set(genes)
        regulations = set(regulations)
    print('Successfully load ' + str(len(genes)) + ' genes and ' + str(len(regulations)) + ' regulations from ' + file)
    return genes, regulations


def save_regs(file, regs):
    with open(file,'w') as f:
        for reg in regs:
            f.write(reg)
            f.write('\n')



# 1. generating gene_statistics.csv
# files= ['datasets/pbmc1_10X_v2_data.csv', 'datasets/pbmc1_10X_v2_data.csv', 'datasets/pbmc1_10X_v3_data.csv']
# order_save_path = 'datasets/'
# gen_GeneOrderFile(files, order_save_path)

# 2. selecting gene regulations related to genes involved in files
# k=5000
# order_file='statistics/gene_statistics.csv'
# trrust_file='gene-regulatory-relation-repository/human/TRRUST.tsv'
# reg_file='gene-regulatory-relation-repository/human/RegNetwork-Experimental.csv'
# biogrid_file='gene-regulatory-relation-repository/human/BIOGRID.txt'
# gredb_file = 'gene-regulatory-relation-repository/human/GREDB.txt'

# genes = load_gene_file(order_file,k)
# trrust_genes, trrust_regulations = load_network_trrust(trrust_file,genes)
# reg_genes, reg_regulations = load_network_reg(reg_file,genes)
# bio_genes, bio_regulations = load_network_biogrid(biogrid_file,genes)
# gre_genes, gre_regulations = load_network_gredb(gredb_file,genes)

# all_regs = set(trrust_regulations.union(bio_regulations,reg_regulations,gre_regulations))
# save_regs('statistics/regulations.txt',all_regs)
# print('Totally '+str(len(all_regs))+' available regulations.')

# 3. encoding regulations by Gene Encoder
G = nx.read_edgelist('statistics/regulations.txt',
                        delimiter=',',create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])

model = SDNE(G, hidden_layers=[800, 400])
model.fit(batch_size=1024, epochs=100)
embeddings = model.get_embeddings()
np.save('statistics/genenodes.npy', embeddings)
print('Succesfullly generate statistics/genenodes.npy.')

