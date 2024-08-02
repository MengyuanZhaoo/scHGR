# scHGR
An automated annotation tool for single-cell transcriptome data, named single-cell hybrid graph neural network with gene regulations (scHGR).
## Overview
Single-cell transcriptome sequencing technologies allow us to deconvolve components of tissues or organs from single-cell resolution and deepen insight into cellular heterogeneity, reveal biodiversity as well as delineate molecular mechanisms. Assigning cell identities is a crucial step in processing single-cell RNA sequencing (scRNA-seq) data. Leveraging the continuously accumulated large-scale single-cell atlases as reference to annotate newly sequenced data is empowering for single-cell omics analysis. However, available tools typically rely on transcriptomic expression profiles, which are susceptible to technologies, platforms, and species, thus limiting their effectiveness in large-span annotation tasks.

Here we present a single-cell Hybrid graph neural network with Genomic Relations (scHGR), which extracts cross-scenario discernible expression patterns by combining genomic relationships with transcriptomic expression patterns. A total of $22$ scenarios involving various tissues, platforms, species, as well as diseases indicate that, scHGR stands out in both accuracy and stability benchmarked with state-of-the-art cell annotation tools. Crucially, scHGR uncovers novel biologically significant subtypes while assigning marker genes to facilitate gene-associated downstream tasks. In addition, scHGR provides exhaustive annotation for COVID-19 data with $56$ cell populations, revealing vital factors reflecting pathogenesis and inspiring therapeutic solutions.

![framework](https://user-images.githubusercontent.com/28176452/232417321-c4c0e4ee-f0e9-4fb7-a00e-7ab2dd4a702a.png)

## Requirements
python==3.7

dgl==0.4.3.post2

matplotlib==3.5.1

numpy==1.19.2

pandas==1.3.5

scikit_learn==1.2.2

scipy==1.6.2

torch==1.12.1

tqdm==4.64.0

## Usage
### Run the demo
All the original datasets can be downloaded ([PBMC-FACS](https://zenodo.org/record/3357167), [AMB](https://zenodo.org/record/3357167), [PBMC1-10X2](https://zenodo.org/record/3357167), [PBMC1-10X3](https://zenodo.org/record/3357167), [PBMC1-SM2](https://zenodo.org/record/3357167), [PBMC1-CS2](https://zenodo.org/record/3357167), [PBMC1-DS](https://zenodo.org/record/3357167), [PBMC1-D](https://zenodo.org/record/3357167), [PBMC1-SW](https://zenodo.org/record/3357167), [PBMC2-10X2](https://zenodo.org/record/3357167), [Human](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133), [Mouse](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133), [Diabetes mellitus](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE84133), [Human lung atlas](https://www.synapse.org/#!Synapse:syn21041850), [COVID-19 Mild](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926) and [COVID-19 Severe](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145926)).


For example, /datasets is equipped with the demo expression profiles and corresponding labels of PBMC1_10X2, PBMC1_10X3 and PBMC2_10X2 datasets. 
```Bash
cd scHGR
python pre-process.py  # generate gene_statistics.csv, regulations.txt and genenodes.npy
python scHGR.py        # train model and annotate query file
```
All output will be shown in /output. Performance will be shown at the bottom if the labels of query file are available. The directory of the output files and the specific instructions are as follows:
```
 |-- output
        |-- statistics
            |-- tissue_genes.txt                   # The list of genes involued in HGNN.
            |-- tissue_cell_type.txt               # The list of cell types involued in training process.
        |-- graphs
            |-- species_data.npz                   # The data provided by expression profiles.
        |-- model
            |-- species_tissue.pt                  # The optimal model after training.
            |-- train_record.pdf                   # Visualization of model training loss, training set accuracy, and validation set accuracy.
        |-- predict
            |-- species_tissue_predict.csv         # The annotation for each cell in query file.
            |-- cell_embedding.csv                 # The cell embeddings optimized by scHGR.
            |-- gene_embedding.csv                 # The gene embeddings optimized by scHGR.
            |-- gene_weight.csv                    # The propagation score of each gene.
            |-- evaluate.csv                       # Confusion matrix and statistical metrics of current annotation, if the labels of query dataset are provided.
```
You can also annotate the datasets with the existing scHGR model.
```Bash
python predict.py
```

### Input data
When using your own data, you have to provide
* the expression profile of reference datasets and cell labels
* the expression profile of query datasets

An optional input is the cell labels of query dataset and if this is provided, scHGR will automatically calculate the confusion matrix and various statistical metrics after annotation.
When using your own gene regulation data, you need to put them in the gene-regulatory-relation-repository folder and load them by pre-process.py.

### Output
The output files with scHGR predicted labels will be stored in the output folder.

