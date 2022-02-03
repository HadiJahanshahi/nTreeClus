# Clustering Nucleotide Sequences
We examine the performance of the nTreeClus on Nucleotide sequences extracted from 
GENCODE human annotation version 39 [https://www.gencodegenes.org/human/release_39.html].
We download the transcript sequences FASTA file containing `Nucleotide sequences of all transcripts on the reference chromosomes`. 
We filter the dataset on the top-10 most frequent gene ids in the dataset, leaving us with 1,728 sequences whose lengths vary from 144 to 24,124. 
The filtered dataset can be downloaded here as `gencode.v39.csv.gz`.

# Information on the dataset
The below table shows the most fequent gene ids existing in Nucleotide sequence database. The number of sequences in each gene id and some statistics on the length of sequences are provided.

| gene-id	|n of sequences|	mean length|	median length|	min length|	max length|
|:---| ---:   | ---:    | ---:   | ---: | ---: |
| ENSG00000179818.16|	283|	1396|	1041|	208|	4755|
| ENSG00000226674.11|	189|	1021|	944	| 144|	4635|
| ENSG00000241469.9|	164|	1443|	1467|	561|	3176|
| ENSG00000242086.8|	142|	757	| 706	| 229|	4208|
| ENSG00000109339.24|	192|	2489|	2466|	414|	8941|
| ENSG00000249859.13|	183|	1220|	1156|	408|	2450|
| ENSG00000229140.11|	142|	1521|	1461|	374|	20097|
| ENSG00000224078.15|	142|	5280|	5185|	470|	24124|
| ENSG00000227195.11|	146|	1093|	1138|	590|	1504|
| ENSG00000215386.14|	145|	1348|	1124|	426| 5889|


# The result of running nTreeClus on the dataset

The below table shows the performance of different versions of nTreeClus on the filtered dataset. 
Our proposed algorithm is capable of handling the clustering task of sequences of different lengths in a short time.

|Methods | Purity   | RI    | ARI   | F-meas | ASW | 1NN |
|:---| ---:   | ---:    | ---:   | ---: | ---: | ---: |
|nTreeClus (DT)  | 0.852 | 0.936 | 0.690 | 0.859 | 0.491 | 0.999 |
|nTreeClus (RF)  | 0.855 | 0.938 | 0.696 | 0.860 | 0.498 | 0.999 |
|nTreeClus (DT)* | 0.867 | 0.945 | 0.724 | 0.872 | 0.491 | 0.999 |
|nTreeClus (RF)* | 0.865 | 0.943 | 0.718 | 0.870 | 0.491 | 0.999 |
