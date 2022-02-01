# Clustering Nucleotide Sequences
We examine the performance of the nTreeClus on Nucleotide sequences extracted from 
GENCODE human annotation version 39 [https://www.gencodegenes.org/human/release_39.html].
We download the transcript sequences FASTA file containing `Nucleotide sequences of all transcripts on the reference chromosomes`. 
We filter the dataset on the top-10 most frequent gene ids in the dataset, leaving us with 1,728 sequences whose lengths vary from 144 to 24,124. 
The filtered dataset can be downloaded here as `gencode.v39.csv.gz`.

# The result of running nTreeClus on the dataset

The below table shows the performance of different versions of nTreeClus on the filtered dataset. 
Our proposed algorithm is capable of handling the clustering task of sequences of different lengths in a short time.

|Methods | Purity   | RI    | ARI   | F-meas | ASW | 1NN |
|:---| ---:   | ---:    | ---:   | ---: | ---: | ---: |
|nTreeClus (DT)  | 0.852 | 0.936 | 0.690 | 0.859 | 0.491 | 0.999 |
|nTreeClus (RF)  | 0.855 | 0.938 | 0.696 | 0.860 | 0.498 | 0.999 |
|nTreeClus (DT)* | 0.867 | 0.945 | 0.724 | 0.872 | 0.491 | 0.999 |
|nTreeClus (RF)* | 0.865 | 0.943 | 0.718 | 0.870 | 0.491 | 0.999 |
