# nTreeClus
## nTreeClus: A NEW METHODOLOGY FOR CLUSTERING CATEGORICAL SEQUENTIAL DATA

by Mustafa Gökçe Baydoğan and Hadi Jahanshahi

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1295516.svg)](https://doi.org/10.5281/zenodo.1295516)






#### Maintained by: Hadi Jahanshahi (hadijahanshahi@gmail.com)
This is open source code repository for nTreeClus. nTreeClus is an intersection of the conceptions behind existing approaches, including Decision Tree Learning, k-mers, and autoregressive models for categorical time series, culminating with a novel numerical representation of the categorical sequences.

If using this code or dataset, please cite it.


## Quick validation of your code
Assume we are going to compare these words together: `evidence`, `evident`, `provide`, `unconventional`, and `convene`. Probably the first three having the same root `vid` (see) should be in the same cluster and the last two words having the prefix `con` (together)  and the root `ven` (come) should be clustered as another cluster. We test it using nTreeClus. 

**NOTE**: Before using the below code, you shoud call Pythonic function of nTreeClus existing in the folder `Python implementation`.

```
my_list = ['evidence','evident','provide','unconventional','convene']
nTreeClusModel = nTreeClus(my_list, method = "All")
nTreeClusModel
```
**Result**
```
# {'C_DT': 2,
# 'C_RF': 2,
# 'distance_DT': array([ 0.05508882,  0.43305329,  0.68551455,  0.43305329,  0.5       ,
#         0.7226499 ,  0.5       ,  0.86132495,  0.75      ,  0.4452998 ]),
# 'distance_RF': array([ 0.0809925 ,  0.56793679,  0.42158647,  0.57878823,  0.56917978,
#         0.47984351,  0.54545455,  0.55864167,  0.71278652,  0.3341997 ]),
# 'labels_DT': array([0, 0, 0, 1, 1]),
# 'labels_RF': array([0, 0, 0, 1, 1])}
```

As the output indicates, the number of optimal clusters `C_DT` and `C_RF` is 2 whether Decision Tree or Random Forest is used. Furthermore, the correct label for both methods is `[0, 0, 0, 1, 1]` indicating the first three are from cluster 0 and the last two items are from cluster 1. The result is consistent with our pre-knowledge about the data.


**Graphical Output**

The graphical dendogram of the example clearly shows how the words are related and which ones have the closest similarity.  

```
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

HC_tree_terminal_cosine = linkage(nTreeClusModel["distance_DT"], 'ward')
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(HC_tree_terminal_cosine,labels=my_list, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()
```
![nTreeClus simple example using DT support](https://image.ibb.co/gPaZs8/n_Tree_Clus_HC_DT.png)


```
HC_RF_terminal_cosine = linkage(nTreeClusModel["distance_RF"], 'ward')
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(HC_RF_terminal_cosine,labels=my_list, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()
```
![nTreeClus simple example using RF support](https://image.ibb.co/nQQsC8/n_Tree_Clus_HC_RF.png)


As the figures demonstrate, the output of RF and DT implementation of nTreeClus is very similar and the method is robust to the selected approach.

**Distance Matrix**

In order to have distance matrix in a square-form, the below code and scipy library (`scipy.spatial.distance.squareform`) should be used: 

```
from scipy.spatial.distance import squareform
squareform(nTreeClusModel["distance_DT"])
```
*output*
```
#array([[ 0.        ,  0.05508882,  0.43305329,  0.68551455,  0.43305329],
#       [ 0.05508882,  0.        ,  0.5       ,  0.7226499 ,  0.5       ],
#       [ 0.43305329,  0.5       ,  0.        ,  0.86132495,  0.75      ],
#       [ 0.68551455,  0.7226499 ,  0.86132495,  0.        ,  0.4452998 ],
#       [ 0.43305329,  0.5       ,  0.75      ,  0.4452998 ,  0.        ]])
```
