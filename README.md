# nTreeClus V. 1.2.1.
## nTreeClus: a Tree-based Sequence Encoder for Clustering Categorical Series	

by Hadi Jahanshahi and Mustafa Gökçe Baydoğan 

Published in Neurocomputing Journal

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5942533.svg)](https://doi.org/10.5281/zenodo.5942533)




#### Maintained by: Hadi Jahanshahi (hadijahanshahi@gmail.com)
This is an open-source code repository for [nTreeClus](https://arxiv.org/abs/2102.10252). nTreeClus is an intersection of the conceptions behind existing approaches, including Decision Tree Learning, k-mers, and autoregressive models for categorical time series, culminating with a novel numerical representation of the categorical sequences.

If using this code or dataset, please cite it.


## Prerequisite libraries
* Python 3.6.0+
* matplotlib 3.3.2
* nltk 3.6.2
* numpy 1.19.2
* pandas 1.1.2
* sklearn 0.24.1
* scipy 1.6.1
* tqdm 4.40.0


## Explaining the nTreeClus function
```{python}
nTreeClus(sequences, n, method = "All", ntree = 10, C= None, verbose=1)
```
### Inputs
* **sequences**: a list of sequences to be clustered
* **n**: "the window length" or "n" in nTreeclus. You may provide it or it will be calculated automatically if no input has been suggested. Currently, the default value of "the square root of average sequences' lengths" is taken.
* **method**: 

    **DT**:          Decision Tree

    **DT_position**: Decision Tree with position indexes
    
    **RF**:          Random Forest

    **RF_position**: Random Forest with position indexes
    
    **All**: all methods (default)
* **ntree**: the number of trees to be used in the RF method. The default value is 10. (Setting a small value decreases accuracy, and a large value may increase the complexity. No less than 5 and no greater than 20 is recommended.)
* **C**: the number of clusters. If it is not provided, it will be calculated for `C` between 2 to 10.
* **verbose [binary]**: It indicates whether to print the outputs or not. 

### Outputs
* **C_DT**: "the optimal number of clusters for Decision Tree",
* **C_RF**: "the optimal number of clusters for Random Forest",
* **Parameter n**: the parameter of the nTreeClus (n) -- either calculated or manually entered
* **distance_DT**: "sparse distance between sequences for Decision Tree",
* **distance_RF**: "sparse distance between sequences for Random Forest",
* **labels_DT**: "labels based on the optimal number of clusters for DT",
* **labels_RF**: "labels based on the optimal number of clusters for RF",
* **running_timeSegmentation**: Running time of the matrix segmentation in seconds,
* **running_timeDT**: Running time of Decision-Tree version of nTreeClus in seconds,
* **'running_timeDT_p**: Running time of Decision-Tree and position-sensitive version of nTreeClus in seconds,
* **'running_timeRF**: Running time of Random-Forest version of nTreeClus in seconds,
* **'running_timeRF_p**: Running time of Random-Forest Tree and position-sensitive version of nTreeClus in seconds,
* **res**: "The result table for the performance of the algorithm. It will be shown if the method `performance` is run.",


### Dendrogram Plotting
`plot` method plots the dendrogram of the result. Use it only if the dataset is small.

### Performance Calculation
`performance` method reports the model performance in terms of F1-Score, Purity, Rand Index, Adjusted Rand Index, Average Sillouhete Width, and 1NN algorithm. It requires the correct labels to calculate the values for internal metrics.

## Quick validation of the code
Assume we are going to compare these words together: `evidence`, `evident`, `provide`, `unconventional`, and `convene`. We expect to see the first three words in the same cluster, knowing they have the same root `vid` (see). The last two words, having the prefix `con` (together) and the root `ven` (come), are expected to be clustered separately. We test it using nTreeClus. 

**NOTE**: Before using the below code, you should import the nTreeClus class from the folder `Python implementation`.

```{python}
sequences = ['evidence','evident','provide','unconventional','convene']
model     = nTreeClus.nTreeClus(my_list, n=None, ntree=10, method="All", verbose=0)
model.nTreeClus()
print(model.output())
```
**Result**
``` {python}
# {'C_DT': 2,
#  'distance_DT': array([0.05508882, 0.43305329, 0.68551455, 0.43305329, 0.5       ,
#                        0.7226499 , 0.5       , 0.86132495, 0.75      , 0.4452998 ]),
#  'labels_DT': array([0, 0, 0, 1, 1]),
#  'C_RF': 3,
#  'distance_RF': array([0.10101553, 0.54006689, 0.59462876, 0.64917679, 0.52326871,
#                        0.69187045, 0.675     , 0.69491039, 0.80930748, 0.27708067]),
#  'labels_RF': array([0, 0, 1, 2, 2]),
#  'C_DT_p': 2,
#  'distance_DT_p': array([0.05508882, 0.43305329, 0.68551455, 0.22848325, 0.5       ,
#                          0.7226499 , 0.38762756, 0.86132495, 0.59175171, 0.54708919]),
#  'labels_DT_p': array([0, 0, 0, 1, 0]),
#  'C_RF_p': 3,
#  'distance_RF_p': array([0.08526374, 0.59128805, 0.63301208, 0.59175171, 0.59090909,
#                          0.68930157, 0.61861496, 0.73215652, 0.76163435, 0.47187677]),
#  'labels_RF_p': array([0, 0, 1, 2, 2]),
#  'running_timeSegmentation': 0,
#  'running_timeDT': 0,
#  'running_timeDT_p': 0,
#  'running_timeRF': 0,
#  'running_timeRF_p': 0,
#  'Parameter n': 4}
```

As the output indicates, the number of optimal clusters `C_DT` and `C_RF` is 2 whether Decision Tree or Random Forest is used. Furthermore, the correct label for both methods is `[0, 0, 0, 1, 1]` indicating the first three are from cluster 0, and the last two items are from cluster 1. The result is consistent with our pre-knowledge about the data. The parameter `n` or windows size is calculated and is 4 in this case.


**Graphical Output**

The graphical dendrogram of the example shows how the words are related and which ones have the closest similarity.  

``` {python}
from matplotlib import pyplot as plt
fig, ax = model.plot('DT', my_list, linkage_method= 'ward', rotation=0)
plt.show()
```
![nTreeClus simple example using DT support](https://i.ibb.co/njcCj00/n-Tree-Clus-DT.png)


``` {python}
from matplotlib import pyplot as plt
fig, ax = model.plot('RF_position', my_list, linkage_method= 'ward', rotation=0)
plt.show()
```
![nTreeClus simple example using RF support](https://i.ibb.co/HBrv56Z/n-Tree-Clus-RF.png)


As the figures demonstrate, the output of RF and DT implementation of nTreeClus is very similar, and the method is robust to method selection.

**Distance Matrix**

In order to have a distance matrix in a square-form, the below code of the scipy library (`scipy.spatial.distance.squareform`) should be used: 

``` {python}
from scipy.spatial.distance import squareform
squareform(model.output()["distance_DT"])
```
*output*
``` {python}
# array([[ 0.        ,  0.05508882,  0.43305329,  0.68551455,  0.43305329],
#        [ 0.05508882,  0.        ,  0.5       ,  0.7226499 ,  0.5       ],
#        [ 0.43305329,  0.5       ,  0.        ,  0.86132495,  0.75      ],
#        [ 0.68551455,  0.7226499 ,  0.86132495,  0.        ,  0.4452998 ],
#        [ 0.43305329,  0.5       ,  0.75      ,  0.4452998 ,  0.        ]])
```

# Citation
Please cite our work as below. You can have a free access to the paper on Arxiv website:
https://arxiv.org/abs/2102.10252

```
@article{jahanshahi2022ntreeclus,
   title = {nTreeClus: a Tree-based Sequence Encoder for Clustering Categorical Series},
   journal = {Neurocomputing},
   year = {2022},
   issn = {0925-2312},
   doi = {https://doi.org/10.1016/j.neucom.2022.04.076},
   url = {https://www.sciencedirect.com/science/article/pii/S0925231222004611},
   author = {Hadi Jahanshahi and Mustafa Gokce Baydogan},
   keywords = {sequence mining, categorical time series, model-based clustering, pattern recognition, tree-based learning},
   abstract = {The overwhelming presence of categorical/sequential data in diverse domains emphasizes the importance of sequence mining. The challenging nature of   sequences proves the need for continuing research to find a more accurate and faster approach providing a better understanding of their (dis)similarities. This paper proposes a new Model-based approach for clustering sequence data, namely nTreeClus. The proposed method deploys Tree-based Learners, k-mers, and autoregressive models for categorical time series, culminating with a novel numerical representation of the categorical sequences. Adopting this new representation, we cluster sequences, considering the inherent patterns in categorical time series. Accordingly, the model showed robustness to its parameter. Under different simulated scenarios, nTreeClus improved the baseline methods for various internal and external cluster validation metrics for up to 10.7% and 2.7%, respectively. The empirical evaluation using synthetic and real datasets, protein sequences, and categorical time series showed that nTreeClus is competitive or superior to most state-of-the-art algorithms.}
}
```


