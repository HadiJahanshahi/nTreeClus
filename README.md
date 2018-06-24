# nTreeClus
## nTreeClus: A NEW METHODOLOGY FOR CLUSTERING CATEGORICAL SEQUENTIAL DATA

by Mustafa Gökçe Baydoğan and Hadi Jahanshahi

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1295516.svg)](https://doi.org/10.5281/zenodo.1295516)






#### Maintained by: Hadi Jahanshahi (hadijahanshahi@gmail.com)
This is open source code repository for nTreeClus. nTreeClus, a model-based approach, is an intersection of the conceptions behind existing approaches, including Decision Tree Learning, k-mers, and autoregressive models for categorical time series, culminating with a novel numerical representation of the categorical sequences.

If using this code or dataset, please cite it.


## Quick validation of your code
Assume we are going to compare these words together: evidence, evident, provide, unconventional, and convene. Probably the first three of them having the same root "vid" should be in the first cluster and the last two words having the prefix "con" and the root "ven" should be clustered in another cluster. We test it using nTreeClus. At first, call pythonoic function of nTreeClus existing in the folder `Python implementation`, and then

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

As the output indicate, the number of optimal clusters `C_DT` and `C_RF` is 2 whether Decision Tree or Random Forest is used. Furthermore, the correct label for both methods is `[0, 0, 0, 1, 1]` indicating the first three are from cluster 0 and the last two items are from cluster 1. 

![alt text](https://media.gettyimages.com/photos/racegoers-arrive-on-the-second-day-of-the-royal-ascot-horse-racing-picture-id979343594)
