#simple example with the output
my_list = ['evidence','evident','provide','unconventional','convene']
nTreeClusModel = nTreeClus(my_list, method = "All")

nTreeClusModel
# {'C_DT': 2,
# 'C_RF': 2,
# 'distance_DT': array([ 0.05508882,  0.43305329,  0.68551455,  0.43305329,  0.5       ,
#         0.7226499 ,  0.5       ,  0.86132495,  0.75      ,  0.4452998 ]),
# 'distance_RF': array([ 0.0809925 ,  0.56793679,  0.42158647,  0.57878823,  0.56917978,
#         0.47984351,  0.54545455,  0.55864167,  0.71278652,  0.3341997 ]),
# 'labels_DT': array([0, 0, 0, 1, 1]),
# 'labels_RF': array([0, 0, 0, 1, 1])}


#It is possible to have the dendogram view of the above example using this code:
# graphical output
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

HC_tree_terminal_cosine = linkage(nTreeClusModel["distance_DT"], 'ward')
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(HC_tree_terminal_cosine,labels=my_list, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()

HC_RF_terminal_cosine = linkage(nTreeClusModel["distance_RF"], 'ward')
fig = plt.figure(figsize=(25, 10))
ax = fig.add_subplot(1, 1, 1)
dendrogram(HC_RF_terminal_cosine,labels=my_list, ax=ax)
ax.tick_params(axis='x', which='major', labelsize=25)
ax.tick_params(axis='y', which='major', labelsize=15)
plt.show()

