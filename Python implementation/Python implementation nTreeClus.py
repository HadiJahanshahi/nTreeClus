def nTreeClus(my_list, n = None, method = "All", ntree = 10, C = None):
    """ nTreeClus is a clustering method by Mustafa Gokce Baydogan and Hadi Jahanshahi.
    The method is suitable for clustering categorical time series (sequences). 
    You can always have access to the examples and description in 
    https://github.com/HadiJahanshahi/nTreeClus
    If you have any question about the code, you may email hadijahanshahi@gmail.com
    
    prerequisites:
        numpy
        pandas
        sklearn
        scipy
    
    Args:
        my_list: is a of sequences to be clustered
        n: "the window length" or "n" in nTreecluss, you may provide it or it will be
            calculated automatically if no input has been suggested.
        method: 
            DT: Decision Tree
            RF: Random Forest
            All: both methods
        ntree: number of trees to be used in RF method. The defualt value is 10.
        C: number of clusters if it is not provided, it will be calculated for 2 to 10.

    Returns:
        'C_DT': "the optimal number of clusters with the aid of Decision Tree",
        'C_RF': "the optimal number of clusters with the aid of Random Forest",
        'distance_DT': "sparse disance between sequences with the aid of Decision Tree",
        'distance_RF': "sparse disance between sequences with the aid of Random Forest",
        'labels_DT': "labels based on the optimal number of clusters using DT",
        'labels_RF': "labels based on the optimal number of clusters using RF".
            
            NOTE: in order to convert the distance output to a square distance matrix, 
                "scipy.spatial.distance.squareform" should be used.
                
    ## simple example with the output
    my_list = ['evidence','evident','provide','unconventional','convene']
    nTreeClusModel = nTreeClus(my_list, method = "All")
    # {'C_DT': 2,
    # 'C_RF': 2,
    # 'distance_DT': array([ 0.05508882,  0.43305329,  0.68551455,  0.43305329,  0.5       ,
    #         0.7226499 ,  0.5       ,  0.86132495,  0.75      ,  0.4452998 ]),
    # 'distance_RF': array([ 0.0809925 ,  0.56793679,  0.42158647,  0.57878823,  0.56917978,
    #         0.47984351,  0.54545455,  0.55864167,  0.71278652,  0.3341997 ]),
    # 'labels_DT': array([0, 0, 0, 1, 1]),
    # 'labels_RF': array([0, 0, 0, 1, 1])}
    
    
    """
    ############# pre processisng #################
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    import scipy.spatial.distance as ssd
    from scipy.cluster.hierarchy import linkage
    from sklearn.metrics import silhouette_score 
    from scipy import cluster

    
    sequence = pd.DataFrame(my_list) # create a DataFrame out of the given list
    sequence_sep = pd.DataFrame([list(x) for x in sequence[0]]) # separating the sequence
    sequence_sep[sequence_sep.shape[1]] = None # adding an empty column to the end of sequence

    if n is None:
        total_avg = round(sum( map(len, my_list) ) / len(my_list)) # average length of strings
        n =round(total_avg**0.5)+1
    
    ############# matrix segmentation #################
    firstiteration = 1
    for i in range(len(my_list)):
        for j in range(sequence_sep.shape[1]):
            if (j < np.min(np.where(sequence_sep.iloc[i].isnull())) - n + 1): #check where is the position of the first "None"
                if (firstiteration==1):
                    seg_mat = pd.DataFrame(sequence_sep.iloc[i,j:j+n]).transpose()
                    seg_mat['OriginalMAT_element'] = i
                    seg_mat.columns = [np.arange(0,n+1)]
                    firstiteration+=1
                else:
                    temp = pd.DataFrame(sequence_sep.iloc[i,j:j+n]).transpose()
                    temp['OriginalMAT_element'] = i
                    temp.columns = [np.arange(0,n+1)]
                    seg_mat = seg_mat.append(temp)
    seg_mat.columns = np.append(np.arange(0,n-1),('Class','OriginalMAT_element')) # renaming the column indexes
    # dummy variable for DT and RF
    le = preprocessing.LabelEncoder()
    seg_mat.loc[:,'Class'] = le.fit_transform(seg_mat.loc[:,'Class']) #make Y to numbers
    #creating dummy columns for categorical data; one-hot encoding
    seg_mat = pd.get_dummies(seg_mat).reset_index(drop=True)

    

	############# nTreeClus method using DT #################
    xtrain = seg_mat.drop(labels=['OriginalMAT_element','Class'],axis=1)
    ytrain = seg_mat['Class']

    if (method == "All") or (method == "DT"):
        dtree = DecisionTreeClassifier()
        fitted_tree = dtree.fit(X=xtrain,y=ytrain)
        predictiontree = dtree.predict(xtrain)

        ### finding the terminal nodes.
        terminal_tree = fitted_tree.tree_.apply(xtrain.values.astype('float32'))  #terminal output
        terminal_output_tree  = pd.DataFrame(terminal_tree)
        terminal_output_tree ['OriginalMAT_element'] = seg_mat['OriginalMAT_element'].values
        #terminal_output_tree.drop(labels=0,axis=1,inplace=True)
        terminal_output_tree.columns = ['ter','OriginalMAT_element']
        terminal_output_tree_F = pd.crosstab(terminal_output_tree.OriginalMAT_element,terminal_output_tree.ter)
        Dist_tree_terminal_cosine = ssd.pdist(terminal_output_tree_F,metric='cosine')
        HC_tree_terminal_cosine = linkage(Dist_tree_terminal_cosine, 'ward')
        #finding the number of clusters
        if C is None:
            max_clusters = min(11,len(my_list))
            ress_sil = []
            for i in range(2,max_clusters):
                assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(HC_tree_terminal_cosine,i).ravel() #.ravel makes it 1D array.
                ress_sil.append((silhouette_score(ssd.squareform(Dist_tree_terminal_cosine),assignment_tree_terminal_cosine,metric='cosine').round(3)*1000)/1000)
            C = ress_sil.index(max(ress_sil)) + 2        
        # assigning the correct label
        optimal_cluster_tree = C
        assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(HC_tree_terminal_cosine,C).ravel() #.ravel makes it 1D array.
    
        
        
	############# nTreeClus method using RF #################
    if (method == "All") or (method == "RF"):
        forest = RandomForestClassifier(n_estimators= ntree ,max_features = 0.36)
        fitted_forest = forest.fit(X=xtrain,y=ytrain)
        predictionforest = forest.predict(xtrain)
        ### Finding Terminal Nodes
        terminal_forest = fitted_forest.apply(xtrain) #terminal nodes access
        terminal_forest = pd.DataFrame(terminal_forest)
        #Adding "columnindex_" to the beginning of all  
        terminal_forest = terminal_forest.astype('str')
        for col in terminal_forest:
            terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
        terminal_forest.head()

        for i in range(terminal_forest.shape[1]):
            if i == 0:
                tempor = pd.concat( [seg_mat['OriginalMAT_element'] , terminal_forest[i]],ignore_index=True,axis=1)
                rbind_terminal_forest = tempor
            else:
                tempor = pd.concat( [seg_mat['OriginalMAT_element'] , terminal_forest[i]],ignore_index=True,axis=1)
                rbind_terminal_forest = pd.concat ([rbind_terminal_forest, tempor],ignore_index=True)

        rbind_terminal_forest.columns = ['OriginalMAT_element','ter']
        terminal_output_forest_F = pd.crosstab(rbind_terminal_forest.OriginalMAT_element,rbind_terminal_forest.ter)
        Dist_RF_terminal_cosine = ssd.pdist(terminal_output_forest_F,metric='cosine')
        HC_RF_terminal_cosine = linkage(Dist_RF_terminal_cosine, 'ward')
        #finding the number of clusters
        if C is None:
            max_clusters = min(11,len(my_list))
            ress_sil = []
            for i in range(2,max_clusters):
                assignment_RF_terminal_cosine = cluster.hierarchy.cut_tree(HC_RF_terminal_cosine,i).ravel() #.ravel makes it 1D array.
                ress_sil.append((silhouette_score(ssd.squareform(Dist_RF_terminal_cosine),assignment_RF_terminal_cosine,metric='cosine').round(3)*1000)/1000)
            C = ress_sil.index(max(ress_sil)) + 2        
        # assigning the correct label
        optimal_cluster_RF = C
        assignment_RF_terminal_cosine = cluster.hierarchy.cut_tree(HC_RF_terminal_cosine,C).ravel() #.ravel makes it 1D array.

        
    if (method == "All"):
        return {"distance_DT":Dist_tree_terminal_cosine, "labels_DT":assignment_tree_terminal_cosine, "C_DT":optimal_cluster_tree, "distance_RF":Dist_RF_terminal_cosine, "labels_RF": assignment_RF_terminal_cosine, "C_RF":optimal_cluster_RF}
    elif (method == "DT"):
        return {"distance_DT":Dist_tree_terminal_cosine, "labels_DT":assignment_tree_terminal_cosine, "C_DT":optimal_cluster_tree}
    else: 
        return {"distance_RF":Dist_RF_terminal_cosine, "labels_RF":assignment_RF_terminal_cosine, "C_DT":optimal_cluster_RF}



#simple example with the output
my_list = ['evidence','evident','provide','unconventional','convene']
nTreeClusModel = nTreeClus(my_list, method = "All")
# {'C_DT': 2,
# 'C_RF': 2,
# 'distance_DT': array([ 0.05508882,  0.43305329,  0.68551455,  0.43305329,  0.5       ,
#         0.7226499 ,  0.5       ,  0.86132495,  0.75      ,  0.4452998 ]),
# 'distance_RF': array([ 0.0809925 ,  0.56793679,  0.42158647,  0.57878823,  0.56917978,
#         0.47984351,  0.54545455,  0.55864167,  0.71278652,  0.3341997 ]),
# 'labels_DT': array([0, 0, 0, 1, 1]),
# 'labels_RF': array([0, 0, 0, 1, 1])}

