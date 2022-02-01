import math
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk import ngrams
from scipy import cluster
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from scipy.special import comb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score, homogeneity_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

# import scipy.spatial.distance as ssd

class nTreeClus:
    def __init__(self, sequences, n, method, ntree=10, C= None, verbose=1):
        """ nTreeClus is a clustering method by Hadi Jahanshahi and Mustafa Gokce Baydogan.
        The method is suitable for clustering categorical time series (sequences). 
        You can always have access to the examples and description in 
        https://github.com/HadiJahanshahi/nTreeClus
        If you have any question about the code, you may email hadijahanshahi [a t] gmail . com
        
        prerequisites:
            numpy
            pandas
            sklearn
            scipy
        
        Args:
            sequences: a list of sequences to be clustered
            n: "the window length" or "n" in nTreeclus. You may provide it or it will be
                calculated automatically if no input has been suggested.
                Currently, the default value of "the square root of average sequences' lengths" is taken.
            method: 
                DT:          Decision Tree
                DT_position: Decision Tree enhanced by position index
                RF:          Random Forest
                RF_position: Random Forest enhanced by position index
                All:         all four methods
            ntree: number of trees to be used in RF method. The default value is 10. 
                (Setting a small value decreases accuracy, and a large value may increase the complexity. 
                 no less than 5 and no greater than 20 is recommended.)
            C: number of clusters. If it is not provided, it will be calculated using silhouette_score.
            verbose [binary]: It indicates whether to print the outputs or not. 

        Returns:
            'C_DT': "the optimal number of clusters for Decision Tree",
            'C_RF': "the optimal number of clusters for Random Forest",
            'Parameter n': the parameter of the nTreeClus (n) - either calculated or manually entered
            'distance_DT': "sparse distance between sequences for Decision Tree",
            'distance_RF': "sparse distance between sequences for Random Forest",
            'labels_DT': "labels based on the optimal number of clusters for DT",
            'labels_RF': "labels based on the optimal number of clusters for RF".
                
                NOTE: in order to convert the distance output to a square distance matrix, 
                    "scipy.spatial.distance.squareform" should be used.
                    
        ## simple example with the output
        sequences = ['evidence','evident','provide','unconventional','convene']
        model     = nTreeClus(sequences, n = None, ntree=5, method = "All")
        model.nTreeClus()
        model.output()
        # {'C_DT': 2,
        # 'distance_DT': array([0.05508882, 0.43305329, 0.68551455, 0.43305329, 0.5       ,
        #        0.7226499 , 0.5       , 0.86132495, 0.75      , 0.4452998 ]),
        # 'labels_DT': array([0, 0, 0, 1, 1]),
        # 'C_RF': 2,
        # 'distance_RF': array([0.10557281, 0.5527864 , 0.58960866, 0.64222912, 0.55      ,
        #       0.72470112, 0.7       , 0.83940899, 0.95      , 0.26586965]),
        # 'labels_RF': array([0, 0, 0, 1, 1]),
        # 'Parameter n': 4}
        """
        self.n                                 = n   # Parameter n
        self.method                            = method
        self.ntree                             = ntree
        self.C_DT                              = C
        self.C_RF                              = C
        self.C_DT_p                            = C
        self.C_RF_p                            = C
        self.sequences                         = sequences
        self.seg_mat                           = None
        self.Dist_tree_terminal_cosine         = None # distance_DT
        self.assignment_tree_terminal_cosine   = None # labels_DT
        self.Dist_tree_terminal_cosine_p       = None # distance_DT + position
        self.assignment_tree_terminal_cosine_p = None # labels_DT   + position
        self.Dist_RF_terminal_cosine           = None # distance_RF
        self.assignment_RF_terminal_cosine     = None # labels_RF
        self.Dist_RF_terminal_cosine_p         = None # distance_RF + position
        self.assignment_RF_terminal_cosine_p   = None # labels_RF   + position
        self.verbose                           = verbose
        self.running_timeSegmentation          = None
        self.running_timeDT                    = None
        self.running_timeDT_p                  = None
        self.running_timeRF                    = None
        self.running_timeRF_p                  = None
    
    @staticmethod
    def purity_score(clusters, classes):
        """
        Calculate the purity score for the given cluster assignments and ground truth classes
        
        :param clusters: the cluster assignments array
        :type clusters: numpy.array
        
        :param classes: the ground truth classes
        :type classes: numpy.array
        
        :returns: the purity score
        :rtype: float
        """
        clusters = np.array(clusters)
        classes = np.array(classes)
        A = np.c_[(clusters,classes)]

        n_accurate = 0.

        for j in np.unique(A[:,0]):
            z = A[A[:,0] == j, 1]
            x = np.argmax(np.bincount(z))
            n_accurate += len(z[z == x])

        return n_accurate / A.shape[0]
    
    @staticmethod
    def rand_index_score(clusters, classes):
        clusters = np.array(clusters)
        classes = np.array(classes)
        tp_plus_fp = comb(np.bincount(clusters), 2).sum()
        tp_plus_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                for i in set(clusters))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        return (tp + tn) / (tp + fp + fn + tn)
    
    @staticmethod
    def _1nn(Ground_Truth, distance):
        jj = 0
        distance_sqr = pd.DataFrame(squareform(distance))
        for ii in (range(distance_sqr.shape[0])):
            the_shortest_dist = distance_sqr.iloc[ii].drop(ii).idxmin()
            if Ground_Truth[ii] == Ground_Truth[the_shortest_dist]:
                jj += 1 
        return ((jj)/distance_sqr.shape[0])
      
    def matrix_segmentation(self):
        seg_mat_list = []
        for i in tqdm(range(len(self.sequences)), desc="Matrix Segmentation (Splitting based on window size)", 
                      disable=1-self.verbose):
            sentence = self.sequences[i]
            ngrams_  = ngrams(list(sentence), self.n)
            for idx, gram in enumerate(ngrams_):
                seg_mat_list.append(list(gram + (idx,) + (i,)))
        self.seg_mat         = pd.DataFrame(seg_mat_list)
        # renaming the column indexes
        self.seg_mat.columns = np.append(np.arange(0,self.n-1),('Class', 'Position', 'OriginalMAT_element')) 

    def finding_the_number_of_clusters(self, HC_tree_terminal_cosine, Dist_tree_terminal_cosine, which_one):
        """
        which_one can take the values of either "DT" or "RF".
        """
        max_clusters = min(11, len(self.sequences))
        ress_sil = []
        for i in tqdm(range(2, max_clusters), desc=f"Finding the best number of clusters ({which_one})", disable=1-self.verbose):
            assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(HC_tree_terminal_cosine,i).ravel() #.ravel makes it 1D array.
            ress_sil.append((silhouette_score(squareform(Dist_tree_terminal_cosine),
                                              assignment_tree_terminal_cosine,metric='cosine').round(3)*1000)/1000)
        if which_one == 'DT':
            self.C_DT = ress_sil.index(max(ress_sil)) + 2
        elif which_one == 'RF':
            self.C_RF = ress_sil.index(max(ress_sil)) + 2
        elif which_one == 'DT_position':
            self.C_DT_p = ress_sil.index(max(ress_sil)) + 2
        elif which_one == 'RF_position':
            self.C_RF_p = ress_sil.index(max(ress_sil)) + 2

    def nTreeClus(self):
        ############# pre processing #################
        if self.n is None:
            if self.verbose: print("Finding the parameter 'n'")
            min_length = min(map(len, self.sequences))
            total_avg  = round(sum( map(len, self.sequences) ) / len(self.sequences)) # average length of strings
            self.n     = min(round(total_avg**0.5)+1, min_length-1)
            if self.verbose: print(f"Parameter 'n' is set to {self.n}")
        if (self.n < 3):
            raise ValueError("""Parameter n could not be less than 3.
                                Remove the sequences with the length shorter than 3 and then re-run the function.""")
        
        ############# matrix segmentation #################
        start_time                    = time.time()
        self.matrix_segmentation()
        self.running_timeSegmentation = round(time.time() - start_time)

        # dummy variable for DT and RF
        if self.verbose: print("one-hot encoding + x/y train")
        le                            = preprocessing.LabelEncoder()
        self.seg_mat.loc[:,'Class']   = le.fit_transform(self.seg_mat.loc[:,'Class']) # Convert Y to numbers
        # creating dummy columns for categorical data; one-hot encoding
        self.seg_mat                  = pd.get_dummies(self.seg_mat).reset_index(drop=True)
        
        ############# nTreeClus method using DT #################        
        if (self.method in ["All","DT"]):
            start_time                                   = time.time()
            xtrain                                       = self.seg_mat.drop(labels=['OriginalMAT_element', 'Position', 'Class'],
                                                                             axis=1).copy()
            ytrain                                       = self.seg_mat['Class'].copy()
            dtree                                        = DecisionTreeClassifier()
            if self.verbose: print("Fit DT")
            fitted_tree                                  = dtree.fit(X=xtrain,y=ytrain)
            ### finding the terminal nodes.
            terminal_tree                                = fitted_tree.tree_.apply(xtrain.values.astype('float32')) #terminal output
            if self.verbose: print("DataFrame of terminal nodes")
            terminal_output_tree                         = pd.DataFrame(terminal_tree)
            terminal_output_tree ['OriginalMAT_element'] = self.seg_mat['OriginalMAT_element'].values
            terminal_output_tree.columns                 = ['ter','OriginalMAT_element']
            i, r                                         = pd.factorize(terminal_output_tree['OriginalMAT_element'])
            j, c                                         = pd.factorize(terminal_output_tree['ter'])
            ij, tups                                     = pd.factorize(list(zip(i, j)))
            terminal_output_tree_F                       = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
            if self.verbose: print("Determining the cosine Distance")
            self.Dist_tree_terminal_cosine               = squareform(np.round(1-cosine_similarity(terminal_output_tree_F),
                                                                               8))
            if self.verbose: print("Applying Ward Linkage")
            self.HC_tree_terminal_cosine                 = linkage(self.Dist_tree_terminal_cosine, 'ward')
            #finding the number of clusters
            if self.C_DT is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_tree_terminal_cosine, 
                                                    self.Dist_tree_terminal_cosine, "DT")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(self.HC_tree_terminal_cosine, 
                                                                              self.C_DT).ravel() #.ravel makes it 1D array.
            self.running_timeDT                          = round(time.time() - start_time)
            
        ############# nTreeClus method using DT + Position #################        
        if (self.method in ["All","DT_position"]):
            start_time                                   = time.time()
            xtrain                                       = self.seg_mat.drop(labels=['OriginalMAT_element', 'Class'],
                                                                             axis=1).copy()
            ytrain                                       = self.seg_mat['Class'].copy()
            dtree                                        = DecisionTreeClassifier()
            if self.verbose: print("Fit DT + POSITION")
            fitted_tree                                  = dtree.fit(X=xtrain,y=ytrain)
            ### finding the terminal nodes.
            terminal_tree                                = fitted_tree.tree_.apply(xtrain.values.astype('float32')) #terminal output
            if self.verbose: print("DataFrame of terminal nodes")
            terminal_output_tree                         = pd.DataFrame(terminal_tree)
            terminal_output_tree ['OriginalMAT_element'] = self.seg_mat['OriginalMAT_element'].values
            terminal_output_tree.columns                 = ['ter','OriginalMAT_element']
            i, r                                         = pd.factorize(terminal_output_tree['OriginalMAT_element'])
            j, c                                         = pd.factorize(terminal_output_tree['ter'])
            ij, tups                                     = pd.factorize(list(zip(i, j)))
            terminal_output_tree_F                       = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
            if self.verbose: print("Determining the cosine Distance")
            self.Dist_tree_terminal_cosine_p               = squareform(np.round(1-cosine_similarity(terminal_output_tree_F),
                                                                               8))
            if self.verbose: print("Applying Ward Linkage")
            self.HC_tree_terminal_cosine_p                 = linkage(self.Dist_tree_terminal_cosine_p, 'ward')
            #finding the number of clusters
            if self.C_DT_p is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_tree_terminal_cosine_p, 
                                                    self.Dist_tree_terminal_cosine_p, "DT_position")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_tree_terminal_cosine_p = cluster.hierarchy.cut_tree(self.HC_tree_terminal_cosine_p,
                                                                                self.C_DT_p).ravel() #.ravel makes it 1D array.
            self.running_timeDT_p                          = round(time.time() - start_time)
            
        ############# nTreeClus method using RF #################
        if (self.method in ["All","RF"]):
            start_time                                     = time.time()
            xtrain                                         = self.seg_mat.drop(labels=['OriginalMAT_element', 'Position', 'Class'],
                                                                               axis=1).copy()
            ytrain                                         = self.seg_mat['Class'].copy()
            np.random.seed(123)
            forest                                         = RandomForestClassifier(n_estimators=self.ntree, max_features=0.36)
            if self.verbose: print("Fit RF")
            fitted_forest                                  = forest.fit(X=xtrain, y=ytrain)
            ### Finding Terminal Nodes
            terminal_forest                                = fitted_forest.apply(xtrain) #terminal nodes access
            terminal_forest                                = pd.DataFrame(terminal_forest)
            #Adding "columnindex_" to the beginning of all  
            terminal_forest                                = terminal_forest.astype('str')
            if self.verbose: print("DataFrame of terminal nodes")
            for col in terminal_forest:
                terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
            terminal_forest.head()
            for i in range(terminal_forest.shape[1]):
                if i == 0:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], 
                                                       terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = temp
                else:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], 
                                                       terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = pd.concat([rbind_terminal_forest, temp], ignore_index=True)
            rbind_terminal_forest.columns                 = ['OriginalMAT_element','ter']
            i, r                                          = pd.factorize(rbind_terminal_forest['OriginalMAT_element'])
            j, c                                          = pd.factorize(rbind_terminal_forest['ter'])
            ij, tups                                      = pd.factorize(list(zip(i, j)))
            terminal_output_forest_F                      = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
            if self.verbose: print("Determining the cosine Distance")
            self.Dist_RF_terminal_cosine                  = squareform(np.round(1-cosine_similarity(terminal_output_forest_F),8))
            if self.verbose: print("Applying Ward Linkage")
            self.HC_RF_terminal_cosine                    = linkage(self.Dist_RF_terminal_cosine, 'ward')
            #finding the number of clusters
            if self.C_RF is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_RF_terminal_cosine, 
                                                    self.Dist_RF_terminal_cosine, "RF")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_RF_terminal_cosine            = cluster.hierarchy.cut_tree(self.HC_RF_terminal_cosine,
                                                                                       self.C_RF).ravel() #.ravel makes it 1D array.
            self.running_timeRF                           = round(time.time() - start_time)
            
        ############# nTreeClus method using RF + position #################
        if (self.method in ["All","RF_position"]):
            start_time                                     = time.time()
            xtrain                                         = self.seg_mat.drop(labels=['OriginalMAT_element', 'Class'],
                                                                            axis=1).copy()
            ytrain                                         = self.seg_mat['Class'].copy()
            np.random.seed(123)
            forest                                         = RandomForestClassifier(n_estimators=self.ntree, max_features=0.36)
            if self.verbose: print("Fit RF + POSITION")
            fitted_forest                                  = forest.fit(X=xtrain, y=ytrain)
            ### Finding Terminal Nodes
            terminal_forest                                = fitted_forest.apply(xtrain) #terminal nodes access
            terminal_forest                                = pd.DataFrame(terminal_forest)
            #Adding "columnindex_" to the beginning of all  
            terminal_forest                                = terminal_forest.astype('str')
            if self.verbose: print("DataFrame of terminal nodes")
            for col in terminal_forest:
                terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
            terminal_forest.head()
            for i in range(terminal_forest.shape[1]):
                if i == 0:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], 
                                                       terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = temp
                else:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], 
                                                       terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = pd.concat([rbind_terminal_forest, temp], ignore_index=True)
            rbind_terminal_forest.columns                 = ['OriginalMAT_element','ter']
            i, r                                          = pd.factorize(rbind_terminal_forest['OriginalMAT_element'])
            j, c                                          = pd.factorize(rbind_terminal_forest['ter'])
            ij, tups                                      = pd.factorize(list(zip(i, j)))
            terminal_output_forest_F                      = csr_matrix((np.bincount(ij), tuple(zip(*tups))))
            if self.verbose: print("Determining the cosine Distance")
            self.Dist_RF_terminal_cosine_p                = squareform(np.round(1-cosine_similarity(terminal_output_forest_F),8))
            if self.verbose: print("Applying Ward Linkage")
            self.HC_RF_terminal_cosine_p                  = linkage(self.Dist_RF_terminal_cosine_p, 'ward')
            #finding the number of clusters
            if self.C_RF_p is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_RF_terminal_cosine_p, 
                                                    self.Dist_RF_terminal_cosine_p, "RF_position")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_RF_terminal_cosine_p          = cluster.hierarchy.cut_tree(self.HC_RF_terminal_cosine_p, 
                                                                                       self.C_RF_p).ravel() #.ravel makes it 1D array.
            self.running_timeRF_p                         = round(time.time() - start_time)

    def output(self):
        return {"C_DT":self.C_DT, "distance_DT":self.Dist_tree_terminal_cosine, 
                "labels_DT":self.assignment_tree_terminal_cosine, 
                "C_RF":self.C_RF, "distance_RF":self.Dist_RF_terminal_cosine, 
                "labels_RF":self.assignment_RF_terminal_cosine, 
                "C_DT_p":self.C_DT_p, "distance_DT_p":self.Dist_tree_terminal_cosine_p, 
                "labels_DT_p":self.assignment_tree_terminal_cosine_p, 
                "C_RF_p":self.C_RF_p, "distance_RF_p":self.Dist_RF_terminal_cosine_p, 
                "labels_RF_p":self.assignment_RF_terminal_cosine_p, 
                "running_timeSegmentation": self.running_timeSegmentation, 
                "running_timeDT": self.running_timeDT, "running_timeDT_p": self.running_timeDT_p,
                "running_timeRF": self.running_timeRF, "running_timeRF_p": self.running_timeRF_p,
                "Parameter n":self.n}
        
    def performance(self, Ground_Truth):
        """[Reporting the performance]

        Args:
            Ground_Truth ([list]): [list of ground truth labels]

        Returns:
            res [pandas DataFrame]: [A dataframe reporting the performance for different metrics]
        """
        self.res = pd.DataFrame()
        if (self.method in ["All","DT"]):
            predictions_DT           = pd.DataFrame({'labels':Ground_Truth, "labels_DT":self.assignment_tree_terminal_cosine})
            replacement = {}
            for i in predictions_DT.labels_DT.unique():
                replacement[i] = ((predictions_DT[predictions_DT.labels_DT == i].labels.mode()[0]))
            predictions_DT.labels_DT = predictions_DT.labels_DT.map(replacement)
            self.res.loc['DT',"F1S"] = max(score(Ground_Truth, self.assignment_tree_terminal_cosine, average='macro',zero_division=0)[2], 
                                    score(Ground_Truth, predictions_DT.labels_DT, average='macro',zero_division=0)[2]).round(3)
            self.res.loc['DT',"ARS"] = math.ceil((adjusted_rand_score(Ground_Truth, self.assignment_tree_terminal_cosine))*1000)/1000
            self.res.loc['DT',"RS"]  = math.ceil((self.rand_index_score(Ground_Truth, self.assignment_tree_terminal_cosine))*1000)/1000
            self.res.loc['DT',"Pur"] = math.ceil((self.purity_score(Ground_Truth, self.assignment_tree_terminal_cosine))*1000)/1000
            self.res.loc['DT',"Sil"] = math.ceil(silhouette_score(squareform(self.Dist_tree_terminal_cosine),
                                                             self.assignment_tree_terminal_cosine,metric='cosine').round(3)*1000)/1000
            self.res.loc['DT',"1NN"] = math.ceil((self._1nn(Ground_Truth, self.Dist_tree_terminal_cosine))*1000)/1000
        if (self.method in ["All","RF"]):
            predictions_RF = pd.DataFrame({'labels':Ground_Truth, "labels_RF":self.assignment_RF_terminal_cosine})
            # Update cluster names based on the mode of the truth labels
            replacement = {}
            for i in predictions_RF.labels_RF.unique():
                replacement[i] = ((predictions_RF[predictions_RF.labels_RF == i].labels.mode()[0]))
            predictions_RF.labels_RF = predictions_RF.labels_RF.map(replacement)
            self.res.loc['RF',"F1S"] = max(score(Ground_Truth, self.assignment_RF_terminal_cosine, average='macro',zero_division=0)[2], 
                                      score(Ground_Truth, predictions_RF.labels_RF, average='macro',zero_division=0)[2]).round(3)
            self.res.loc['RF',"ARS"] = math.ceil((adjusted_rand_score(Ground_Truth, self.assignment_RF_terminal_cosine))*1000)/1000
            self.res.loc['RF',"RS"]  = math.ceil((self.rand_index_score(Ground_Truth, self.assignment_RF_terminal_cosine))*1000)/1000
            self.res.loc['RF',"Pur"] = math.ceil((self.purity_score(Ground_Truth, self.assignment_RF_terminal_cosine))*1000)/1000
            self.res.loc['RF',"Sil"] = math.ceil(silhouette_score(squareform(self.Dist_RF_terminal_cosine),
                                                             self.assignment_RF_terminal_cosine,metric='cosine').round(3)*1000)/1000
            self.res.loc['RF',"1NN"] = math.ceil((self._1nn(Ground_Truth, self.Dist_RF_terminal_cosine))*1000)/1000
        if (self.method in ["All","DT_position"]):
            predictions_DT = pd.DataFrame({'labels':Ground_Truth, "labels_DT":self.assignment_tree_terminal_cosine_p})
            replacement = {}
            for i in predictions_DT.labels_DT.unique():
                replacement[i] = ((predictions_DT[predictions_DT.labels_DT == i].labels.mode()[0]))
            predictions_DT.labels_DT = predictions_DT.labels_DT.map(replacement)
            self.res.loc['DT_p',"F1S"] = max(score(Ground_Truth, self.assignment_tree_terminal_cosine_p, average='macro',zero_division=0)[2], 
                                    score(Ground_Truth, predictions_DT.labels_DT, average='macro',zero_division=0)[2]).round(3)
            self.res.loc['DT_p',"ARS"] = math.ceil((adjusted_rand_score(Ground_Truth, self.assignment_tree_terminal_cosine_p))*1000)/1000
            self.res.loc['DT_p',"RS"]  = math.ceil((self.rand_index_score(Ground_Truth, self.assignment_tree_terminal_cosine_p))*1000)/1000
            self.res.loc['DT_p',"Pur"] = math.ceil((self.purity_score(Ground_Truth, self.assignment_tree_terminal_cosine_p))*1000)/1000
            self.res.loc['DT_p',"Sil"] = math.ceil(silhouette_score(squareform(self.Dist_tree_terminal_cosine_p),
                                                             self.assignment_tree_terminal_cosine_p,metric='cosine').round(3)*1000)/1000
            self.res.loc['DT_p',"1NN"] = math.ceil((self._1nn(Ground_Truth, self.Dist_tree_terminal_cosine_p))*1000)/1000
        if (self.method in ["All","RF_position"]):
            predictions_RF = pd.DataFrame({'labels':Ground_Truth, "labels_RF":self.assignment_RF_terminal_cosine_p})
            # Update cluster names based on the mode of the truth labels
            replacement = {}
            for i in predictions_RF.labels_RF.unique():
                replacement[i] = ((predictions_RF[predictions_RF.labels_RF == i].labels.mode()[0]))
            predictions_RF.labels_RF = predictions_RF.labels_RF.map(replacement)
            self.res.loc['RF_p',"F1S"] = max(score(Ground_Truth, self.assignment_RF_terminal_cosine_p, average='macro',zero_division=0)[2], 
                                      score(Ground_Truth, predictions_RF.labels_RF, average='macro',zero_division=0)[2]).round(3)
            self.res.loc['RF_p',"ARS"] = math.ceil((adjusted_rand_score(Ground_Truth, self.assignment_RF_terminal_cosine_p))*1000)/1000
            self.res.loc['RF_p',"RS"]  = math.ceil((self.rand_index_score(Ground_Truth, self.assignment_RF_terminal_cosine_p))*1000)/1000
            self.res.loc['RF_p',"Pur"] = math.ceil((self.purity_score(Ground_Truth, self.assignment_RF_terminal_cosine_p))*1000)/1000
            self.res.loc['RF_p',"Sil"] = math.ceil(silhouette_score(squareform(self.Dist_RF_terminal_cosine_p),
                                                             self.assignment_RF_terminal_cosine_p,metric='cosine').round(3)*1000)/1000
            self.res.loc['RF_p',"1NN"] = math.ceil((self._1nn(Ground_Truth, self.Dist_RF_terminal_cosine_p))*1000)/1000            
        return self.res
    
    def plot(self, which_model, labels, save=False, color_threshold=None, linkage_method= 'ward', annotate = False, xy = (0,0),
             rotation=90):
        if which_model == 'RF':
            distance = self.Dist_RF_terminal_cosine
        elif which_model == 'RF_position':
            distance = self.Dist_RF_terminal_cosine_p
        elif which_model == 'DT':
            distance = self.Dist_tree_terminal_cosine
        elif which_model == 'DT_position':
            distance = self.Dist_tree_terminal_cosine_p
        else:
            raise Exception(f'Model {which_model} not supported.')
        HC_tree_terminal_cosine = linkage(distance, linkage_method)
        fig = plt.figure(figsize=(25, 10))
        ax = fig.add_subplot(1, 1, 1)
        if color_threshold == None:
            dendrogram(HC_tree_terminal_cosine,labels=labels, ax=ax)
        else:
            dendrogram(HC_tree_terminal_cosine,labels=labels, ax=ax, color_threshold=color_threshold)            
        ax.tick_params(axis='x', which='major', labelsize=18, rotation=rotation)
        ax.tick_params(axis='y', which='major', labelsize=18)
        if annotate:
            ax.annotate(f"""
                        F1-score = {round(self.res.loc['DT_p', 'F1S'],2)}
                        ARS        = {round(self.res.loc['DT_p', 'ARS'],2)}
                        RS          = {round(self.res.loc['DT_p', 'RS'],2)}
                        Purity     = {round(self.res.loc['DT_p', 'Pur'],2)}
                        ASW       = {round(self.res.loc['DT_p', 'Sil'],2)}
                        1NN       = {round(self.res.loc['DT_p', '1NN'],2)}            
                        """, xy=xy, xytext =(0, 0), fontsize=18, 
                        textcoords='offset points', va='top', ha='left')        
        if save:
            plt.savefig(f"dendrogram_{which_model}.png", dpi=300, bbox_inches='tight')        
        return fig, ax
    
    def __version__(self):
        print('1.2.1')
    
    def updates(self):
        print("""
              - Adding Plotting option
              - Adding Executing time.
              - Adding positional version of nTreeClus 
              - Adding 1NN to the performance metrics
              - Fixing Some bugs in performance calculation
              """)