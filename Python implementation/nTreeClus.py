import numpy as np
import pandas as pd
import scipy.spatial.distance as ssd
from nltk import ngrams
from scipy import cluster
from scipy.cluster.hierarchy import linkage
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


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
            n: "the window length" or "n" in nTreecluss. You may provide it or it will be
                calculated automatically if no input has been suggested.
                Currently, the default value of "the square root of average sequences' lengths" is taken.
            method: 
                DT: Decision Tree
                RF: Random Forest
                All: both methods
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
        self.n                               = n   # Parameter n
        self.method                          = method
        self.ntree                           = ntree
        self.C_DT                            = C
        self.C_RF                            = C
        self.sequences                       = sequences
        self.seg_mat                         = None
        self.Dist_tree_terminal_cosine       = None # distance_DT
        self.assignment_tree_terminal_cosine = None # labels_DT
        self.Dist_RF_terminal_cosine         = None # distance_RF
        self.assignment_RF_terminal_cosine   = None # labels_RF
        self.verbose                         = verbose
      
    def matrix_segmentation(self):
        seg_mat_list = []
        for i in tqdm(range(len(self.sequences)), desc="Matrix Segmentation (Splitting based on window size)", disable=1-self.verbose):
            sentence = self.sequences[i]
            sixgrams = ngrams(list(sentence), self.n)
            for gram in sixgrams:
                seg_mat_list.append(list(gram + (i,)))
        self.seg_mat = pd.DataFrame(seg_mat_list)
        self.seg_mat.columns = np.append(np.arange(0,self.n-1),('Class','OriginalMAT_element')) # renaming the column indexes

    def finding_the_number_of_clusters(self, HC_tree_terminal_cosine, Dist_tree_terminal_cosine, which_one):
        """
        which_one can take the values of either "DT" or "RF".
        """
        max_clusters = min(11, len(self.sequences))
        ress_sil = []
        for i in tqdm(range(2, max_clusters), desc=f"Finding the best number of clusters ({which_one})", disable=1-self.verbose):
            assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(HC_tree_terminal_cosine,i).ravel() #.ravel makes it 1D array.
            ress_sil.append((silhouette_score(ssd.squareform(Dist_tree_terminal_cosine),
                                              assignment_tree_terminal_cosine,metric='cosine').round(3)*1000)/1000)
        if which_one == 'DT':
            self.C_DT = ress_sil.index(max(ress_sil)) + 2
        elif which_one == 'RF':
            self.C_RF = ress_sil.index(max(ress_sil)) + 2

    def nTreeClus(self):
        ############# pre processing #################
        if self.n is None:
            if self.verbose: print("Finding the parameter 'n'")
            min_length = min(map(len, self.sequences))
            total_avg  = round(sum( map(len, self.sequences) ) / len(self.sequences)) # average length of strings
            self.n     = min(round(total_avg**0.5)+1, min_length-1)
        
        if (self.n < 3):
            raise ValueError("""Parameter n could not be less than 3.
                                Remove the sequences with the length shorter than 3 and then re-run the function.""")
        
        
        ############# matrix segmentation #################
        self.matrix_segmentation()

        # dummy variable for DT and RF
        if self.verbose: print("one-hot encoding + x/y train")
        le                          = preprocessing.LabelEncoder()
        self.seg_mat.loc[:,'Class'] = le.fit_transform(self.seg_mat.loc[:,'Class']) # Convert Y to numbers
        # creating dummy columns for categorical data; one-hot encoding
        self.seg_mat                = pd.get_dummies(self.seg_mat).reset_index(drop=True)
        
        xtrain                      = self.seg_mat.drop(labels=['OriginalMAT_element','Class'],axis=1)
        ytrain                      = self.seg_mat['Class']

        ############# nTreeClus method using DT #################        
        if (self.method == "All") or (self.method == "DT"):
            dtree                                       = DecisionTreeClassifier()
            if self.verbose: print("Fit DT")
            fitted_tree                                 = dtree.fit(X=xtrain,y=ytrain)
            ### finding the terminal nodes.
            terminal_tree                                = fitted_tree.tree_.apply(xtrain.values.astype('float32'))  #terminal output
            if self.verbose: print("DataFrame of terminal nodes")
            terminal_output_tree                         = pd.DataFrame(terminal_tree)
            terminal_output_tree ['OriginalMAT_element'] = self.seg_mat['OriginalMAT_element'].values
            terminal_output_tree.columns                 = ['ter','OriginalMAT_element']
            terminal_output_tree_F                       = pd.crosstab(terminal_output_tree.OriginalMAT_element, terminal_output_tree.ter)
            if self.verbose: print("Determining the cosine Distance")
            self.Dist_tree_terminal_cosine               = ssd.pdist(terminal_output_tree_F, metric='cosine')
            if self.verbose: print("Applying Ward Linkage")
            self.HC_tree_terminal_cosine                 = linkage(self.Dist_tree_terminal_cosine, 'ward')
            #finding the number of clusters
            if self.C_DT is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_tree_terminal_cosine, self.Dist_tree_terminal_cosine, "DT")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_tree_terminal_cosine = cluster.hierarchy.cut_tree(self.HC_tree_terminal_cosine, self.C_DT).ravel() #.ravel makes it 1D array.
        
            
        ############# nTreeClus method using RF #################
        if (self.method == "All") or (self.method == "RF"):
            np.random.seed(123)
            forest               = RandomForestClassifier(n_estimators=self.ntree, max_features=0.36)
            if self.verbose: print("Fit RF")
            fitted_forest        = forest.fit(X=xtrain, y=ytrain)
            ### Finding Terminal Nodes
            terminal_forest      = fitted_forest.apply(xtrain) #terminal nodes access
            terminal_forest      = pd.DataFrame(terminal_forest)
            #Adding "columnindex_" to the beginning of all  
            terminal_forest      = terminal_forest.astype('str')
            if self.verbose: print("DataFrame of terminal nodes")
            for col in terminal_forest:
                terminal_forest[col] = '{}_'.format(col) + terminal_forest[col]
            terminal_forest.head()
            for i in range(terminal_forest.shape[1]):
                if i == 0:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = temp
                else:
                    temp                  = pd.concat([self.seg_mat['OriginalMAT_element'], terminal_forest[i]], ignore_index=True, axis=1)
                    rbind_terminal_forest = pd.concat([rbind_terminal_forest, temp], ignore_index=True)
            rbind_terminal_forest.columns = ['OriginalMAT_element','ter']
            terminal_output_forest_F      = pd.crosstab(rbind_terminal_forest.OriginalMAT_element, rbind_terminal_forest.ter)
            if self.verbose: print("Determining the cosine Distance")            
            self.Dist_RF_terminal_cosine  = ssd.pdist(terminal_output_forest_F, metric='cosine')
            if self.verbose: print("Applying Ward Linkage")
            self.HC_RF_terminal_cosine    = linkage(self.Dist_RF_terminal_cosine, 'ward')
            #finding the number of clusters
            if self.C_RF is None:
                if self.verbose: print("Finding the optimal number of clusters")
                self.finding_the_number_of_clusters(self.HC_RF_terminal_cosine, self.Dist_RF_terminal_cosine, "RF")
            # assigning the correct label
            if self.verbose: print("Cutting The Tree")
            self.assignment_RF_terminal_cosine = cluster.hierarchy.cut_tree(self.HC_RF_terminal_cosine, self.C_RF).ravel() #.ravel makes it 1D array.

    def output(self):
        return {"C_DT":self.C_DT, "distance_DT":self.Dist_tree_terminal_cosine, "labels_DT":self.assignment_tree_terminal_cosine, 
                "C_RF":self.C_RF, "distance_RF":self.Dist_RF_terminal_cosine, "labels_RF": self.assignment_RF_terminal_cosine, "Parameter n":self.n}
