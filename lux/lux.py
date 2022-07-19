import sys
sys.path.append('./pyuid3')

from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyuid3.data import Data
from pyuid3.uid3 import UId3
from pyuid3.uncertain_entropy_evaluator import UncertainEntropyEvaluator
from sklearn.neighbors import NearestNeighbors


class LUX(BaseEstimator):
    def __init__(self,predict_proba, neighborhood_size=0.1,max_depth=2,  node_size_limit = 1, grow_confidence_threshold = 0 ):
        self.neighborhood_size=neighborhood_size
        self.max_depth=max_depth
        self.node_size_limit=node_size_limit
        self.grow_confidence_threshold=grow_confidence_threshold
        self.predict_proba = predict_proba
        self.attributes_names = None
            
    def fit(self,X,y, instance_to_explain, exclude_neighbourhood=False, use_parity=True,class_names=None):
        if class_names is None:
            class_names = np.unique(y)
        if class_names is not None and len(class_names)!=len(np.unique(y)):
            raise ValueError('Length of class_names not aligned with number of classess in y')
            
        self.attributes_names=X.columns
            
        if isinstance(instance_to_explain, (list, np.ndarray)):
            if isinstance(instance_to_explain, (list)):
                instance_to_explain = np.array([instance_to_explain])
            if len(instance_to_explain.shape) == 2:
                return self.fit_bounding_boxes(X=X,y=y,boundiong_box_points=instance_to_explain,exclude_neighbourhood=exclude_neighbourhood, use_parity=use_parity,class_names=class_names)
            else:
                raise ValueError('Dimensions of point to explain not aligned with dataset')
        
    def fit_bounding_boxes(self,X,y, boundiong_box_points, exclude_neighbourhood=False, use_parity=True, class_names=None):
        if class_names is None:
            class_names = np.arange(0,len(y))
        if class_names is not None and len(class_names)!=len(np.unique(y)):
            raise ValueError('Length of class_names not aligned with number of classess in y')
        
        if isinstance(boundiong_box_points, (list)):
                boundiong_box_points = np.array(boundiong_box_points)
        if len(boundiong_box_points.shape) != 2:
            raise ValueError('Bounding box should be 2D.')
            
            
        X_train_sample = self.__create_sample_bb(X,y,boundiong_box_points,exclude_neighbourhood=exclude_neighbourhood, use_parity=use_parity,class_names=class_names)

        y_train_sample = self.predict_proba(X_train_sample)
        uarff=LUX.generate_uarff(X_train_sample,y_train_sample, class_names=class_names)
        
        data = Data.parse_uarff_from_string(uarff)
        self.uid3 = UId3(max_depth=self.max_depth)
        self.tree = self.uid3.fit(data, entropyEvaluator=UncertainEntropyEvaluator(), depth=0)
        
        
    def __create_sample_bb(self,X, y,boundiong_box_points,exclude_neighbourhood=False, use_parity=False, class_names=None):
        neighbourhoods = []
        X_train_sample=[]
        if use_parity:
            for c in class_names:
                X_c_only = X[y==c]
                if self.neighborhood_size <= 1.0:
                    n_neighbors=min(len(X_c_only)-1,max(1,int(self.neighborhood_size*len(X_c_only))))
                    nn = NearestNeighbors(n_neighbors=max(1,int(n_neighbors/len(boundiong_box_points))))
                else:
                    nn = NearestNeighbors(n_neighbors=self.neighborhood_size)
                nn.fit(X_c_only.values)
                for instance_to_explain in boundiong_box_points:
                    _,ids_c = nn.kneighbors(np.array(instance_to_explain).reshape(1,-1))
                    neighbourhoods.append(X_c_only.iloc[ids_c.ravel()])
                if exclude_neighbourhood:
                    X_train_sample.append(X_c_only[~X_c_only.index.isin(pd.concat(neighbourhoods).index)])
                    neighbourhoods = []
    
            if exclude_neighbourhood:
                X_train_sample = pd.concat(X_train_sample)
            else:
                X_train_sample = pd.concat(neighbourhoods)
                
            return X_train_sample
        else:
            X_c_only = X
            if self.neighborhood_size <= 1.0:
                n_neighbors=min(len(X_c_only)-1,max(1,int(self.neighborhood_size*len(X_c_only))))
                nn = NearestNeighbors(n_neighbors=max(1,int(n_neighbors/len(boundiong_box_points))))
            else:
                nn = NearestNeighbors(n_neighbors=self.neighborhood_size)
            nn.fit(X_c_only.values)
            for instance_to_explain in boundiong_box_points:
                _,ids_c = nn.kneighbors(np.array(instance_to_explain).reshape(1,-1))
                neighbourhoods.append(X_c_only.iloc[ids_c.ravel()])

            if exclude_neighbourhood:    
                X_train_sample = X_c_only[~X_c_only.index.isin(pd.concat(neighbourhoods).index)]
            else:
                X_train_sample = X_c_only[X_c_only.index.isin(pd.concat(neighbourhoods).index)]
                
            return X_train_sample
        
        
    def predict(self,X,y=None):
        if isinstance(X, pd.DataFrame):
            pass
        elif isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=self.attributes_names)
        else:
            raise ValueError("Only 2D arrrays are allowed as an input")
            
        if y is None:
            y = pd.Series(np.arange(X.shape[0]),name='target_unused',index=X.index) # This is not used, but Data resered last 
                
        X=pd.concat((X,y),axis=1)
        XData = Data.parse_dataframe(X,'lux')
        return [f.get_name() for f in self.uid3.predict(XData.get_instances())]
    
    def justify(self,X):
        """Traverse down the path for given x."""
        if isinstance(X, pd.DataFrame):
            pass
        elif isinstance(X,np.ndarray):
            X = pd.DataFrame(X,columns=self.attributes_names)
        else:
            raise ValueError("Only 2D arrrays are allowed as an input")
          
        y = pd.Series(np.arange(X.shape[0]),name='target_unused',index=X.index) # This is not used, but Data resered last 
        X=pd.concat((X,y),axis=1)     
        XData = Data.parse_dataframe(X,'lux')
        
        return [ self.uid3.tree.justification_tree(i).to_pseudocode()  for i in XData.get_instances()]
        
        
    def to_HMR(self):
        return self.tree.to_HMR()
    
    
    
    @staticmethod
    def generate_uarff(X,y,class_names):
        uarff="@relation lux\n\n"
        for f,t in zip(X.columns,X.dtypes):
            if t in (int, float):
                uarff+=f'@attribute {f} @REAL\n'
            else:
                domain = ','.join(list(X[f].nunique()))
                uarff+='@attribute '+f+'{'+domain+'}\n'
        

        domain = ','.join([str(cn) for cn in class_names])
        uarff+='@attribute class {'+domain+'}\n'
        
        uarff += '@data\n'
        for i in range(0, X.shape[0]):
            for j in range(0,X.shape[1]):
                uarff+='{:.2f}'.format(X.iloc[i,j])+'[1],'
            uarff+=';'.join([f'{c}[{p}]' for c,p in zip(class_names, y[i,:])])+'\n'
        return uarff