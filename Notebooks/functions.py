import pandas as pd
import numpy as np
import random
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load():
    path=""
    train=pd.read_csv("train.csv")
    test=pd.read_csv("test.csv")
    target=pd.read_csv("train_label.csv",delimiter=";")
    
    #drop ID
    train.drop(["Id"],axis=1,inplace=True)
    test.drop(["Id"],axis=1,inplace=True)
    target.drop(["Id"],axis=1,inplace=True)
    
    return train,test,target

def metric_challenge(y_true,y_pred):
    w=[0.6,0.4]
    auc_2014=roc_auc_score(y_true["2014"],y_pred["2014"])
    auc_2015=roc_auc_score(y_true["2015"],y_pred["2015"])
    return(w[0]*auc_2014 + w[1]*auc_2015)


#exploratory analysis motivates the creation of new features.
#TimeToFailure and AlreadyFailed
def feature_engineering(data):
    df_time=pd.DataFrame(data["YearLastFailureObserved"]-data["YearConstruction"],columns=["TimeToFailure"])
    
    
    index_no_NaN=data["YearLastFailureObserved"].dropna().index.values
    already_failed=pd.DataFrame(["NoFail"]*len(data),columns=["AlreadyFailed"])
    already_failed.loc[index_no_NaN]="Fail"
    
    data=pd.concat([data,df_time,already_failed],axis=1)
    return(data)

def preprocess(data):
    #we fill nan values with -1
    data = data.fillna(-1)
    
    #we change categorical features to dummy_features.
    cat_features=["Feature1","Feature2","Feature4","AlreadyFailed"]
    for col in cat_features:
        data = pd.concat([data,pd.get_dummies(data[col])],axis=1)
        data.drop([col],axis=1,inplace=True)
    
    #to avoid redundancy in the dummy coding we drop one dummy for each category
    data.drop(["P","U","C","Fail"],axis=1,inplace=True)
    
    #we scale continuous features
    cont_features=["YearConstruction","Length","TimeToFailure"]
    data[cont_features]=preprocessing.scale(data[cont_features])
    
    return data

def roc_plot_utils(y_pred,y_true):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true=np.array(pd.get_dummies(y_true))
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(np.array(y_true)[:, i],y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr,tpr,roc_auc


    




