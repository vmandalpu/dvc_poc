import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json
import yaml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score,roc_auc_score
from sklearn.metrics import plot_roc_curve,classification_report, confusion_matrix,plot_confusion_matrix


def unpickle(pklfilename):
    with open(pklfilename,'rb') as f:
        output = pickle.load(f)
    return output


def pickler(tobepickled, pklfilename):
    with open(pklfilename, 'wb+') as f:
        pickle.dump(tobepickled, f)


if __name__ == "__main__":

    params = yaml.safe_load(open('params.yaml'))['training']
    data_dir = params['data_dir']
    outpath = params['outpath']
    penalty_type = params['penalty']
    alpha_param = params['alpha']
    depth = params['depth']
    samples_split = params['min_samples_split']


    for entry in os.listdir(data_dir):
        if 'X_pca_resampled.pkl' in entry:
            X_pca_resampled = unpickle(os.path.join(data_dir, entry))
        elif 'y_resampled.pkl' in entry:
            y_resampled = unpickle(os.path.join(data_dir, entry))

       

    # Check the crossvalidation scores and recall scores for shortlisted classifiers for PCA decomposed dataset
    model_list = []
    model_list.append(('GradientBoosting', GradientBoostingClassifier(max_depth=depth, min_samples_split =samples_split)))
    model_list.append(('RandomForestClassifier', RandomForestClassifier(max_depth=depth, min_samples_split =samples_split)))
    model_list.append(('SGDCClassifier', SGDClassifier(penalty=penalty_type, alpha=alpha_param)))

    crossval_score_dict ={}
    results = []
    for name, model in model_list:
        model.fit(X_pca_resampled, y_resampled)
        scores = cross_val_score(model, X_pca_resampled, y_resampled, cv= 10, scoring='f1_macro')
        print('Mean Cross Validcation Score for', name, scores.mean())
        pickler(model, os.path.join(outpath, name + '.pkl'))
        crossval_score_dict[name+'_mean_crossval_score'] = scores.mean()

    print(crossval_score_dict)    
    with open('./metrics/train_metric.json', 'w') as f:
        json.dump(crossval_score_dict, f)
    
    print("done")
        