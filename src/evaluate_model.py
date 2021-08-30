import argparse
import os
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, recall_score,roc_auc_score
from sklearn.metrics import plot_roc_curve,classification_report, confusion_matrix,plot_confusion_matrix

#Helper function to evaluate a classifier
def eval_metric(actual, predict):
    target_names =['class_0', 'class_1']
    metric_f1_score = f1_score(actual, predict, average = 'micro')
    metric_recall = recall_score(actual, predict, average = None )
    metric_balanced_accuracy = balanced_accuracy_score(actual, predict)
   
    print(classification_report(actual, predict, target_names=target_names))
    print(confusion_matrix(actual, predict))
    print('f1_score', metric_f1_score)
    print('recall', metric_recall)
    print('balanced_accuracy', metric_balanced_accuracy)
    return (metric_f1_score, metric_recall, metric_balanced_accuracy)

def unpickle(pklfilename):
    with open(pklfilename,'rb') as f:
        output = pickle.load(f)
    return output


def pickler(tobepickled, pklfilename):
    with open(pklfilename, 'wb+') as f:
        pickle.dump(tobepickled, f)


if __name__ == "__main__":

    # commmand line arguments
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_dir", help="path to local data directory")
    parse.add_argument("--model_dir", help="path to local model directory")
    parse.add_argument("--outpath", help="path to local directory to which to save artifacts")
    args = parse.parse_args()

    if args.data_dir is None or os.path.isdir(args.data_dir) is False:
        print('Quitting. Plese specify data directory with --data_dir.')
        exit()
        
    if args.outpath is None:
        args.outpath = os.getcwd()
    

    for entry in os.listdir(args.data_dir):
        if 'X_pca_test.pkl' in entry:
            X_pca_test = unpickle(os.path.join(args.data_dir, entry))
        elif 'y_test.pkl' in entry:
            y_test = unpickle(os.path.join(args.data_dir, entry))
    
    model_list = []

    for entry in os.listdir(args.model_dir):
        if 'pkl' in entry:
            model_list.append((entry[:-4], unpickle(os.path.join(args.model_dir, entry))))

       

    # Check the recall scores for shortlisted classifiers for PCA decomposed dataset
  

    models = []
    names=[]
    results = []
    metrics_dict ={}
    for name, model in model_list:
        pred = model.predict(X_pca_test)
        names.append(name)
        models.append(model)
        print("f1_score for ",name, ":", f1_score(y_test, pred, average = 'macro'))
        print("Recall score for ", name, ":",  recall_score(y_test, pred, average = None ))
        metric_f1_score, metric_recall, metric_balanced_accuracy = eval_metric(y_test, pred) 
        metrics_dict[name] = {'f1_score' : metric_f1_score, 'recall' :metric_recall[0], 'balanced_accuracy' : metric_balanced_accuracy }
        df = pd.DataFrame({'actual': y_test, 'predicted': pred})
        df.to_csv(os.path.join('metrics', name+'_predictions.csv'), index=None)

    
    with open('./metrics/evaluate_metrics.json', 'w') as f:
        json.dump(metrics_dict, f)

    