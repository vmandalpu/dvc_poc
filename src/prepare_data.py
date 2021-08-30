#preprocessing libraries
import argparse
import os
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import LabelEncoder,StandardScaler, normalize,PowerTransformer
from sklearn.decomposition import PCA, FastICA
from imblearn.over_sampling import ADASYN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pickle
from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    # commmand line arguments
    parse = argparse.ArgumentParser()
    parse.add_argument("--datafile", help="path to local data file")
    parse.add_argument("--seed", help="Random state in Integer")
    parse.add_argument("--split", help= "Train test split value between 0.2 to 0.4")
    parse.add_argument("--outpath", help="path to local directory to which to save artifacts")
    args = parse.parse_args()


    # read params
    params = yaml.safe_load(open('params.yaml'))['prepare_data']

    args.seed = params['seed']
    args.split = params['split']
    args.datafile = params['datafile']
    args.outpath = params['outpath']
    
    # check if SDE file is specified
    if args.datafile is None or os.path.isfile(args.datafile) is False:
        print('Quitting. Plese specify datafile file with --datafile.')
        exit()
        
    if args.outpath is None:
        args.outpath = os.getcwd()
    
    if args.seed is None:
        args.seed = 42

    if args.split is None:
        args.split = 0.2


    train_df = pd.read_csv(args.datafile)
    train_df = train_df.set_index('anonID')

    region_encoder = LabelEncoder()
    region_encoder.fit(train_df.region)
    train_df['regionEncoded'] = region_encoder.transform(train_df.region)
    print(region_encoder.classes_)

    feature_cols = ['areaWealthLevel','areaPopulation', 'badWeather', 'weatherRestrictions',
        'routeTotalDistance', 'numberOfShops', 'marketShare', 'avgAreaBenefits',
        'timeFromAvg', 'advertising', 'employeeLYScore', 'employeeTenure',
        'employeePrevComps', 'gender_F', 'gender_M', 'gender_X',
        'driverAge', 'weekDay', 'regionEncoded']

    imp = IterativeImputer(random_state=int(args.seed))
    imp.fit(train_df[feature_cols])#make sure Success column is not considered for  iterative imputing


    train_imp = pd.DataFrame(imp.transform(train_df[feature_cols]), columns = feature_cols, index = train_df.index)
    train_imp[['regionEncoded', 'weekDay', 'areaWealthLevel']] = train_imp[['regionEncoded', 'weekDay', 'areaWealthLevel']].applymap(lambda x : round(x))
    train_imp =pd.merge(train_imp, train_df['success'], left_index=True,right_index=True  )



    numberOfShops_99 =train_imp.numberOfShops.quantile(np.linspace(0.01, 1, 99, 0)).tail(1).values[0]
    areaPopulation_99 =train_imp.areaPopulation.quantile(np.linspace(0.01, 1, 99, 0)).tail(1).values[0]
    routeTotalDistance_99 =train_imp.routeTotalDistance.quantile(np.linspace(0.01, 1, 99, 0)).tail(1).values[0]
    avgAreaBenefits_99 =train_imp.avgAreaBenefits.quantile(np.linspace(0.01, 1, 99, 0)).tail(1).values[0]

    percentile_99 = {'numberOfShops' : numberOfShops_99, 'areaPopulation':areaPopulation_99,
                            'routeTotalDistance' :routeTotalDistance_99, 'avgAreaBenefits':avgAreaBenefits_99}

    #replace outliers with 99th percentile value
    train_imp['numberOfShops'] = train_imp['numberOfShops'].apply(lambda x: numberOfShops_99 if x>numberOfShops_99 else x )
    train_imp['areaPopulation'] = train_imp['areaPopulation'].apply(lambda x: areaPopulation_99 if x>areaPopulation_99 else x )
    train_imp['routeTotalDistance'] = train_imp['routeTotalDistance'].apply(lambda x: routeTotalDistance_99 if x>routeTotalDistance_99 else x )
    train_imp['avgAreaBenefits'] = train_imp['avgAreaBenefits'].apply(lambda x: avgAreaBenefits_99 if x>avgAreaBenefits_99 else x )



    X_train, X_test, y_train, y_test = train_test_split(train_imp[feature_cols], train_imp.success,
                                                        test_size=float(args.split),random_state=int(args.seed))
    print(y_train.value_counts())



    # Over sample data too address imbalance
    ada = ADASYN()
    X_resampled, y_resampled = ada.fit_resample(X_train, y_train)
    print(y_resampled.value_counts())


    #scaling the data
    std_scaler = StandardScaler()
    std_scaler.fit(X_train)
    X_scaled_train = std_scaler.transform(X_train)
    X_scaled_resampled = std_scaler.transform(X_resampled)
    X_scaled_test = std_scaler.transform(X_test)


    # PCA Decompose
    pca_transformer = PCA(0.7)
    pca_transformer.fit(X_scaled_train)
    X_pca_resampled = pca_transformer.transform(X_scaled_resampled)
    X_pca_test = pca_transformer.transform(X_scaled_test)

    print(X_pca_resampled.shape)


    def pickler(tobepickled, pklfilename):
        with open(pklfilename, 'wb+') as f:
            pickle.dump(tobepickled, f)
    


    pickler(std_scaler, os.path.join(args.outpath, 'scaler.pkl'))
    pickler(pca_transformer,os.path.join(args.outpath, 'pca_transformer.pkl'))
    pickler(imp,os.path.join(args.outpath, 'imputer.pkl' ))
    pickler(region_encoder, os.path.join(args.outpath, 'region_encoder.pkl' ))
    pickler(percentile_99, os.path.join(args.outpath, 'percentile_99.pkl'))
    pickler(X_pca_resampled, os.path.join(args.outpath, 'X_pca_resampled.pkl'))
    pickler(X_pca_test, os.path.join(args.outpath, 'X_pca_test.pkl'))
    pickler(y_resampled, os.path.join(args.outpath, 'y_resampled.pkl'))
    pickler(y_test, os.path.join(args.outpath, 'y_test.pkl'))

