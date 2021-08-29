import argparse
import os
import pandas as pd
import numpy as np
import json
import time
from scipy import stats



if __name__ == "__main__":

    # commmand line arguments
    parse = argparse.ArgumentParser()
    parse.add_argument("--datafile", help="path to local data file")
    parse.add_argument("--outpath", help="path to local directory to which to save mp4 files.  If not specified, saves to current directory.")
    args = parse.parse_args()
    
    # check if SDE file is specified
    if args.datafile is None or os.path.isfile(args.datafile) is False:
        print('Quitting. Plese specify datafile file with --datafile.')
        exit()
        
    if args.outpath is None:
        args.outpath = os.getcwd()
    
     #read the given dataset
    raw_data = pd.read_json(args.datafile)
   


    #clean up the data, use semi colon delimiter to separate columns
    clean_df = raw_data[raw_data.columns[0]].str.split(';', expand=True)
    clean_df.columns = raw_data.columns[0].split(';')
    print('Dataset Rows and Columns:', clean_df.shape)

    cols_except_region = clean_df.columns.to_list()
    cols_except_region.remove('region')
    clean_df[cols_except_region] = clean_df[cols_except_region].replace('NA', np.nan)

    #convert data type to right format
    clean_df[['areaPopulation', 'routeTotalDistance', 'numberOfShops',
       'marketShare', 'avgAreaBenefits', 'timeFromAvg']] = clean_df[['areaPopulation', 'routeTotalDistance', 'numberOfShops',
                                                                    'marketShare', 'avgAreaBenefits', 'timeFromAvg'
                                                                    ]].applymap(lambda x : pd.to_numeric(x, downcast='float', errors='coerce'))
    clean_df[[ 'advertising','employeeLYScore', 'employeeTenure', 
          'employeePrevComps', 'success']] = clean_df[[ 'advertising','employeeLYScore', 'employeeTenure', 
                                                       'employeePrevComps', 'success'
                                                      ]].applymap(lambda x : pd.to_numeric(x, downcast='integer', errors='coerce'))
    clean_df[['badWeather','weatherRestrictions']] = clean_df[['badWeather','weatherRestrictions']].applymap(lambda x: 1 if x == 'Yes' else 0 )
    clean_df[['birthdate', 'routeDate']] = clean_df[['birthdate', 'routeDate']].applymap(lambda x: pd.to_datetime(x, format='%d/%m/%Y'))

    
    # method to get ordinal values
    def get_ordinal_Wealthlevel(x):
        if x == 'High':
            return 3
        elif x == 'Mid':
            return 2
        elif x == 'Low':
            return 1
        else:
            return np.nan
    
    # Derive new features like age of the driver and weekday of the routeDate
    clean_df['driverAge'] = clean_df['birthdate'].apply(lambda x: 2020 - x.year) # this approximate age, hardcoded year as 2020 as all routeDate are in 2020
    clean_df['weekDay'] = clean_df['routeDate'].apply(lambda x : x.weekday())
    clean_df['areaWealthLevel']=clean_df['areaWealthLevel'].apply(lambda x: get_ordinal_Wealthlevel(x))

    clean_df = pd.get_dummies(clean_df, columns=['gender'], drop_first=False, dummy_na=False)

    unseen_df=clean_df[clean_df.success.isnull()].set_index('anonID')
    train_df = clean_df[clean_df.success.notnull()].set_index('anonID')
    
    train_out = os.path.join(args.outpath, 'train_data.csv')
    unseen_out = os.path.join(args.outpath, 'unseen_data.csv')
    
    train_df.to_csv(train_out, index=['anonID'])
    unseen_df.to_csv(unseen_out, index=['anonID'])