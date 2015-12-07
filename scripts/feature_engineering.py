from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = '../data/'

# Feature Engineering
def build_features(data):
    # Handle missing Data
    data.loc[data.Open.isnull(), 'Open'] = (data['DayOfWeek'] != 7).astype(int)

    # Use following features directly
    features = []
    features.extend(['Date', 'Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday', 'StoreType', 'Assortment',\
                     'StateHoliday', 'State'])

    # Mapping data of following features to numbers
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StoreType.replace(mappings, inplace=True)
    data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    # Mapping from state names to numbers
    state_mappings = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5, 'SN': 6, 'BE': 7, 'HE': 8, 'SH': 9, 'BY': 10, 'NW': 11}
    data.State.replace(state_mappings, inplace=True)

    # Extracting the date features from Date
    features.extend(['DayOfWeek', 'Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # Extracting Competitor's data and promo data
    features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + (data.Month - data.CompetitionOpenSinceMonth)
    data.loc['CompetitionDistance'] = ((data['CompetitionOpen'] > 0).astype(int) * data['CompetitionDistance'])

    #CompDistance : cut it into bins and add the columns to data
    #features.append('CompDistBins')
    #compds = pd.qcut(data['CompetitionDistance'], 5, labels=[1,2,3,4,5])
    #compDB = compds.to_frame(name='CompDistBins')
    #compDB.loc[compDB.CompDistBins.isnull(), 'CompDistBins'] = 0
    #data = pd.merge(data, compDB, left_index=True, right_index=True)
    

    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

    data['monthStr'] = data.Month.map(month2str)
    data['PromoInterval'] = data['PromoInterval'].astype(str)
    data.loc[data.PromoInterval == 'nan', 'PromoInterval'] = ''
    data['IsPromoMonth'] = 0
    for interval in data.PromoInterval.unique():
        if interval != '':
            for month in interval.split(','):
                data.loc[(data.monthStr == month) & (data.PromoInterval == interval),\
                         'IsPromoMonth'] = 1

    return data, features


# Calculate the Root Mean Square Percentage Error
def rmspe(exp, pred):
    return np.sqrt(np.mean(((exp - pred)/exp) ** 2))


def combine_data():
    print('Extract the Training, Test, Store and States csv file')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 170)
    types = {   'Store': np.dtype(str),
                #'DayOfWeek': np.dtype(int),
                #'Open': np.dtype(str),
                #'Promo': np.dtype(str),
                'CompetitionDistance' : np.dtype(float),
                'CompetitionOpenSinceYear': np.dtype(float),
                'CompetitionOpenSinceMonth': np.dtype(float),
                #'StateHoliday': np.dtype(str),
                'Promo2SinceWeek': np.dtype(float),
                'Promo2SinceYear': np.dtype(float)}
                #'SchoolHoliday': np.dtype(str),
                #'PromoInterval': np.dtype(str)}

    train = pd.read_csv(DATA_DIR+'train.csv', parse_dates=[2], dtype=types)
    #test = pd.read_csv(DATA_DIR+'test.csv', parse_dates=[3], dtype=types)
    store = pd.read_csv(DATA_DIR+'store.csv', dtype=types)
    store_states = pd.read_csv(DATA_DIR+'store_states.csv', dtype=types)

    print 'Merging Training and Test CSV with Store and State CSV'
    train = pd.merge(train, store, on='Store')
    train = pd.merge(train, store_states, on='Store')
    #test = pd.merge(test, store, on='Store')
    #test = pd.merge(test, store_states, on='Store')

    columns = train.columns.tolist()
    del(columns[columns.index('Sales')])
    columns.append('Sales')
    train = train[columns]

    test = None
    return train, test


train, test = combine_data()
data, features = build_features(train)

