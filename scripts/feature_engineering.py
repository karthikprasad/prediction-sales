from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = '../data/'

def read_data():
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
    test = pd.read_csv(DATA_DIR+'test.csv', parse_dates=[3], dtype=types)
    store = pd.read_csv(DATA_DIR+'store.csv', dtype=types)
    store_states = pd.read_csv(DATA_DIR+'store_states.csv', dtype=types)

    print 'Merging Training and Test CSV with Store and State CSV'
    train = pd.merge(train, store, on='Store')
    train = pd.merge(train, store_states, on='Store')
    test = pd.merge(test, store, on='Store')
    test = pd.merge(test, store_states, on='Store')

    columns = train.columns.tolist()
    del(columns[columns.index('Sales')])
    columns.append('Sales')
    train = train[columns]

    return train, test


# Feature Engineering
def engineer_data(data):
    features = []
    # --> Remove Store as we don't want the individual store to affect the prediction.
    #     Instead, look at the other charecteristics
    # Add Date
    features.append('Date')
    # Convert DayOfWeek to dummy variables
    for x in range(0,7):
        features.append('DayOfWeek' + str(x+1))
        data['DayOfWeek' + str(x+1)] = data.Date.dt.dayofweek.map(lambda y: 1 if y == x else 0)

    # Add a special feature to capture the store opening on Sunday
    features.append('SundayOpen')
    data.loc[(data.DayOfWeek7 == 1) & (data.Open == 1), 'SundayOpen'] = 1
    data.loc[data.SundayOpen.isnull(), 'SundayOpen'] = 0
    
    data.loc[data.Store == str(x), 'Sales'] = data[data.Store == str(x)].Sales.clip(data[data.Store == str(x)].Sales.mean() - 3*data[data.Store == str(x)].Sales.std(), data[data.Store == str(x)].Sales.mean() + 3*data[data.Store == str(x)].Sales.std())
    
    

    features.extend(['Date', 'Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday', 'StoreType', 'Assortment',\
                     'StateHoliday', 'State'])
    # Handle missing Data
    data.loc[data.Open.isnull(), 'Open'] = (data['DayOfWeek'] != 7).astype(int)

    # Use following features directly
    

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

def remove_outliers(data):
    data = data[data.Sales > 0]
    # Clip Sales for all stores to remove outliers
    for store_id in range(1,1136):
        data_store = data[data.Store == str(store_id)]
        mean_sales = data_store.Sales.mean()
        std_sales = data_store.Sales.std()
        data.loc[data.Store == str(store_id), 'Sales'] = data_store.Sales.clip(mean-3.3*std_sales, mean+3.3*std_sales)

train, test = read_data()
train = remove_outliers(train)

#data, features = engineer_data(train)

