from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = ''

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
                'Promo2SinceYear': np.dtype(float),
                #'SchoolHoliday': np.dtype(str),
                'PromoInterval': np.dtype(str)}

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
    # Convert DayOfWeek to dummy variables
    for x in range(0,7):
        features.append('DayOfWeek' + str(x+1))
        data['DayOfWeek' + str(x+1)] = data.Date.dt.dayofweek.map(lambda y: 1 if y == x else 0)

    print 'hey 1'
    # Extracting the date features from Date
    features.extend(['Store', 'Promo2', 'Year', 'Day'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    #month_imp_mapping = {12:12, 11:7, 10:11, 9:3, 8:8, 7:2, 6:4, 5:10, 4:6, 3:9, 2:5, 1:1}

    for x in range(1,13):
        features.append('Month' + str(x+1))
        data['Month' + str(x+1)] = data.Month.map(lambda y: 1 if y == x else 0)

    #data.Month.replace(month_imp_mapping, inplace=True)
    data['Day'] = data.Date.dt.day

    print 'hey 2'
    # Add QuarterOfMonth
    features.append('QuarterOfMonth')
    data['QuarterOfMonth'] = (data.Date.dt.day / 7).astype(int)
    print 'hey 3'
    # Add a special feature to capture the store opening on Sunday
    features.append('SundayOpen')
    data.loc[(data.DayOfWeek7 == 1) & (data.Open == 1), 'SundayOpen'] = 1
    data.loc[data.SundayOpen.isnull(), 'SundayOpen'] = 0
    print 'hey 4'
    features.append('Promo')

    features.append('StateHoliday')
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StateHoliday.replace(mappings, inplace=True)
    print 'hey 5'
    features.append('SchoolHoliday')

    for x in ['a', 'b', 'c', 'd']:
        features.append('StoreType' + x)
        data['StoreType' + x] = data['StoreType'].map(lambda y: 1 if y == x else 0)
    print 'hey 6'
    for x in ['a', 'b', 'c']:
        features.append('Assortment' + x)
        data['Assortment' + x] = data['Assortment'].map(lambda y: 1 if y == x else 0)
    print 'hey 7'
    # TODO: convert compdist to log scale


    features.extend(['CompetitionOpen', 'CompetitionDistance'])
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    print 'hey 8'
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.Date.dt.weekofyear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0
    print 'hey 9'
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
    print 'hey 10'
    # Mapping from state names to numbers
    features.append('State')
    state_mappings = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5, 'SN': 6, 'BE': 7, 'HE': 8, 'SH': 9, 'BY': 10, 'NW': 11}
    data.State.replace(state_mappings, inplace=True)

    #compds = pd.qcut(data['CompetitionDistance'], 30, labels=range(0,30))
    #compDB = compds.to_frame(name='CompDistBins')
    #data = pd.merge(data, compDB, left_index=True, right_index=True)

    data.fillna(0, inplace=True)

    return data, features


# Calculate the Root Mean Square Percentage Error
def rmspe(exp, pred):
    return np.sqrt(np.mean(((exp - pred)/exp) ** 2))

def remove_outliers(data):
    data = data[data.Sales > 0]
    # Clip Sales for all stores to remove outliers
    for store_id in range(1,1116):
        data_store = data[data.Store == str(store_id)]
        mean_sales = data_store.Sales.mean()
        std_sales = data_store.Sales.std()
        data.loc[data.Store == str(store_id), 'Sales'] = data_store.Sales.clip(mean_sales-3.3*std_sales, mean_sales+3.3*std_sales)
    data['Sales'] = np.log1p(data.Sales)
    return data


def random_forest(train, test, features, num_trees = 20):
    x_train = train[features]
    y_train = train.Sales
    print "Training the data with Random Forest Algorithm"
    rf = RandomForestRegressor(n_jobs = -1, n_estimators = num_trees)
    rf.fit(x_train[features], y_train)

    f_imp = rf.feature_importances_
    for imp, f in zip(f_imp, features):
        print str(f) + "\t->\t" + str(imp)

    # Ensure same columns in test data as training
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.zeros(test.shape[0])

    test = test.sort_index(axis=1).set_index('Id')
    print('\nRunning the RF algorithm on test data')

    # Make predictions
    #X_test = test.drop(['Sales', 'Customers'], axis=1)
    y_test = rf.predict(test[features])
    y_test = np.asarray(np.expm1(y_test))
    # Make Submission
    result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()

    # Replace sales with 0 value for stores which are not opened
    closed_stores = test[test.Open == 0]
    #print closed_stores
    #print result
    result.loc[result.index.isin(closed_stores.index), 'Sales'] = 0

    result.to_csv('submission.csv')
    print('Created a csv file for submission')

train, test = read_data()
train = remove_outliers(train)
data, features = engineer_data(train)
test, _ = engineer_data(test)
random_forest(data, test, features, 80)

