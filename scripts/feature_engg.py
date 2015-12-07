from __future__ import  division
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import time

DATA_DIR = ''

# Feature Engineering
def build_features(data):

    # Replace missing data with 0
    data.loc[data.Open.isnull(), 'Open'] = 1
    data.fillna(0, inplace=True)

    # Use following features directly
    features = []
    #features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday', 'StoreType', 'Assortment',\
    #                 'StateHoliday', 'State', 'CompetitionOpen'])

    # CompDistance : cut it into bins and add the columns to data
    #compds = pd.qcut(data['CompetitionDistance'], 10, labels=[0,1,2,3,4,5,6,7,8,9])
    #compDB = compds.to_frame(name='CompDistBins')
    #data = pd.merge(data, compDB, left_index=True, right_index=True)

    features.extend(['Store', 'CompetitionDistance', 'Promo', 'Promo2', 'SchoolHoliday', 'StateHoliday', \
                     'State', 'CompetitionOpen'])

    # Mapping data of following features to numbers
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    #data.StoreType.replace(mappings, inplace=True)

    for x in ['a', 'b', 'c', 'd']:
        features.append('StoreType' + x)
        data['StoreType' + x] = data['StoreType'].map(lambda y: 1 if y == x else 0)

    for x in ['a', 'b', 'c']:
        features.append('Assortment' + x)
        data['Assortment' + x] = data['Assortment'].map(lambda y: 1 if y == x else 0)



    #data.Assortment.replace(mappings, inplace=True)
    data.StateHoliday.replace(mappings, inplace=True)

    # Mapping from state names to numbers
    state_mappings = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5, 'SN': 6, 'BE': 7, 'HE': 8, 'SH': 9,\
                      'BY': 10, 'NW': 11}
    data.State.replace(state_mappings, inplace=True)

    # Extracting the date features from Date
    features.extend(['Month', 'Day', 'Year', 'WeekOfYear'])
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    #data['DayOfWeek'] = data.Date.dt.dayofweek
    data['WeekOfYear'] = data.Date.dt.weekofyear

    # Day of week
    for x in [0,1,2,3,4,5,6]:
        features.append('DayOfWeek' + str(x))
        data['DayOfWeek' + str(x)] = data.Date.dt.dayofweek.map(lambda y: 1 if y == x else 0)

    # Extracting Competitor's data and promo data
    #features.append('CompetitionOpen')
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)

    # CompDistance : cut it into bins and add the columns to data
    #compos = pd.qcut(data['CompetitionOpen'], 5, labels=[0,1,2,3,4])
    #compoB = compos.to_frame(name='CompOpenBins')
    #data = pd.merge(data, compoB, left_index=True, right_index=True)

    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.WeekOfYear - data.Promo2SinceWeek) / 4.0
    data['PromoOpen'] = data.PromoOpen.apply(lambda x: x if x > 0 else 0)
    data.loc[data.Promo2SinceYear == 0, 'PromoOpen'] = 0

    # Indicate that sales on that day are in promo interval
    features.append('IsPromoMonth')
    month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', \
                 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

    data['monthStr'] = data.Month.map(month2str)
    data.loc[data.PromoInterval == 0, 'PromoInterval'] = ''
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


## Start of main script
def main():
    print("Extract the Training, Test, Store and States csv file")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 170)
    types = {'CompetitionOpenSinceYear': np.dtype(int),
             'CompetitionOpenSinceMonth': np.dtype(int),
             'StateHoliday': np.dtype(str),
             'Promo2SinceWeek': np.dtype(int),
             'SchoolHoliday': np.dtype(int),
             'PromoInterval': np.dtype(str)}

    train = pd.read_csv(DATA_DIR+'train.csv', parse_dates=[2], dtype=types)
    test = pd.read_csv(DATA_DIR+'test.csv', parse_dates=[3], dtype=types)
    store = pd.read_csv(DATA_DIR+'store.csv')
    store_states = pd.read_csv(DATA_DIR+'store_states.csv')

    print '\nMerging Training and Test CSV with Store and State CSV'
    train = pd.merge(train, store, on='Store')
    train = pd.merge(train, store_states, on='Store')
    test = pd.merge(test, store, on='Store')
    test = pd.merge(test, store_states, on='Store')
    test1 = test

    #print test[test.Open == 0]
    #exit()

    print '\nPrinting training data without building features'
    print train.tail(2)

    print '\nAdding and modifying the features of train and test data'
    train, features = build_features(train)
    train[features + ['Sales']].to_csv('my_features.csv', index=False);
    test, _ = build_features(test)
    #print 'Done'

    #print '\nPrinting training data after building the features'
    #print train[features].tail(2)
    #print '############'
    #print train.tail(10)
    #print train.describe()
    #print '############'
    #print train.info()
    #print '############'
    #print features
    #print '############'

    #exit()

    # Including training data where sales is greater than zero
    train = train[train['Sales'] > 0]

    print '\nTraining the data with Random Forest Algorithm'
    x_train = train.drop(['Sales', 'Customers'], axis = 1)
    y_train = train.Sales
    y_train = np.log1p(y_train)

    ############################
    # Training the RF algorithm
    ############################
    print features
    # Note - n_estimators refers to number of trees. More the number, more will be the running time.
    rf = RandomForestRegressor(n_jobs = -1, n_estimators = 100)
    rf.fit(x_train[features], y_train)

    # Printing the importance of individual features
    f_imp = rf.feature_importances_
    for imp, f in zip(f_imp, features):
        print str(f) + "\t->\t" + str(imp)
    #exit()

    '''
    ########################
    # Running the algorithm on trained data to make predictions
    ########################

    print '\nRunning the algorithm on training data again'

    x_test = train.drop(['Sales'], axis = 1)
    x_test = x_train.head(100000)

    y_test = rf.predict(x_test[features])


    y_train = train.Sales
    y_train = np.asarray(y_train.head(100000).tolist())
    y_test  = np.asarray(y_test)
    y_test = np.asarray(np.expm1(y_test))

    print y_train
    print '#######################'
    print y_test

    error = rmspe(y_train, y_test)
    print error
    exit()
    '''
    ########################
    # Running the algorithm on Original test data for Kaggle Competition
    ########################

    # Ensure same columns in test data as training
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.zeros(test.shape[0])

    test = test.sort_index(axis=1).set_index('Id')
    print('\nRunning the RF algorithm on test data')

    # Make predictions
    X_test = test.drop(['Sales', 'Customers'], axis=1)
    y_test = rf.predict(X_test[features])
    y_test = np.asarray(np.expm1(y_test ))
    # Make Submission
    result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()

    # Replace sales with 0 value for stores which are not opened
    xa = test1[test1.Open == 0]
    result.loc[result.index.isin(xa.Id), 'Sales'] = 0

    result.to_csv('submission.csv')
    print('Created a csv file for submission')


if __name__ == '__main__':
    main()
