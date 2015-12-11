#!/usr/bin/python2.7
# CS273A: Machine Learning Project
# Prediction of Sales using Ensemble of Regressors
# @author:  Karthik Prasad #42686317
#           Rishabh Shah   #79403075
#           Phani Shekhar  #85686586
#           Sushruth Gopal #57803787
#


from __future__ import  division
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

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


def analyze_data(input_data):
    data = input_data.copy()
    ## 1.
    data['YearMonth'] = data['Date'].apply(lambda x: str(x)[:7])
    average_sales = data.groupby('YearMonth')["Sales"].mean()
    ax0 = average_sales.plot(legend=True, marker='o', title="Month Wise Average Sales")
    ax0.set_xticks(range(len(average_sales)))
    ax0.set_xticklabels(average_sales.index.tolist(), rotation=90)
    plt.show()

    ## 2.
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    # stores closed on sun
    stores_open_sun = data[(data['DayOfWeek'] == 6) & (data['Open'] == 1)]['Store']
    rossmann_open_sun = data[data['Store'].isin(list(set(stores_open_sun.tolist())))][['DayOfWeek', 'Sales']]
    rossmann_closed_sun = data[~data['Store'].isin(list(set(stores_open_sun.tolist())))][['DayOfWeek', 'Sales']]
    # avg sales for the two types of stores
    avg_sales_closed_sun = rossmann_closed_sun.groupby('DayOfWeek').mean()
    avg_sales_open_sun = rossmann_open_sun.groupby('DayOfWeek').mean()

    fig, (axis1,axis2) = plt.subplots(1,2,sharey=True,figsize=(13,8))
    ax1 = avg_sales_closed_sun.plot(kind='bar', legend=True, ax=axis1, title="Average Sales", color='blue', alpha=0.78)
    ax1.set_xticks(range(len(avg_sales_closed_sun)))
    ax1.set_xticklabels(avg_sales_closed_sun.index.tolist(), rotation=90)

    ax2 = avg_sales_open_sun.plot(kind='bar', legend=True, ax=axis2, title="Average Sales", color='red', alpha=0.78)
    ax2.set_xticks(range(len(avg_sales_open_sun)))
    ax2.set_xticklabels(avg_sales_open_sun.index.tolist(), rotation=90)
    plt.show()

    ## 3.
    data[['Sales', 'Assortment']].groupby('Assortment').mean().plot(kind='bar', legend=True, color='orange', alpha=0.78)
    plt.show()
    data[['Sales', 'StoreType']].groupby('StoreType').mean().plot(kind='bar', legend=True, color='purple', alpha=0.78)
    plt.show()
    data[['Sales', 'Promo']].groupby('Promo').mean().plot(kind='bar', legend=True, color='orange', alpha=0.78)
    plt.show()
    data[['Sales', 'SchoolHoliday']].groupby('SchoolHoliday').mean().plot(kind='bar', legend=True, color='green', alpha=0.78)
    plt.show()

    ## 4.
    '''
    for i in set(train.Store.tolist()):
        train[train.Store == str(i)][['Date', 'Sales']].plot(x='Date', y='Sales', title='Store # ' + str(i))
        plt.show()
    '''


# Feature Engineering
def engineer_data(data):
    features = []

    # Extracting the date features from Date
    features.extend(['Store', 'Promo2', 'Day'])
    data['Store'] = data['Store'].astype(int)
    data['Year'] = data.Date.dt.year
    data['Month'] = data.Date.dt.month
    data['Day'] = data.Date.dt.day
    data['DayOfWeek'] = data.Date.dt.dayofweek

    # Add QuarterOfMonth
    features.append('QuarterOfMonth')
    data['QuarterOfMonth'] = (data.Date.dt.day / 7).astype(int)

    # Add a special feature to capture the store opening on Sunday
    features.append('SundayOpen')
    data.loc[(data.DayOfWeek == 6) & (data.Open == 1), 'SundayOpen'] = 1
    data.loc[data.SundayOpen.isnull(), 'SundayOpen'] = 0
    features.append('Promo')

    features.append('StateHoliday')
    mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}
    data.StateHoliday.replace(mappings, inplace=True)
    features.append('SchoolHoliday')

    Storetype = pd.get_dummies(data.StoreType, prefix="StoreType")
    Storetype.drop(Storetype.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, Storetype, left_index=True, right_index=True)
    features.extend(list(Storetype.columns.values))

    Assortment = pd.get_dummies(data.Assortment, prefix="Assortment")
    Assortment.drop(Assortment.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, Assortment, left_index=True, right_index=True)
    features.extend(list(Assortment.columns.values))

    Year = pd.get_dummies(data.Year, prefix="Year")
    Year.drop(Year.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, Year, left_index=True, right_index=True)
    features.extend(list(Year.columns.values))

    DayOfWeek = pd.get_dummies(data.DayOfWeek, prefix="DayOfWeek")
    DayOfWeek.drop(DayOfWeek.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, DayOfWeek, left_index=True, right_index=True)
    features.extend(list(DayOfWeek.columns.values))

    Month = pd.get_dummies(data.Month, prefix="Month")
    Month.drop(Month.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, Month, left_index=True, right_index=True)
    features.extend(list(Month.columns.values))

    features.extend(['CompetitionOpen', 'CompetitionDistance'])
    data['CompetitionOpen'] = 12 * (data.Year - data.CompetitionOpenSinceYear) + \
        (data.Month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    features.append('PromoOpen')
    data['PromoOpen'] = 12 * (data.Year - data.Promo2SinceYear) + \
        (data.Date.dt.weekofyear - data.Promo2SinceWeek) / 4.0
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
    
    # Mapping from state names to numbers
    #features.append('State')
    state_mappings = {'HB,NI': 0, 'HH': 1, 'TH': 2, 'RP': 3, 'ST': 4, 'BW': 5, 'SN': 6, 'BE': 7, 'HE': 8,\
                      'SH': 9, 'BY': 10, 'NW': 11}
    data.State.replace(state_mappings, inplace=True)

    State = pd.get_dummies(data.State, prefix="State")
    State.drop(State.columns[[-1]], axis =1, inplace=True)
    data = pd.merge(data, State, left_index=True, right_index=True)
    features.extend(list(State.columns.values))

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
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
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
    y_test = rf.predict(test[features])
    y_test = np.asarray(np.expm1(y_test))
    # Make Submission
    result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()
    # Replace sales with 0 value for stores which are not opened
    closed_stores = test[test.Open == 0]
    result.loc[result.index.isin(closed_stores.index), 'Sales'] = 0
    result.to_csv('submission_rf.csv')
    print('Created a csv file for submission')


def XGBoost(train, test, features):
    x_train = train[features]
    y_train = train.Sales
    print "Training the data with Gradient Boosting"
    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=42)
    params = {"max_depth": 10,"eta": 0.1,"subsample": 0.9,"colsample_bytree": 0.5,"min_child_weight": 6,
          "silent": 1,"seed": 1255}
    num_iter = 800
    training_data  = xgb.DMatrix(X_train[features], Y_train)
    booster = xgb.train(params, training_data , num_iter)
    importance = booster.get_fscore()
    # Ensure same columns in test data as training
    for col in train.columns:
        if col not in test.columns:
            test[col] = np.zeros(test.shape[0])
    test = test.sort_index(axis=1).set_index('Id')
    print('\nRunning Gradient Boosting algorithm on test data')
    # Make predictions
    y_test = booster.predict(xgb.DMatrix(test[features]))
    y_test = np.asarray(np.expm1(y_test))
    # Make Submission
    result = pd.DataFrame({'Id': test.index.values, 'Sales': y_test}).set_index('Id')
    result = result.sort_index()
    # Replace sales with 0 value for stores which are not opened
    closed_stores = test[test.Open == 0]
    result.loc[result.index.isin(closed_stores.index), 'Sales'] = 0
    result.to_csv('submission_xgb.csv')
    print('Created a csv file for submission')


def weighted_average(csv1 = "submission_xgb.csv", csv2 = "submission_rf.csv"):
    xgb = pd.read_csv(csv1)
    xgb['Sales'] = xgb['Sales'].astype(float)
    rf  = pd.read_csv(csv2)
    rf['Sales'] = rf['Sales'].astype(float)
    result = pd.DataFrame()
    result['Id'] = xgb['Id']
    result['Sales'] = (0.7*xgb['Sales'] + 0.3*rf['Sales'])
    result.to_csv('final_submission.csv')


train, test = read_data()
analyze_data(train)
train = remove_outliers(train)
data, features = engineer_data(train)
test, _ = engineer_data(test)
random_forest(data, test, features, 100)
XGBoost(data,test,features)
weighted_average()