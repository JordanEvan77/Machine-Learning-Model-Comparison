import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer

def run_eda():
    path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/Kaggle,OutsideSchool/Car Prices/'

    df_train = pd.read_csv(path + 'train-data.csv')
    df_train.head(5)

    df_train.describe()

    cols = df_train.columns.tolist()

    print(cols)
    cars = df_train.drop("Price", axis=1)  # drop the target!
    cars_labels = df_train["Price"].copy()

    cars['Name'].value_counts()  # most common car types to the top
    cars['Location'].value_counts()  # relatively even distribution of location
    cars['Owner_Type'].value_counts()  # fist owner is vastly more common
    cars['Fuel_Type'].value_counts()  # cars that aren't gas or diesel appear to be outliers
    cars['Transmission'].value_counts()  # automatic cars make up about 1/3rd of the data, which is
    # surprising

    # looking at number and string combos:
    cars['Mileage'] = cars['Mileage'].map(lambda x: str(x).strip().split(" ")[0])
    cars['Power'] = cars['Power'].map(lambda x: str(x).strip().split(" ")[0])
    cars['Model'] = cars['Name'].map(lambda x: str(x).strip().split(" ")[1])
    cars['Model'].value_counts()
    model_lister = cars.Model.unique().tolist()
    len(model_lister)  # 212 unique values! Swift, City, i20, Verna and Innova are the most common.
    cars['Model'].hist()

    # check distribution of Seats, Power, Name, Mileage and Engine

    # SEATS
    cars['Seats'].hist(bins=50)
    plt.show()  # few outliers of 10 seats, those must be huge cars!

    # POWER
    cars['Power'].hist(bins=50)
    plt.show()

    # MILEAGE
    cars['Mileage'].hist(bins=50)
    plt.show()

    cars['Mileage'].mean()
    # left skew, but with km/kg as expected, mean at  median at

    # null value check:
    cars.shape  # 6,019 observations
    cars.isnull().sum()  # New price is almost all completely null, should be dropped. at 86%
    # Seats, Power and Engine have minimal missing values.

    cars = cars.drop('New_Price', axis=1)

    # now check type of each column, and if it needs to be changed:
    print(cars.dtypes)
    print('EDA DONE!')

def clean_and_drop(df1):
    """"Takes in a dataframe from car_price data, changes data types, drops bad columns"""
    #dropping Heavy Nulls
    lim1 = len(df1) * .80
    df2 = df1.dropna(thresh=lim1, axis=1)
    print(df2.shape)
    df2['Model'] = df2['Name'].map(lambda x: str(x).strip().split(" ")[1])
    int_maker = ["Year", "Kilometers_Driven"]
    split_list = ["Mileage", "Engine", "Power"]
    for b in split_list:
        print(b)
        df2[b] = df2[b].map(lambda x: str(x).strip().split(" ")[0])
    df2['Power'] = df2['Power'].replace(['null'], 'NaN')
    print('Done1')
    return df2


path = 'C:/Users/jorda/OneDrive/Desktop/PyCharm Community Edition 2021.2.2/Kaggle,OutsideSchool/Car Prices/'
df_train = pd.read_csv(path + 'train-data.csv')
cars_og = df_train.drop("Price", axis=1)  #drop the target!
cars_target = df_train["Price"].copy() #hold the target!



cars = cars_og
cars_labels = cars_target

print(cars.head(5))

#necessary for pipeline:
df_cln = clean_and_drop(cars)  # would be cool to set ignore warning somehow
df_cln = df_cln.iloc[:, 1:]

categoricals = ["Name", "Location", "Fuel_Type", "Transmission", "Owner_Type", "Engine", "Seats",
                "Model"]
cars_num = df_cln.drop(categoricals, axis=1)

#now just the pipeline approach!

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
print('transformer next')

car_num_tr = num_pipeline.fit_transform(cars_num)  # matches his!


num_attribs = list(cars_num)


full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), categoricals),
    ])

car_prepared = full_pipeline.fit_transform(df_cln)

car_prepared_df = pd.DataFrame(car_prepared)
print('yes')
print(car_prepared_df.head(5))
#car_prepared_df.to_csv(path + 'pre-algorithm.csv') # very exciting!

lin_reg = LinearRegression()
lin_reg.fit(car_prepared, cars_labels)
price_predictions = lin_reg.predict(car_prepared)
lin_mse = mean_squared_error(cars_labels, price_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse) # 2.1046

# Next model evaluation:
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(car_prepared, cars_labels)

#model trained, lets evaluate!
car_predictions = tree_reg.predict(car_prepared)
tree_mse = mean_squared_error(cars_labels, car_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse # 0.02046

# this seems even better, close to no error! Seems suspicious, lets check with Cross-Validation

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, car_prepared, cars_labels, scoring="neg_mean_squared_error",
                         cv=10)
tree_rmse_scores = np.sqrt(-scores)

print('Scores', tree_rmse_scores)
print('Mean', tree_rmse_scores.mean()) #4.777
print('SD', tree_rmse_scores.std())

#IT certainly seems like the Decision Tree is actually over fitting here!
linear_scores = cross_val_score(lin_reg, car_prepared, cars_labels, scoring =
"neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-linear_scores)
print('Scores', lin_rmse_scores)
print('Mean', lin_rmse_scores.mean()) #4.71 performing sligthly better than decision tree
print('SD', lin_rmse_scores.std())

# so the linear regression and DecisionTree Model are preforming about the same!

# I want to try random forest now to compare:

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(car_prepared, cars_labels)
#predictions!
forest_preds = forest_reg.predict(car_prepared)
forest_mse = mean_squared_error(cars_labels, forest_preds)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)  # now at 1.3124 lets check the cross validation!

forest_scores = cross_val_score(forest_reg, car_prepared, cars_labels, scoring =
"neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)
print('Scores', forest_rmse_scores)
print('Mean', forest_rmse_scores.mean()) #It is better! 3.544, takes a while to run
print('SD', forest_rmse_scores.std())

print('yes')

# it is very cool that RandomForest on it's own is working better, now to try with a GridSearch
#optimization

#now attempt with grid search: explain how this works:

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30, 50, 100], 'max_features':[2, 4, 6, 8, 10]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features': [2, 3, 4]},
]

forest_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg2, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(car_prepared, cars_labels)
#the below takes a long time to run
grid_search.best_params_ # best parameters! it looks like bootstrap=False, estimators=10,
# features=100)

forest_reg3 = RandomForestRegressor(n_estimators= 10, max_features=100,
                                    random_state=42)
forest_reg3.fit(car_prepared, cars_labels)

forest_preds3 = forest_reg3.predict(car_prepared)
forest_mse3 = mean_squared_error(cars_labels, forest_preds3)
forest_rmse3 = np.sqrt(forest_mse3)
forest_rmse3  # very small, 1.792, which is great!

forest_scores3 = cross_val_score(forest_reg3, car_prepared, cars_labels, scoring =
"neg_mean_squared_error", cv=10)

forest_rmse_scores3 = np.sqrt(-forest_scores3)
print('Scores', forest_rmse_scores3)
print('Mean', forest_rmse_scores3.mean()) #It is slightly worse some how at 4.2, compared 20 3.4
# forest.
print('SD', forest_rmse_scores3.std())

# mean is back down to 4.2, still not our best model! just going with plain random forest

#keeping and testing with test data set and forest_reg:
#performing the Train test split just because: doing it after formatting because of OHE constraints!

df_train, df_test, train_labels, test_labels =train_test_split(car_prepared, cars_target,
                                                                  test_size=0.2, random_state =20)

#transform pipeline:
#  cars_test, cars_labels_test

forest_reg_split = RandomForestRegressor()
forest_reg_split.fit(df_train, train_labels)  # fit with training set!
#predictions!
forest_preds_split = forest_reg.predict(df_test) # predict on test set!
forest_mse = mean_squared_error(test_labels, forest_preds_split)
forest_rmse = np.sqrt(forest_mse)
forest_rmse  # much smaller at 1.12 non validation score!

print('Done, last out put')

#ok, run it through and correct out put numbers, and then immediately begin a write up on this!
#talk about process, and what surprised me.