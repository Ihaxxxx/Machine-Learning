# Generated from: index.ipynb
# Converted at: 2026-03-12T20:25:44.096Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

from IPython.display import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge , LassoCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    ExtraTreesClassifier,
    RandomForestRegressor,
    AdaBoostRegressor
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Model selection
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV , RandomizedSearchCV , KFold

# Metrics
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    recall_score,
    confusion_matrix,
    precision_score,
    f1_score,
    classification_report,
    r2_score
)

# Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder , PowerTransformer
from sklearn.impute import SimpleImputer

df_train = pd.read_csv("./data/train.csv")
df_test = pd.read_csv("./data/test.csv")

train_ID = df_train['Id']
test_ID = df_test['Id']

df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)


datasets = {
    'Train Dataset': df_train,
    'Test Dataset': df_test,
}


df_train.head()

sns.histplot(
    x="SalePrice",
    data=df_train
)

# # Standardizing Data


log = PowerTransformer()
log.fit(df_train[['SalePrice']])

df_train['SalePrice'] = log.transform(df_train[["SalePrice"]])

sns.histplot(
    x="SalePrice",
    data=df_train
)

# # Removing Outliers


mean = df_train['SalePrice'].mean()
std = df_train['SalePrice'].std()

cut_off = std * 3

lower , upper = mean - cut_off , mean + cut_off

df_train = df_train[(df_train['SalePrice'] < upper) & (df_train['SalePrice'] > lower)]

sns.histplot(
    x="SalePrice",
    data=df_train
)

for name, dataset in datasets.items():

    na_values = dataset.isna().sum().reset_index()
    na_values = na_values.rename(columns={'index':'feature', 0:'count'})
    na_values = na_values.sort_values(by='count', ascending=False)

    na_values['missing_ratio'] = na_values['count'] / len(dataset) * 100
    na_values = na_values[na_values['missing_ratio'] != 0]

    sns.barplot(
        x='feature',
        y='missing_ratio',
        data=na_values,
        hue='feature'
    )

    plt.title(name)
    plt.xticks(rotation=90)
    plt.show()

# # PoolQC 
# - No pool 
# - No Qulity


for name , dataset in datasets.items():
    dataset['PoolQC'] = dataset['PoolQC'].fillna('None')

# # MiscFeature
# - As per the description if there aren't any MiscFeature it is NA : none


for name , dataset in datasets.items():
    dataset['MiscFeature'] = dataset['MiscFeature'].fillna('None')

# # Alley 
# - No alley
# - None


df_test['Alley']

for name , dataset in datasets.items():
    dataset['Alley'] = dataset['Alley'].fillna('None')
    print(dataset['Alley'].head())

# # Fence
# - No fence
# - None


for name , dataset in datasets.items():
    dataset['Fence'] = dataset['Fence'].fillna('None')

# # FireplaceQU
# - As description says NA


for name , dataset in datasets.items():
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('None')

# # Garage Stuff


for name , dataset in datasets.items():
    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        dataset[col] = dataset[col].fillna('None')

for name , dataset in datasets.items():
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        dataset[col] = dataset[col].fillna(0)

# # Basement Stuff


for name , dataset in datasets.items():
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        dataset[col] = dataset[col].fillna(0)

for name , dataset in datasets.items():
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        dataset[col] = dataset[col].fillna(0)

# # Functional


for name , dataset in datasets.items():
    dataset['Functional'] = dataset['Functional'].fillna('Typ')

# # Masonry veneer type


for name , dataset in datasets.items():
    dataset["MasVnrType"] = dataset["MasVnrType"].fillna("None")
    dataset["MasVnrArea"] = dataset["MasVnrArea"].fillna(0)


# # Lot Frontage


for name , dataset in datasets.items():
    dataset["LotFrontage"] = dataset.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median()))

# # MS Zoning


for name , dataset in datasets.items():
    dataset['MSZoning'] = dataset['MSZoning'].fillna(dataset['MSZoning'].mode()[0])


# # functional


for name , dataset in datasets.items():
    dataset['Functional'] = dataset['Functional'].fillna('Typ')


# # Electrical


for name , dataset in datasets.items():
    dataset['Electrical'] = dataset['Electrical'].fillna(dataset['Electrical'].mode()[0])

# # Utilities


for name , dataset in datasets.items():
    dataset = dataset.drop(['Utilities'], axis=1)

# # Kitchen Qual


for name , dataset in datasets.items():
    dataset['KitchenQual'] = dataset['KitchenQual'].fillna(dataset['KitchenQual'].mode()[0])

# # Exterior2nd


for name , dataset in datasets.items():
    dataset['Exterior1st'] = dataset['Exterior1st'].fillna(dataset['Exterior1st'].mode()[0])
    dataset['Exterior2nd'] = dataset['Exterior2nd'].fillna(dataset['Exterior2nd'].mode()[0])

# # Sale type


for name , dataset in datasets.items():
    dataset['SaleType'] = dataset['SaleType'].fillna(dataset['SaleType'].mode()[0])

# Checking any remaining values


df_train.drop(['Utilities'],axis=1)
df_train.columns
df_train.shape


print(df_test.shape)
print(df_train.shape)

cols_unique_to_df1 = df_train.columns.difference(df_test.columns)
print(cols_unique_to_df1)

df_train.dtypes

# **Transforming some numerical variables that are really categorical**


for name , dataset in datasets.items():
    dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)


    #Changing OverallCond into a categorical variable
    dataset['OverallCond'] = dataset['OverallCond'].astype(str)


    #Year and month sold are transformed into categorical features.
    dataset['YrSold'] = dataset['YrSold'].astype(str)
    dataset['MoSold'] = dataset['MoSold'].astype(str)

for name , dataset in datasets.items():
    dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']
    print(dataset.columns)

print(df_test.shape)
print(df_train.shape)

cols = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
       'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
       'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2',
       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
       'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt',
       'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond',
       'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
       'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal',
       'MoSold', 'YrSold', 'SaleType', 'SaleCondition']


from sklearn.preprocessing import LabelEncoder


for col in cols:
    
    df_train[col] = df_train[col].fillna('None')
    df_test[col] = df_test[col].fillna('None')
    
    lbl = LabelEncoder()
    
    # combine values from both datasets
    combined = list(df_train[col]) + list(df_test[col])
    
    lbl.fit(combined)
    
    df_train[col] = lbl.transform(df_train[col])
    df_test[col] = lbl.transform(df_test[col])

print(df_test.shape)
print(df_train.shape)

cols_unique_to_df1 = df_test.columns.difference(df_train.columns)
print(cols_unique_to_df1)

cols_unique_to_df1 = df_train.columns.difference(df_test.columns)
print(cols_unique_to_df1)

print(df_test.columns)
print(df_train.columns)

# # Numeric Features and normalising


# df_train['SalePrice'] = log.transform(df_train[["SalePrice"]])

df_train['Alley'].value_counts()

numeric_feats = df_test.dtypes[df_test.dtypes != "object"].index
for col in numeric_feats:
    pt = PowerTransformer()
    print(col)
    if col in df_train.columns:
        pt.fit(df_train[[col]])
        df_train[col] = pt.transform(df_train[[col]])
    else:
        pt.fit(df_test[[col]])
    df_test[col] = pt.transform(df_test[[col]])

df_train.head()

df_train['Alley'].unique()

X_train = df_train
y = df_train.SalePrice

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

cv_ridge.min()

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()

coef = pd.Series(model_lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")

dtrain = xgb.DMatrix(X_train, label = y)

dtest = xgb.DMatrix(df_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)

model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()

model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)

df_test.columns

df_train.columns

xgb_preds = np.expm1(model_xgb.predict(df_test))
lasso_preds = np.expm1(model_lasso.predict(df_test))