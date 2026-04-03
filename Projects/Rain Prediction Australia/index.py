# Generated from: index.ipynb
# Converted at: 2026-03-26T09:21:39.083Z
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
from sklearn.linear_model import Ridge
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV , RandomizedSearchCV

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

import platform
print(platform.architecture())

sns.color_palette("Paired")

plt.ticklabel_format(style='plain', axis='both')

weather = pd.read_csv('weatherAUS.csv')
weather['Date'] = pd.to_datetime(weather['Date'])
weather['month'] = weather['Date'].dt.month
weather['month_name'] = weather['Date'].dt.month_name()
weather['Year'] = weather['Date'].dt.year
weather = weather[(weather['Year'] >= 2008) & (weather['Year'] <= 2016)]

weather.head()

weather.info()

weather.describe()

weather['RainTomorrow'] = np.where(weather['RainTomorrow'] == 'Yes',1,0)
weather['RainToday'] = np.where(weather['RainTomorrow'] == 'Yes',1,0)

plt.figure(figsize=(12,6))
sns.heatmap(weather.corr(numeric_only=True))

weather.dtypes

weather['WindDir3pm']

weather['RainTomorrow'].value_counts()

weather['Location'].unique()

weather['Rainfall'].describe()

# # Avg yearly rainfall in Australia after 2014


newDf = weather.groupby(["Year",'month_name','month'])["Rainfall"].mean().reset_index().sort_values(by='month')
newDf = newDf[newDf['Year'] >= 2014]

sns.lineplot(
    data=newDf,
    x='month_name',
    y='Rainfall',
    hue='Year'
)

plt.xticks(rotation=45)
plt.show()

# # Maximum rainfall in each month since 2014


newDf = weather.groupby(["Year",'month_name','month'])["Rainfall"].max().reset_index().sort_values(by='month')
newDf = newDf[newDf['Year'] >= 2014]
sns.lineplot(
    data=newDf,
    x='month_name',
    y='Rainfall',
    hue='Year'
)

plt.xticks(rotation=45)
plt.show()

# # Citites with the most rainfall in 2017


top5_monthly = (
    weather[weather['Year'] == 2014]
    .groupby(['Location'])['Rainfall']
    .sum()
    .reset_index()
    .sort_values('Rainfall',ascending=False)
    .head(5)
)

rain_by_location_month = (
    weather[weather['Location'].isin(top5_monthly['Location'])]
    .groupby(['Location','month','month_name'])['Rainfall']
    .sum()
    .reset_index()
    .sort_values(['Location','month'])
)

sns.lineplot(
    data=rain_by_location_month,
    x='month_name',
    y='Rainfall',
    hue='Location'
)

plt.xticks(rotation=45)
plt.show()

weather.head()

# # Lowest regions for rainfall


top5_monthly = (
    weather[weather['Year'] == 2014]
    .groupby(['Location'])['Rainfall']
    .sum()
    .reset_index()
    .sort_values('Rainfall',ascending=True)
    .head(5)
)

rain_by_location_month = (
    weather[weather['Location'].isin(top5_monthly['Location'])]
    .groupby(['Location','month','month_name'])['Rainfall']
    .sum()
    .reset_index()
    .sort_values(['Location','month'])
)

sns.lineplot(
    data=rain_by_location_month,
    x='month_name',
    y='Rainfall',
    hue='Location'
)

plt.xticks(rotation=45)
plt.show()

# # Comparision of rainfall over the years in Melbourne


newDf = weather[weather['Location'] == 'Melbourne'].groupby(["month_name","Year",'month'])["Rainfall"].sum().reset_index().sort_values(by='month')
sns.lineplot(
    data=newDf,
    x='month_name',
    y='Rainfall',
    hue='Year'
)
plt.xticks(rotation=45)
plt.show()

pivot = weather[weather['Location'] == 'Melbourne'].pivot_table(
    values='Rainfall',
    index='Year',
    columns='month_name',
    aggfunc='sum'
)
plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="Blues", annot=True,fmt='g')
plt.title("Rainfall Heatmap - Melbourne")
plt.show()

pivot = newDf.pivot(index='month_name', columns='Year', values='Rainfall')

pivot.plot(kind='area', figsize=(10,6))
plt.xticks(rotation=45)
plt.title("Rainfall Accumulation by Month - Melbourne")
plt.show()

season_map = {
    "December": "Summer",
    "January": "Summer",
    "February": "Summer",

    "March": "Autumn",
    "April": "Autumn",
    "May": "Autumn",

    "June": "Winter",
    "July": "Winter",
    "August": "Winter",

    "September": "Spring",
    "October": "Spring",
    "November": "Spring"
}

weather['Season'] = weather['month_name'].map(season_map)

weather[(weather['Season'] == 'Spring')]

newDf = weather.groupby(["Season",'Year'])["Rainfall"].sum().reset_index()
sns.lineplot(
    x='Year',
    y='Rainfall',
    data=newDf,
    hue='Season'
)

pivot = weather.pivot_table(
    values='Rainfall',
    index='Year',
    columns='Season',
    aggfunc='sum'
)
plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="Blues", annot=True,fmt='g')
# plt.title("Rainfall Heatmap - Melbourne")
plt.show()

weather[(weather['Year'] == 2007) & (weather['Season'] == 'Autumn')]['Rainfall'].sum()

# # Yearly Rainfall trend


weather[weather['Year'] == 2008]['Rainfall'].sum()

newDf = weather[(weather['Year'] > 2007) & (weather['Year'] < 2017)].groupby("Year")["Rainfall"].sum().reset_index()
sns.lineplot(
    data=newDf,
    x='Year',
    y='Rainfall'

)

# # Rainfall by Year


driest = weather.groupby('Location')['Rainfall'].sum().reset_index().sort_values(by='Rainfall').min()
wettest = weather.groupby('Location')['Rainfall'].sum().reset_index().sort_values(by='Rainfall').max()
print(driest)
print(wettest)
location = [driest.Location,wettest.Location]

newDf = weather[weather['Location'].isin(location)].groupby(["month_name",'Location','month'])['Rainfall'].sum().reset_index().sort_values(by='month')

sns.lineplot(
    data=newDf,
    x='month_name',
    y='Rainfall',
    hue='Location'
)
plt.xticks(rotation=45)
plt.title("Wettest and the driest regions")
plt.show()

newDf = weather.groupby("Year")["RainToday"].count()
newDf

rainy_days = weather[weather['Rainfall'] > 0]
newDf = rainy_days[rainy_days['Location'].isin(location)].groupby(['Year','Location'])['Rainfall'].count().reset_index()
newDf

# # Number of rainy days in the wettest and driest regions


sns.lineplot(
    x='Year',
    y='Rainfall',
    data=newDf,
    hue='Location'
)
plt.title("Number of rainy days in the driest and the wettest regions")

sns.heatmap(weather.corr(numeric_only=True))

weather.columns

sns.heatmap(weather[['Rainfall','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud3pm','Cloud9am']].corr())

weather['Location'].unique()

coastal_cities = {
    'Albury': 0,
    'BadgerysCreek': 0,
    'Cobar': 0,
    'CoffsHarbour': 1,
    'Moree': 0,
    'Newcastle': 1,
    'NorahHead': 1,
    'NorfolkIsland': 1,
    'Penrith': 0,
    'Richmond': 0,
    'Sydney': 1,
    'SydneyAirport': 1,
    'WaggaWagga': 0,
    'Williamtown': 1,
    'Wollongong': 1,
    'Canberra': 0,
    'Tuggeranong': 0,
    'MountGinini': 0,
    'Ballarat': 0,
    'Bendigo': 0,
    'Sale': 0,
    'MelbourneAirport': 1,
    'Melbourne': 1,
    'Mildura': 0,
    'Nhil': 0,
    'Portland': 1,
    'Watsonia': 0,
    'Dartmoor': 0,
    'Brisbane': 1,
    'Cairns': 1,
    'GoldCoast': 1,
    'Townsville': 1,
    'Adelaide': 1,
    'MountGambier': 0,
    'Nuriootpa': 0,
    'Woomera': 0,
    'Albany': 1,
    'Witchcliffe': 0,
    'PearceRAAF': 0,
    'PerthAirport': 1,
    'Perth': 1,
    'SalmonGums': 0,
    'Walpole': 1,
    'Hobart': 1,
    'Launceston': 0,
    'AliceSprings': 0,
    'Darwin': 1,
    'Katherine': 0,
    'Uluru': 0
}
weather['Coastal'] = weather['Location'].map(coastal_cities)

print(weather['Coastal'].value_counts())

sns.heatmap(weather.corr(numeric_only=True))

newDf = weather.groupby(['Coastal','Year'])[['MinTemp','MaxTemp']].mean().reset_index().sort_values(['Coastal','Year'])
newDf

sns.lineplot(
    data=newDf,
    x='Year',
    y='MinTemp',
    hue='Coastal'
)

plt.title("Avergage Min Temps of coastal cities")

sns.lineplot(
    data=newDf,
    x='Year',
    y='MaxTemp',
    hue='Coastal'
)

plt.title("Avergage Max Temps of coastal cities")

weather.head()

weather.isna().sum()

categorical = [var for var in weather.columns if weather[var].dtype=='O']
numerical = [var for var in weather.columns if weather[var].dtype!='O']

cols = ['Evaporation','Rainfall']

for col in cols:
    weather[col] = weather[col].fillna(0)

weather.isna().sum()

# draw boxplots to visualize outliers

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = weather.boxplot(column='Rainfall')
fig.set_title('')
fig.set_ylabel('Rainfall')


plt.subplot(2, 2, 2)
fig = weather.boxplot(column='Evaporation')
fig.set_title('')
fig.set_ylabel('Evaporation')


plt.subplot(2, 2, 3)
fig = weather.boxplot(column='WindSpeed9am')
fig.set_title('')
fig.set_ylabel('WindSpeed9am')


plt.subplot(2, 2, 4)
fig = weather.boxplot(column='WindSpeed3pm')
fig.set_title('')
fig.set_ylabel('WindSpeed3pm')

plt.figure(figsize=(15,10))


plt.subplot(2, 2, 1)
fig = weather.Rainfall.hist(bins=10)
fig.set_xlabel('Rainfall')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 2)
fig = weather.Evaporation.hist(bins=10)
fig.set_xlabel('Evaporation')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 3)
fig = weather.WindSpeed9am.hist(bins=10)
fig.set_xlabel('WindSpeed9am')
fig.set_ylabel('RainTomorrow')


plt.subplot(2, 2, 4)
fig = weather.WindSpeed3pm.hist(bins=10)
fig.set_xlabel('WindSpeed3pm')
fig.set_ylabel('RainTomorrow')


IQR = weather.Rainfall.quantile(0.75) - weather.Rainfall.quantile(0.25)
Lower_fence = weather.Rainfall.quantile(0.25) - (IQR * 3)
Upper_fence = weather.Rainfall.quantile(0.75) + (IQR * 3)
print('Rainfall outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))

cols = ['Rainfall','Evaporation','WindSpeed9am','WindSpeed3pm']

weather_no_outliers = weather.copy()

for col in cols:
    Q1 = weather[col].quantile(0.25)
    Q3 = weather[col].quantile(0.75)
    IQR = Q3 - Q1
    Lower_fence = Q1 - (IQR * 3)
    Upper_fence = Q3 + (IQR * 3)
    
    print(f'{col} outliers are values < {Lower_fence} or > {Upper_fence}')
    
    # Filter the outliers
    weather_no_outliers = weather_no_outliers[(weather_no_outliers[col] >= Lower_fence) & 
                                              (weather_no_outliers[col] <= Upper_fence)]

weather_no_outliers.isna().sum()

# Fill missing values only for numeric columns to avoid median on string/object dtypes
numeric_cols = weather_no_outliers.select_dtypes(include=[np.number]).columns.tolist()

for col in numeric_cols:
    col_median = weather_no_outliers[col].median()
    weather_no_outliers[col] = weather_no_outliers[col].fillna(col_median)

date_cols = weather_no_outliers.select_dtypes(include=['datetime64[ns]']).columns.tolist()

for col in date_cols:
    col_median = weather_no_outliers[col].median()
    weather_no_outliers[col] = weather_no_outliers[col].fillna(col_median)

weather_no_outliers['WindGustDir'] = weather_no_outliers['WindGustDir'].fillna(weather_no_outliers['WindGustDir'].mode()[0])
weather_no_outliers['WindDir9am'] = weather_no_outliers['WindDir9am'].fillna(weather_no_outliers['WindDir9am'].mode()[0])
weather_no_outliers['WindDir3pm'] = weather_no_outliers['WindDir3pm'].fillna(weather_no_outliers['WindDir3pm'].mode()[0])
weather_no_outliers['RainToday'] = weather_no_outliers['RainToday'].fillna(weather_no_outliers['RainToday'].mode()[0])

categorical = [var for var in weather_no_outliers.columns if weather_no_outliers[var].dtype == 'str']

weather_encoded = pd.get_dummies(weather_no_outliers, columns=categorical, drop_first=True)

print(weather_encoded)

X = weather_encoded.drop(['RainTomorrow','Date'], axis=1)
y = weather_encoded['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.shape, X_test.shape

weather_encoded.isna().sum()

import category_encoders as ce

encoder = ce.BinaryEncoder(cols=['RainToday'])

X_train = encoder.fit_transform(X_train)

X_test = encoder.transform(X_test)

scaler = StandardScaler()

X_train.columns

# # Logistic Regression


logreg = LogisticRegression(solver='liblinear', random_state=0)
logreg.fit(X_train, y_train)
y_pred_test = logreg.predict(X_test)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

# # Descision Tree Regressor


param_dist = {
    "max_depth": [3, 5, 7, 10, 15, None],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [1, 2, 5, 10, 20],
    "max_features": [None, "sqrt", "log2"]
}

model = DecisionTreeRegressor(random_state=42)

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=50,                 # number of random combinations to test
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

best_tree = random_search.best_estimator_
y_pred = random_search.predict(X_test)

print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

rf = RandomForestRegressor(n_estimators=400,min_samples_leaf=0.12,random_state=1)

# Fit 'rf' to training set
rf.fit(X_train,y_train)

# Predict the test set labels 'y_pred'
y_pred = rf.predict(X_test)

# Evaluate the test set RMSE
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))