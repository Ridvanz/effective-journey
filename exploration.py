# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 01:00:50 2021

@author: Ridvanz
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.inspection import permutation_importance

import category_encoders as ce
from helpers import CardinalityTruncater as ct

import time 
import pickle

import lightgbm as lgb
import optuna
import optuna.integration.lightgbm as optuna_lgb

import shap
#%% Load Data

# import datasets
features_df = pd.read_csv('data/water_pump_set.csv')  
labels_df = pd.read_csv('data/water_pump_labels.csv')  

features_df.set_index('id', inplace=True)
labels_df.set_index('id', inplace=True)

# Store for reference
original_df = pd.concat([features_df, labels_df],1).sort_index()
#%% Initial Exploration (I use Spyder's variable explorer)

variable_types = features_df.dtypes
unique_counts = features_df.nunique()
nan_counts= features_df.isna().sum()

unique_value_ratios = [features_df[x].value_counts(normalize=True) for x in features_df]
label_ratio = labels_df.status_group.value_counts(normalize=True)

descriptions = features_df.describe().T

#%% Initial cleaning

# Replace anomalous values by NaNs
null_values = {   'amount_tsh':0,
                  'date_recorded':0,
                  'gps_height':0,
                  'longitude':0,
                  'latitude':-2e-08,
                  'population':0,
                  'construction_year':0}

for feature, null_item in null_values.items():
    features_df[feature].replace(null_item, np.nan, inplace=True)

# Also replace unknown values to NaNs
features_df.replace(["unknown" ,"0" ], np.nan, inplace=True)

nan_counts= features_df.isna().sum()
#%%

#Change variable dtypes 
for feature in features_df:
    features_df[feature] = features_df[feature].astype('object')
for feature in ["amount_tsh","longitude", "gps_height", "latitude", "population", "construction_year", "num_private"]:
    features_df[feature] = features_df[feature].astype('float64')
    
features_df["date_recorded"] = features_df["date_recorded"].astype('string')

variable_types = features_df.dtypes

#%%

# Infer age of the waterpump from date_recorded and construction_year
    
date_recorded = pd.to_datetime(features_df.date_recorded)
year1900 = pd.to_datetime('1900-01-01')
years = [i.days/365 for i in (date_recorded - year1900)]

waterpump_age= years-(features_df.construction_year-1900)
waterpump_age[waterpump_age<0]=0
features_df['age'] = waterpump_age.astype("float64")
features_df['date_recorded'] = years

#Remove the now redundant construction year feature
features_df.drop(['construction_year'], inplace=True, axis=1)

#%% Split the dataset to avoid leakage during exploration 
X_train, X_test, y_train, y_test = train_test_split(features_df, labels_df, stratify=labels_df, test_size=0.2,random_state=42)

# Split the features in categorical and numerical columns
X_train_cat = X_train.select_dtypes(include=['object']).copy()
X_train_num = X_train.select_dtypes(exclude=['object']).copy()
X_test_cat = X_test.select_dtypes(include=['object']).copy()
X_test_num = X_test.select_dtypes(exclude=['object']).copy()

#%%
# For all categorical variables, combine the infrequently occuring categories into a new category called "other"

cardinal_trunc = ct.CardinalityTruncater(treshold = 0.001)
cardinal_trunc.fit(X_train_cat)
X_train_cat = cardinal_trunc.transform(X_train_cat)
X_test_cat = cardinal_trunc.transform(X_test_cat)

# cat_unique_counts = X_train_cat.nunique()
cat_nan_counts= X_train_cat.isna().sum()
cat_other_counts = (X_train_cat=="other").sum()

# Calculate the ratio of useful samples (samples neither NaN or "other")
cat_useful = 1-(cat_nan_counts+cat_other_counts)/X_train_cat.shape[0]

#%%

# Remove columns that are redundant, are a lower granularity of other variables, have a low ratio of useful samples or have only 1 category.

dropped_cols =   ['management_group', 
                 'extraction_type_group', 
                 'extraction_type_class', 
                 "payment_type", 
                 "quality_group", 
                 "source_type", 
                 "source_class", 
                 "waterpoint_type_group",
                 'recorded_by', 
                 'wpt_name', 
                 "subvillage",
                 "ward", 
                 "region",
                 "district_code",
                 "quantity_group",
                 "scheme_name",
                 "basin",
                 "scheme_management"]

X_train_cat.drop(dropped_cols, inplace=True, axis=1)
X_test_cat.drop(dropped_cols, inplace=True, axis=1)

#Remove numerical variables with low ratio of useful samples
X_train_num.drop(["amount_tsh", "num_private"] , inplace=True, axis=1)
X_test_num.drop(["amount_tsh", "num_private"] , inplace=True, axis=1)


for feature in X_train_cat:
    X_train_cat[feature] = X_train_cat[feature].astype('category')
    X_test_cat[feature] = X_test_cat[feature].astype('category')

# unique_cat_ratios = [X_train_cat[x].value_counts(normalize=False) for x in X_train_cat]


#%% A baseline Random Forest classifier and feature importance attempt

X_base_cat = X_train_cat.copy()
X_base_num = X_train_num.copy()

# Add redundant continuous and categorical variable as a reference

X_base_cat["random_cat"] = np.random.choice(10,X_base_cat.shape[0])
X_base_cat["random_cat"] = X_base_cat["random_cat"].astype("category")
X_base_num["random_num"] = np.random.randn(X_base_num.shape[0])

# Encode  categorical variables
ce_ordinal = ce.OrdinalEncoder()
X_base_cat = ce_ordinal.fit_transform(X_base_cat)
# ce_binary = ce.BinaryEncoder()
# X_base_cat = ce_binary.fit_transform(X_base_cat)
# ce_onehot = ce.OneHotEncoder()
# X_base_cat = ce_onehot.fit_transform(X_base_cat)

imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
X_base_num[:] = imp_mean.fit_transform(X_base_num)

X_base = pd.concat([X_base_cat, X_base_num],1)

#%% Get crossvalidation score with Random Forest


forest = RandomForestClassifier(random_state=0, n_estimators=300, max_depth=8, n_jobs=-1)
scores = cross_val_score(forest, X_base, y_train.values.ravel(), cv=5)
print("%0.5f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))


#%% Calculate and plot feature importances (MDI)

start_time = time.time()
forest.fit(X_base, y_train.values.ravel())
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

indices = np.argsort(importances)

plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', ecolor ='r', align='center', xerr=std[indices])
plt.yticks(range(len(indices)), X_base.columns[indices])
plt.xlabel('Relative Importance')

#%% Calculate and plot feature importances (feature permutation)

start_time = time.time()
result = permutation_importance(forest, X_base, y_train, n_repeats=5, random_state=42, n_jobs=-1)
elapsed_time = time.time() - start_time

indices = np.argsort(result.importances_mean)

plt.figure()
plt.title('Feature Importances')
plt.barh(range(len(indices)), result.importances_mean[indices], color='g', ecolor ='r', align='center', xerr=result.importances_std[indices])
plt.yticks(range(len(indices)), X_base.columns[indices])
plt.xlabel('Relative Importance')

#%%
# Drop public_meeting and permit because they have low feature importance in both tests

X_train_cat.drop(["public_meeting", "permit"] , inplace=True, axis=1)
X_test_cat.drop(["public_meeting", "permit"] , inplace=True, axis=1)

#%% Prepare data for use in LightGBM classifier

ce_ordinal = ce.OrdinalEncoder(handle_unknown = "return_nan", handle_missing = "return_nan")
X_train_cat_gbm = ce_ordinal.fit_transform(X_train_cat).fillna(-1).astype("int")
X_test_cat_gbm = ce_ordinal.transform(X_test_cat).fillna(-1).astype("int")

X_train_gbm = pd.concat([X_train_cat_gbm, X_train_num],1)
X_test_gbm = pd.concat([X_test_cat_gbm, X_test_num],1)

y_train_gbm = y_train.status_group.astype('category').cat.codes
y_test_gbm = y_test.status_group.astype('category').cat.codes
#%% I use the Optuna framework to tune hyperparameters of the LightGBM model. 
# The objective is to maximixe the crossvalidated multi_logloss. Optuna has a wrapper for LightGBM,
# so we do not have to manually suggest hyperparameter settings.

dtrain = lgb.Dataset(X_train_gbm, label=y_train_gbm, categorical_feature=X_train_cat_gbm.columns.tolist())
dtrain = lgb.Dataset(X_train_gbm, label=(y_train_gbm==1).astype(int), categorical_feature=X_train_cat_gbm.columns.tolist())

params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": 3,
    "metric": ["multi_logloss"],
    "class_weight": "balanced",
    "verbosity": -1,
}

# params = {
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": ["binary_logloss"],
#     "is_unbalance": "true",
#     "verbosity": -1,
# }

# params = {
#     "boosting_type": "gbdt",
#     "objective": "regression",
#     "metric": "l2",
#     "verbosity": -1,
# }


# tuner = optuna_lgb.LightGBMTunerCV(
#     params, dtrain, verbose_eval=0, early_stopping_rounds=20, folds=StratifiedKFold(n_splits=3, shuffle=True), return_cvbooster=True,time_budget=300
#     )
tuner = optuna_lgb.LightGBMTunerCV(
    params, dtrain, verbose_eval=0, early_stopping_rounds=20, nfold = 3, stratified =True, return_cvbooster=True, time_budget=300
    )


tuner.run()

print("Best score:", tuner.best_score)
best_params = tuner.best_params
print("Best params:", best_params)
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))
#%% Analyze tuning history

slice_fig = optuna.visualization.plot_slice(tuner.study)
slice_fig.write_html('slice_fig.html', auto_open=True)

history_fig = optuna.visualization.plot_optimization_history(tuner.study)
history_fig.write_html('history_fig.html', auto_open=True)

#%% Use the best parameters found in the previous step to fit the LightGBM on the whole training set 
# and finally test the performance on the test set.

gbm = lgb.train(best_params, dtrain, verbose_eval=0)
preds = gbm.predict(X_test_gbm)
y_pred_gbm = preds.argmax(1).astype(int)
# y_pred_gbm = np.rint(preds).astype(int)

accuracy = accuracy_score(y_test_gbm, y_pred_gbm)
print(accuracy)

cr = classification_report(y_test_gbm, y_pred_gbm)
print(cr)

# plot confusion matrix
cm = confusion_matrix(y_test_gbm, y_pred_gbm) 
ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
ax.set(xlabel='Predicted', ylabel='True')

# Despite taking precautions, our model has not been able to deal with the imbalance of output labels. 
# Some waterpumps that are in need of repair are wrongly classified as functional.

# Save the LightGBM model
with open('models/lgb_classifier.pkl', 'wb') as f:
    pickle.dump(gbm, f)    

#%%
X_plotly = X_train_gbm.copy()
X_plotly = pd.concat([X_plotly, y_train], axis=1)

X_plotly.dropna(subset=["longitude","latitude"], inplace=True)

# Save dataframes for plotly dashboard
with open('data/plotly_df.pkl', 'wb') as f:
    pickle.dump(X_plotly, f)
with open('data/original_df.pkl', 'wb') as f:
    pickle.dump(original_df.loc[X_plotly.index], f)       
#%%


# plot builtin feature importance
optuna_lgb.plot_importance(gbm)

# Use SHAP library for more advanced model interpretation
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X_train_gbm)

shap.summary_plot(shap_values, X_train_gbm)
shap.plots.waterfall(explainer.shap_values(X_train_gbm))

with open('models/shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)    

shap.initjs()
plot = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=False)
shap.save_html("index.htm" , plot)

shap.plots.beeswarm(shap_values)


 
#%%
stophier
filename = "/models/lgb_classifier"
pickle.dump(gbm, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))


    
with open('models/lgb_classifier.pkl', 'rb') as f:
    testgbm = pickle.load(f)    
    
clf = lgb.LGBMClassifier(max_depth=10)
clf.fit(X_train_gbm, y_train.status_group.astype('category').cat.codes)

scores = cross_val_score(clf, X_train_gbm, y_train.values.ravel(), cv=5)
print("%0.3f accuracy with a standard deviation of %0.5f" % (scores.mean(), scores.std()))

lgb.plot_importance(clf)

shap_values = shap.TreeExplainer(clf).shap_values(X_train_gbm)
shap.summary_plot(shap_values, X_train_gbm)

shap.initjs()
plot = shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=False)
shap.save_html("index.htm" , plot)
shap.plots.beeswarm(shap_values)

