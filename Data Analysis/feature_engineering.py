## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import re

from data_management import data_manage

pd.set_option('display.max_columns', 10)

# 加载数据集
datapath = '../dataset/clean_data/clean_merged_data.csv'
savepath = '../outputImg/hospital_data/'
data = data_manage(datapath)
data1 = data.copy()
print(data1.head())

# 按日期降序排序
data_sorted = data1.sort_values('date', ascending=False)

# 获取每个国家的最新数据
latest_data = data_sorted.groupby('entity').first().reset_index()

latest_data = latest_data.rename(columns={"mortality_quartile": "Y"})

subset1 = latest_data.copy()

#Univariate plot
bc1 = sns.countplot(x='Y', data=subset1)
plt.xlabel('New Deaths Number')
plt.title('Daily Deaths rates ')
plt.show()

#count of Y variable
print('count of Y Variable')
c1 = subset1.groupby('Y').size()
print(c1)

#percentages for Y variable
print('percentages of Y variable counts')
p1 = subset1.groupby('Y').size() * 100 / len(subset1)
print(p1)

#Distplot for variable AGE
#numerical

bc2 = sns.displot(subset1.hospital_beds_per_thousand.dropna(), kde=True)
bc2.set_titles('Hospital beds per thousand distribution')
bc2.set_axis_labels('hospital beds distribute', 'Count')
plt.show()

bc2 = sns.displot(subset1.population.dropna(), kde=True)
bc2.set_titles('Population distribution')
bc2.set_axis_labels('population distribution', 'Count')
plt.show()

#Create a figure of a distplot and a boxplot for age
#Create a figure of a distplot and a boxplot for age
fig, ax = plt.subplots(1, 2)
fig.suptitle('Population distribution and outliers')
ax[0].title.set_text('distribution')
sns.histplot(subset1['population'].dropna(), kde=True, ax=ax[0])
plt.xlabel('population')
plt.title('')
plt.xticks(rotation=45)

ax[1].title.set_text('Outliers')
tmp_dtf = pd.DataFrame(subset1['population'])
#tmp_dtf['AGE'] = np.log(tmp_dtf['AGE'])
tmp_dtf.boxplot(column='population', ax=ax[1])
plt.show()

# subset1_optimized = subset1.copy()

#Bivariate plots
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(28, 16))
fig.suptitle('hospital_beds_per_thousand   vs   total Deaths', fontsize=20)

### distribution
ax[0].title.set_text('density')
sns.histplot(subset1, x="hospital_beds_per_thousand", multiple="stack", hue='Y', element="step", ax=ax[0], legend=False)
ax[0].grid(True)

ax[1].title.set_text('outliers')
sns.boxplot(x=subset1['Y'], y=subset1['hospital_beds_per_thousand'], data=subset1, whis=np.inf, ax=ax[1])
ax[1].set_xlabel('hospital_beds_per_thousand')
ax[1].grid(True)

plt.xticks(rotation=45)

subset1['Ystring'] = subset1['Y'].apply(str)
ax[2].title.set_text('bins')
ax[2] = sns.countplot(data=subset1, x=subset1['deaths_quartile'], hue=subset1['Ystring'], ax=ax[2])
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=40, )
ax[2].grid(True)
ax[2].get_legend().remove()

plt.show()

#Bivariate plots
fig, ax = plt.subplots(nrows=1, ncols=3, sharex=False, sharey=False, figsize=(28, 16))
fig.suptitle('population   vs   total Deaths', fontsize=20)

### distribution
ax[0].title.set_text('density')
sns.histplot(subset1, x="population_density", multiple="stack", hue='Y', element="step", ax=ax[0], legend=False)
ax[0].grid(True)

ax[1].title.set_text('outliers')
sns.boxplot(x=subset1['Y'], y=subset1['population_density'], data=subset1, whis=np.inf, ax=ax[1])
ax[1].set_xlabel('population')
ax[1].grid(True)

plt.xticks(rotation=45)

subset1['Ystring'] = subset1['Y'].apply(str)
ax[2].title.set_text('bins')
ax[2] = sns.countplot(data=subset1, x=subset1['deaths_quartile'], hue=subset1['Ystring'], ax=ax[2])
ax[2].set_xticklabels(ax[2].get_xticklabels(), rotation=40, )
ax[2].grid(True)
ax[2].get_legend().remove()

plt.show()

'''
ANOVA
'''

#Test using ANOVA to see if the relationship is significant.

model = smf.ols('Y ~ hospital_beds_per_thousand', data=subset1).fit()
print(model.summary())

#To make interpretation easy you can display the following:
table = sm.stats.anova_lm(model)
print(table)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
print("Anova F: the variables are", conclusion, "(p-value: " + str(p) + ")")

model = smf.ols('Y ~ population_density', data=subset1).fit()
print(model.summary())

#To make interpretation easy you can display the following:
table = sm.stats.anova_lm(model)
print(table)
p = table["PR(>F)"][0]
coeff, p = None, round(p, 3)
conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
print("Anova F: the variables are", conclusion, "(p-value: " + str(p) + ")")

'''
Data Classification
'''

#Next we start to preprocess the dataset ready for our machine learning model.

#first we create a subset without the columns passengerid, name, ticket, cabin, and agegroup as none of these are to be used as predictors.
dtf = subset1[['Y', 'hospital_beds_per_thousand', 'population_density', 'population', 'entity', 'continent']]

#partitioning the dataset into train and test sets
dtf_train, dtf_test = model_selection.train_test_split(dtf,
                                                       test_size=0.3)
## print info
print("X_train shape:", dtf_train.drop("Y", axis=1).shape, "| X_test shape:", dtf_test.drop("Y", axis=1).shape)
print("y_train mean:", round(np.mean(dtf_train["Y"]), 2), "| y_test mean:", round(np.mean(dtf_test["Y"]), 2))
print(dtf_train.shape[1], "features:", dtf_train.drop("Y", axis=1).columns.to_list())

#replace missing data in AGE with the average of the age in the training and test set.
dtf_train.isnull().sum()
dtf_test.isnull().sum()
dtf_train.loc[dtf_train['hospital_beds_per_thousand'].isnull(), ['hospital_beds_per_thousand']] = dtf_train[
    'hospital_beds_per_thousand'].mean()
dtf_test.loc[dtf_test['hospital_beds_per_thousand'].isnull(), ['hospital_beds_per_thousand']] = dtf_test[
    'hospital_beds_per_thousand'].mean()
#show that the missing values in AGE have been replaced
dtf_train.isnull().sum()
dtf_test.isnull().sum()

'''
feature engineering
'''

# All further data preprocessing must be carried out on both the train set and the test set.

# train dataset
# encoding of categorical variable into a binary vector
# create dummy for population_density, continent and entity
dummy = pd.get_dummies(dtf_train["population_density"],
                       prefix="population_density", drop_first=True)
dtf_train = pd.concat([dtf_train, dummy], axis=1)
print(dtf_train.filter(like="population_density", axis=1).head())

# create a dummy for continent
print(dtf['continent'].value_counts())
dummy = pd.get_dummies(dtf_train["continent"],
                       prefix="continent", drop_first=True)
dtf_train = pd.concat([dtf_train, dummy], axis=1)
print(dtf_train.filter(like="continent", axis=1).head())

# create a dummy for EMBARKED
print(dtf['hospital_beds_per_thousand'].value_counts())
dummy = pd.get_dummies(dtf_test["hospital_beds_per_thousand"],
                       prefix="hospital_beds_per_thousand", drop_first=True)
dtf_test = pd.concat([dtf_test, dummy], axis=1)
print(dtf_test.filter(like="hospital_beds_per_thousand", axis=1).head())

# create a dummy for entity
print(dtf['entity'].value_counts())
dummy = pd.get_dummies(dtf_train["entity"],
                       prefix="entity", drop_first=True)
dtf_train = pd.concat([dtf_train, dummy], axis=1)
print(dtf_train.filter(like="entity", axis=1).head())

# drop the original categorical columns
dtf_train.drop(columns=['entity', 'continent'], inplace=True, axis=1)
print(dtf_train.head())

# test dataset
# encoding of categorical variable into a binary vector
# create dummy for population_density, continent and entity
dummy = pd.get_dummies(dtf_test["population_density"],
                       prefix="population_density", drop_first=True)
dtf_test = pd.concat([dtf_test, dummy], axis=1)
print(dtf_test.filter(like="population_density", axis=1).head())

# create a dummy for EMBARKED
print(dtf['continent'].value_counts())
dummy = pd.get_dummies(dtf_test["continent"],
                       prefix="continent", drop_first=True)
dtf_test = pd.concat([dtf_test, dummy], axis=1)
print(dtf_test.filter(like="continent", axis=1).head())

# create a dummy for EMBARKED
print(dtf['hospital_beds_per_thousand'].value_counts())
dummy = pd.get_dummies(dtf_test["hospital_beds_per_thousand"],
                       prefix="hospital_beds_per_thousand", drop_first=True)
dtf_test = pd.concat([dtf_test, dummy], axis=1)
print(dtf_test.filter(like="hospital_beds_per_thousand", axis=1).head())

# create a dummy for CABIN_SECTION
print(dtf['entity'].value_counts())
dummy = pd.get_dummies(dtf_test["entity"],
                       prefix="entity", drop_first=True)
dtf_test = pd.concat([dtf_test, dummy], axis=1)
print(dtf_test.filter(like="entity", axis=1).head())

# drop the original categorical columns
dtf_test.drop(columns=['entity', 'continent'], inplace=True, axis=1)
print(dtf_test.head())

# Scaling the features in train and test data sets
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(dtf_train.drop(columns=["Y"], axis=1))
dtf_scaled = pd.DataFrame(X, columns=dtf_train.drop(columns=["Y"], axis=1).columns, index=dtf_train.index)
dtf_scaled["Y"] = dtf_train["Y"]
dtf_scaled.head()

scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(dtf_test.drop(columns=["Y"], axis=1))
dtf_scaled = pd.DataFrame(X, columns=dtf_test.drop(columns=["Y"], axis=1).columns, index=dtf_test.index)
dtf_scaled["Y"] = dtf_test["Y"]
dtf_scaled.head()

# Feature Selection using Lasso regularisation
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns
# Anova
selector = feature_selection.SelectKBest(score_func=feature_selection.f_classif, k=10).fit(X, y)
anova_selected_features = feature_names[selector.get_support()]

# Lasso regularization
selector = feature_selection.SelectFromModel(estimator=linear_model.LogisticRegression(C=1,
                                                                                       penalty="l1",
                                                                                       solver='liblinear'),
                                             max_features=10
                                             ).fit(X, y)
lasso_selected_features = feature_names[selector.get_support()]

# Plot the barplot showing the feature selection
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, figsize=(8, 6))
fig.suptitle('Lasso vs Anova', fontsize=20)
dtf_features = pd.DataFrame({"features": feature_names})
dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
dtf_features["method"] = dtf_features[["anova", "lasso"]].apply(lambda x: (x[0] + " " + x[1]).strip(), axis=1)
dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
sns.barplot(ax=ax, y="features", x="selection", hue="method",
            data=dtf_features.sort_values("selection", ascending=False), dodge=False)
plt.show()

# Ensemlbe method - Random Forest
X = dtf_train.drop("Y", axis=1).values
y = dtf_train["Y"].values
feature_names = dtf_train.drop("Y", axis=1).columns.tolist()

# Importance - random forests classifier
model = ensemble.RandomForestClassifier(n_estimators=100,
                                        criterion="entropy", random_state=0)
model.fit(X, y)
importances = model.feature_importances_

# Put in a pandas dtf, 2 variables in the data frame, IMPORTANCE and VARIABLE sorted by importance.
dtf_importances = (pd.DataFrame({"IMPORTANCE": importances,
                                 "VARIABLE": feature_names})
                   .sort_values("IMPORTANCE", ascending=False))
# cummulative sum
dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
# set the index to VARIABLE column
dtf_importances = dtf_importances.set_index("VARIABLE")

# Plot on a fig with two columns
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=(8, 6))
fig.suptitle("Features Importance", fontsize=20)

# first chart is a bar plotting variables by importance
ax[0].title.set_text('variables')
dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(
    kind="barh", legend=False, ax=ax[0]).grid(axis="x")
ax[0].set(ylabel="")

# second chart is a line showing cumullative importance.
ax[1].title.set_text('cumulative')
dtf_importances[["cumsum"]].plot(kind="line", linewidth=4,
                                 legend=False, ax=ax[1])
ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)),
          xticklabels=dtf_importances.index)
plt.xticks(rotation=70)
plt.grid(axis='both')
plt.show()

# Select the following features, train and test
print(list(dtf_train.columns))
search_keywords = ['hospital_beds_per_thousand', 'population_density', 'population', 'continent', 'entity']
train_X_names = []
test_X_names = []
for keyword in search_keywords:
    train_X_names.extend([col for col in dtf_train.columns if re.search(keyword, col)])

# 去除重复的列名
train_filtered_columns = list(set(train_X_names))

for keyword in search_keywords:
    test_X_names.extend([col for col in dtf_test.columns if re.search(keyword, col)])

# 去除重复的列名
test_filtered_columns = list(set(test_X_names))

X_train = dtf_train[train_X_names].values
y_train = dtf_train["Y"].values
X_test = dtf_test[test_X_names].values
y_test = dtf_test["Y"].values

# Designing the model requires figuring out what parameter settings to use
# RandomSearch helps to find the best combination of parameters and puts them into the model using the X_train and Y_train datasets
model = ensemble.GradientBoostingClassifier()
# define hyperparameters combinations to try
param_dic = {'learning_rate': [0.15, 0.1, 0.05, 0.01, 0.005, 0.001],
             # weighting factor for the corrections by new trees when added to the model
             'n_estimators': [100, 250, 500, 750, 1000, 1250, 1500, 1750],  # number of trees added to the model
             'max_depth': [2, 3, 4, 5, 6, 7],  # maximum depth of the tree
             'min_samples_split': [2, 4, 6, 8, 10, 20, 40, 60, 100],  # sets the minimum number of samples to split
             'min_samples_leaf': [1, 3, 5, 7, 9],  # the minimum number of samples to form a leaf
             'max_features': [2, 3, 4, 5, 6, 7],  # square root of features is usually a good starting point
             'subsample': [0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
                           1]}  # the fraction of samples to be used for fitting the individual base learners. Values
# lower than 1 generally lead to a reduction of variance and an increase in bias.
# random search
random_search = model_selection.RandomizedSearchCV(model,
                                                   param_distributions=param_dic, n_iter=100,
                                                   scoring="accuracy").fit(X_train, y_train)
print("Best Model parameters:", random_search.best_params_)
print("Best Model mean accuracy:", random_search.best_score_)
model = random_search.best_estimator_

# use the model now
model.fit(X_train, y_train)
# test
predicted_prob = model.predict_proba(X_test)[:, 1]
predicted = model.predict(X_test)

#Evaluation: AUC, Precision and Recall
## Accuray e AUC
accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob)
print("Accuracy (overall correct predictions):",  round(accuracy,2))
print("Auc:", round(auc,2))

## Precision and Recall
recall = metrics.recall_score(y_test, predicted)
precision = metrics.precision_score(y_test, predicted)
print("Recall (all 1s predicted right):", round(recall,2))
print("Precision (confidence when predicting a 1):", round(precision,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in np.unique(y_test)]))

#confusion matrix
#confusion matrix
classes = np.unique(y_test)
cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

# plot
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax)

# 设置标题和标签
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')

# 显示图片
plt.show()

# test the prediction on random observation from the test set and see what is predicted.
print("True:", y_test[2], "--> Pred:", predicted[2], "| Prob:", np.max(predicted_prob[2]))
print("True:", y_test[3], "--> Pred:", predicted[3], "| Prob:", np.max(predicted_prob[3]))
