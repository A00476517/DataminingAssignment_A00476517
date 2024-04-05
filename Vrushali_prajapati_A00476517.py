import warnings
warnings.filterwarnings("ignore")

# Importing necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Setting up seaborn for visualization
sns.set(style='white', context='notebook', palette='deep')

# Loading the data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
combine = pd.concat([train.drop('Survived', axis=1), test])

# Printing summary statistics of the train dataset
train.head(8)
train.describe()

# Checking for missing values in train and test datasets
print(train.isnull().sum())
print(test.info())

# Visualizing survival distribution
surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "blue"
nosurv_col = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"\
      %(len(surv), 1.*len(surv)/len(train)*100.0,\
        len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))

# Visualizing various features
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(nosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col,
            axlabel='Age')
plt.subplot(332)
sns.barplot(x='Sex', y='Survived', data=train, palette='coolwarm')
plt.subplot(333)
sns.barplot(x='Pclass', y='Survived', data=train, palette='coolwarm')
plt.subplot(334)
sns.barplot(x='Embarked', y='Survived', data=train, palette='coolwarm')
plt.subplot(335)
sns.barplot(x='SibSp', y='Survived', data=train, palette='coolwarm')
plt.subplot(336)
sns.barplot(x='Parch', y='Survived', data=train, palette='coolwarm')
plt.subplot(337)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col,axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

# Further analysis
print("Median age survivors: %.1f, Median age non-survivors: %.1f"\
      %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))
tab = pd.crosstab(train['SibSp'], train['Survived'])

# Heatmap of numeric columns correlation
numeric_columns = train.drop(['PassengerId', 'Name'], axis=1).select_dtypes(include=np.number)
plt.figure(figsize=(14,12))
foo = sns.heatmap(numeric_columns.corr(), vmax=0.6, square=True, annot=True, cmap='coolwarm')

# Feature engineering
combine = pd.concat([train.drop('Survived',axis=1),test])
survived = train['Survived']
combine['Child'] = combine['Age']<=10
combine['Family'] = combine['SibSp'] + combine['Parch']
test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

# Visualization based on new features
g = sns.catplot(x="Sex", y="Survived", hue="Child", col="Pclass",
                data=train, kind="point", aspect=0.9, height=3.5, ci=95.0)

# More cross-tabulations
tab = pd.crosstab(train['Child'], train['Pclass'])
print(tab)
tab = pd.crosstab(train['Child'], train['Sex'])
print(tab)
tab = pd.crosstab(train['Family'], train['Survived'])

# Visualizing Family vs Survival
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Family members')
dummy = plt.ylabel('Percentage')

# Train-test split
training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
     %(train.shape[0],training.shape[0],testing.shape[0]))

# Selecting features and target variable
cols = ['Sex','Pclass','Parch','SibSp','Child']
X_train = training.loc[:, cols].dropna()
y_train = np.ravel(training.loc[:, ['Survived']])

X_test = testing.loc[:, cols].dropna()
y_test = np.ravel(testing.loc[:, ['Survived']])

# One-hot encoding categorical features
X_train = pd.get_dummies(X_train, columns=['Sex', 'Pclass', 'Child'])
X_test = pd.get_dummies(X_test, columns=['Sex', 'Pclass', 'Child'])

# Initializing and fitting Logistic Regression model
clf_log = LogisticRegression()
clf_log.fit(X_train, y_train)

# Predictions on test data
predictions = clf_log.predict(X_test)

# Model evaluation
accuracy = clf_log.score(X_test, y_test)
print("Accuracy with LogisticRegression:", accuracy)

# Decision Tree Classifier
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)
predictions = clf_tree.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy with DecisionTreeClassifier:", accuracy)

# Random Forest Classifier
clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf_rf.fit(X_train, y_train)
predictions = clf_rf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy with Random Forest:", accuracy)
