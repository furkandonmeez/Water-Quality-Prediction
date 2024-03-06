# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:36:46 2023

@author: hp
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('water_potability.csv')

print(df['Potability'].describe())

#histogram
sns.distplot(df['Potability'])
#plt.savefig('potability_histogram.png', dpi=300)

#scatter plot
var = 'ph'
data = pd.concat([df['Potability'], df[var]], axis=1)
data.plot.scatter(x=var, y='Potability');
#plt.savefig('scatter_plot.png', dpi=300)



#*******************************************************************


df['ph'].fillna(df['ph'].mean(), inplace=True)
df['Sulfate'].fillna(df['Sulfate'].mean(), inplace=True)
df['Trihalomethanes'].fillna(df['Trihalomethanes'].mean(), inplace=True)


#Standarization
scaler = StandardScaler()
X = scaler.fit_transform(df)
X = pd.DataFrame(X, columns=df.columns)


# correlation matrix
corrmat = df.corr()
cols = corrmat.nlargest(10, 'Potability')['Potability'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
sns.heatmap(cm, annot=True,cmap='RdBu', fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# Spearman korelasyon matrisi
spearman_corr = df.corr(method='spearman')

# Korelasyon matrisi
sns.heatmap(spearman_corr, annot=True, cmap='RdBu', fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


# Kendall korelasyon matrisi
kendall_corr = df.corr(method='kendall')

# Korelasyon matrisi
sns.heatmap(kendall_corr, annot=True, cmap='RdBu', fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)


#scatterplot
sns.set()
cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity', 'Potability']
sns.pairplot(df[cols], size = 2.5)
#plt.savefig('havalı_grafik.png', dpi=300)
plt.show();


# Smooth curve ile scatterplot oluşturma
sns.set()
sns.lmplot(x='Sulfate', y='Solids', data=df, ci=None , line_kws={'color': 'darkblue'})  # ph ve Hardness değişkenleri için smooth curve çiziyoruz
plt.show()


import statsmodels.api as sm

for column in df.columns[:-1]:  # Potability dışındaki sütunlar için
    # Loess eğrisini hesapla
    lowess = sm.nonparametric.lowess(df['Potability'], df[column], frac=0.3)  # Frac değeri loess pürüzsüzlüğünü ayarlar

    # Veriyi ve loess eğrisini görselleştir
    plt.figure(figsize=(8, 6))
    plt.scatter(df[column], df['Potability'], label='Data')
    plt.plot(lowess[:, 0], lowess[:, 1], 'r-', label='Loess Curve', linewidth=2)
    plt.xlabel(column)
    plt.ylabel('Potability')
    plt.title(f'Loess Curve for {column} vs. Potability')
    plt.legend()
    plt.show()
    
    
    
#********************************************

#define x and y
y = df["Potability"]
X = df.drop(columns=["Potability"])

# import train-test split 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)




""" Using LOGISTIC REGRESSSION """
from sklearn.linear_model import LogisticRegression


# Create a Logistic Regression model
model_lg = LogisticRegression()

# Define the grid of hyperparameters to search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2'],             # Penalty norm
    'solver': ['liblinear', 'saga']      # Solver for optimization
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model_lg, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score_lg = grid_search.best_score_

print("Best Parameters for Logistic Regression:", best_params)
print("Best Score for these parameters:", best_score_lg)

# Perform cross-validation with 5 folds
cv_scores_lg = cross_val_score(model_lg, X_train, y_train, cv=5)

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores_lg)
print("Mean CV Accuracy:", cv_scores_lg.mean())

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy with Best Model:", test_accuracy) 
#This accuracy score on the test set represents how well the model is expected to perform on new, unseen data.






""" Using DECISION TREE CLASSIFIER """
from sklearn.tree import DecisionTreeClassifier
# Create a Decision Tree model
model_dt = DecisionTreeClassifier()

# Define the grid of hyperparameters to search
param_grid = {
    'criterion': ['gini', 'entropy'],       # Criterion for splitting
    'max_depth': [None, 5, 10, 15, 20],     # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],        # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]           # Minimum samples required at each leaf node
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model_dt, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score_dt = grid_search.best_score_

print("Best Parameters for Decision Tree Classifier:", best_params)
print("Best Score for these parameters:", best_score_dt)

# Perform cross-validation with 5 folds
cv_scores_dt = cross_val_score(model_dt, X_train, y_train, cv=5)

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores_dt)
print("Mean CV Accuracy:", cv_scores_dt.mean())

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy with Best Model:", test_accuracy)





""" Using RANDOM FOREST """
from sklearn.ensemble import RandomForestClassifier

# Creating model object
model_rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=0.16, random_state=42)

# Training Model
model_rf.fit(X_train, y_train)

# Making Prediction
pred_rf = model_rf.predict(X_test)

# Calculating Accuracy Score
rf = accuracy_score(y_test, pred_rf)
print("Random Forest accuracy score: " ,rf)

print(classification_report(y_test,pred_rf))




""" Using KNeighbours """
from sklearn.neighbors import KNeighborsClassifier
# Create a KNeighborsClassifier model
model_kn = KNeighborsClassifier()

# Define the grid of hyperparameters to search
param_grid = {
    'n_neighbors': [3, 5, 7, 9],          # Number of neighbors to consider
    'weights': ['uniform', 'distance'],   # Weight function used in prediction
    'p': [1, 2]                           # Power parameter for the Minkowski metric
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model_kn, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score_kn = grid_search.best_score_

print("Best Parameters for KNeighbours:", best_params)
print("Best Score for these parameters:", best_score_kn)

# Perform cross-validation with 5 folds
cv_scores_kn = cross_val_score(model_kn, X_train, y_train, cv=5)

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores_kn)
print("Mean CV Accuracy:", cv_scores_kn.mean())

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy with Best Model:", test_accuracy)


""" Using ADABOOST CLASSIFIER """
from sklearn.ensemble import AdaBoostClassifier
# Create an AdaBoostClassifier model
model_ada = AdaBoostClassifier()

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 200],     # Number of weak learners
    'learning_rate': [0.01, 0.1, 1.0]   # Learning rate
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model_ada, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score_ada = grid_search.best_score_

print("Best Parameters for AdaBoost Classifier:", best_params)
print("Best Score for these parameters:", best_score_ada)

# Perform cross-validation with 5 folds
cv_scores_ada = cross_val_score(model_ada, X_train, y_train, cv=5)

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores_ada)
print("Mean CV Accuracy:", cv_scores_ada.mean())

# Evaluate the model with best parameters on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy with Best Model:", test_accuracy)



""" Using SVM """
from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', random_state = 42)

model_svm.fit(X_train, y_train)

# Making Prediction
pred_svm = model_svm.predict(X_test)

# Calculating Accuracy Score
sv = accuracy_score(y_test, pred_svm)
print("SVM accuracy score: " ,sv)

print(classification_report(y_test,pred_svm))



"""XGBoost"""
import xgboost as xgb

# Creating an XGBoost model
model_xgb = xgb.XGBClassifier()

# Defining parameter grid for GridSearchCV
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'gamma': [0, 0.1, 0.2],
}

# Performing cross-validation with GridSearchCV
grid_search_xgb = GridSearchCV(estimator=model_xgb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)

# Getting the best parameters and best score
best_params_xgb = grid_search_xgb.best_params_
best_score_xgb = grid_search_xgb.best_score_

print("Best Parameters for XGBoost:", best_params_xgb)
print("Best score for these parameters:", best_score_xgb)

# Evaluating the best model on the test set
best_model_xgb = grid_search_xgb.best_estimator_
y_pred = best_model_xgb.predict(X_test)
test_accuracy_xgb = accuracy_score(y_test, y_pred)
print("Test Accuracy with Best Model:", test_accuracy_xgb)




"""Using Gaussian Naive Bayes"""
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

# Create a Gaussian Naive Bayes model
model_nb = GaussianNB()

# Define the grid of hyperparameters to search
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Additive (Laplace/Lidstone) smoothing parameter
}

# Perform grid search with cross-validation
grid_search_nb = GridSearchCV(estimator=model_nb, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search_nb.fit(X_train, y_train)

# Get the best parameters and the best score
best_params_nb = grid_search_nb.best_params_
best_score_nb = grid_search_nb.best_score_

print("Best Parameters for Gaussian Naive Bayes:", best_params_nb)
print("Best Score for these parameters:", best_score_nb)

# Perform cross-validation with 5 folds
cv_scores_nb = cross_val_score(grid_search_nb.best_estimator_, X_train, y_train, cv=5)

# Print the cross-validation scores
print("\nCross-validation scores:", cv_scores_nb)
print("Mean CV Accuracy:", cv_scores_nb.mean())

# Evaluate the model with best parameters on the test set
best_model_nb = grid_search_nb.best_estimator_
test_accuracy_nb = best_model_nb.score(X_test, y_test)
print("Test Accuracy with Best Gaussian Naive Bayes:", test_accuracy_nb)




""" models """
# Verileri oluşturun
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'KNeighbours',  'AdaBoost',  'Gaussian Naive Bayes', 'XGBoost', 'SVM', 'Random Forest'],
    'Accuracy_score': [best_score_lg, best_score_dt, best_score_kn,  best_score_ada, best_score_nb, best_score_xgb, sv, rf]
})

# Verileri doğru şekilde sıralayın
sorted_models = models.sort_values(by='Accuracy_score', ascending=False)
 
# Çubuk grafiğini çizin
plt.figure(figsize=(8, 6))
sns.barplot(x='Accuracy_score', y='Model', data=sorted_models, palette='viridis')
plt.xlabel('Accuracy Score')
plt.title('Model Accuracies')
plt.tight_layout()  
plt.savefig('model_oranları.png', dpi=300)
plt.show()





# Set up the best classifier
xgb_best_model = xgb.XGBClassifier(gamma=0.1, learning_rate=0.01, max_depth=5, n_estimators=300)

# Train the model
xgb_best_model.fit(X_train, y_train)

# Predict on unseen data
y_pred = xgb_best_model.predict(X_test)

# Get the score of the model
accuracy = accuracy_score(y_test, y_pred, normalize=True) * 100.0
print('Correct Prediction (%): ', accuracy)

# Get feature importance
feature_imp = pd.Series(xgb_best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

# Create subplot and pie chart
plt.figure(figsize=(8, 8))  # Set the figure size

# Plot the pie chart
plt.pie(feature_imp, colors=sns.cubehelix_palette(start=.5, rot=-.75, n_colors=9), labels=feature_imp.index,
        autopct='%1.1f%%', startangle=0, rotatelabels=False)

# Draw circle
centre_circle = plt.Circle((0, 0), 0.75, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')

# Set title and adjust layout
plt.title(f'Feature Importance in XGBoost\nAccuracy: {accuracy:.2f}%')
plt.subplots_adjust(top=4)  # Adjust the space between the title and the pie chart

plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

print(best_score_lg, best_score_dt, best_score_kn,  best_score_ada, best_score_nb, best_score_xgb, sv, rf)


