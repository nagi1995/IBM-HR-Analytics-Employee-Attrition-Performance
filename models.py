#!/usr/bin/env python
# coding: utf-8

# In[88]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# In[2]:


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# In[4]:


df["attrition"] = df["Attrition"]

"""
removing "MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrentManager" 
because their correlation with "JobLevel" and "YearsAtCompany" >= 0.75 [found this in EDA.ipynb]
"""
df.drop(columns = ["Attrition", "MonthlyIncome", "TotalWorkingYears", "YearsInCurrentRole", "YearsWithCurrManager"], inplace = True)
df.head()


# In[5]:


df.nunique()


# In[6]:


"""
EmployeeCount, Over18, StandardHours has only one unique value. So this variables can be ignored/removed.
"""
df.drop(columns = ["EmployeeCount", "Over18", "StandardHours"], inplace = True)
df.head()


# In[8]:


df["attrition"] = df["attrition"].astype('category').cat.codes
df["BusinessTravel"] = df["BusinessTravel"].astype('category').cat.codes
df["Department"] = df["Department"].astype('category').cat.codes
df["EducationField"] = df["EducationField"].astype('category').cat.codes
df["Gender"] = df["Gender"].astype('category').cat.codes
df["JobRole"] = df["JobRole"].astype('category').cat.codes
df["MaritalStatus"] = df["MaritalStatus"].astype('category').cat.codes
df["OverTime"] = df["OverTime"].astype('category').cat.codes
df


# In[9]:


x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
x.shape, y.shape


# In[15]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled.shape


# # Logistic Regression

# ### without SMOTE

# In[61]:


precision_recall_fscore_supports = []
model = LogisticRegression()
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[62]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# ### with SMOTE

# In[59]:


precision_recall_fscore_supports = []
model = LogisticRegression()
cv = KFold(n_splits = 10, random_state = 0)
smote = SMOTE()

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    model.fit(x_smote, y_smote)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[60]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# # SVM 

# ### without SMOTE

# In[66]:


precision_recall_fscore_supports = []
model = SVC(kernel = "rbf", class_weight = "balanced", probability = True)
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[67]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# ### with SMOTE

# In[70]:


precision_recall_fscore_supports = []
model = SVC(kernel = "rbf")
cv = KFold(n_splits = 10, random_state = 0)
smote = SMOTE()

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    model.fit(x_smote, y_smote)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[71]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# # Decision Tree

# ### without SMOTE

# In[73]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 0)
x_train.shape, x_test.shape


# In[77]:


# pre prunning 
model = DecisionTreeClassifier(random_state = 0)
param_grid = {"max_depth" : [8, 12, 16, 20]}

gs = GridSearchCV(estimator = model, param_grid = param_grid)
gs.fit(x_train, y_train)
print("best parameters:", gs.best_estimator_)
print("best score:", gs.best_score_*100)


# In[78]:


precision_recall_fscore_supports = []
model = DecisionTreeClassifier(max_depth = 8, class_weight = "balanced")
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[79]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# ### with SMOTE

# In[80]:


precision_recall_fscore_supports = []
model = DecisionTreeClassifier(max_depth = 8, class_weight = "balanced")
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    model.fit(x_smote, y_smote)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[81]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# # Random Forest

# ### without SMOTE

# In[82]:


x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.3, random_state = 0)
x_train.shape, x_test.shape


# In[83]:


# pre prunning 
model = RandomForestClassifier()
param_grid = {"max_depth" : [8, 12, 16, 20],
             "class_weight" : ["balanced", "balanced_subsample"]}

gs = GridSearchCV(estimator = model, param_grid = param_grid)
gs.fit(x_train, y_train)
print("best parameters:", gs.best_estimator_)
print("best score:", gs.best_score_*100)


# In[84]:


precision_recall_fscore_supports = []
model = RandomForestClassifier(max_depth = 8, class_weight = "balanced")
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[85]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# ### with SMOTE

# In[86]:


precision_recall_fscore_supports = []
model = RandomForestClassifier(max_depth = 8, class_weight = "balanced")
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    model.fit(x_smote, y_smote)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[87]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# # XGBoost

# ### without SMOTE

# In[89]:


precision_recall_fscore_supports = []
model = XGBClassifier(n_estimators = 100, max_depth = 8)
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model.fit(x_train, y_train)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[90]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# ### with SMOTE

# In[93]:


precision_recall_fscore_supports = []
model = XGBClassifier(n_estimators = 100, max_depth = 8)
cv = KFold(n_splits = 10, random_state = 0)

for train_index, test_index in cv.split(x_scaled):
    #print(train_index, test_index)
    x_train, x_test = x_scaled[train_index], x_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    model.fit(x_smote, y_smote)
    precision_recall_fscore_supports.append(precision_recall_fscore_support(y_test, model.predict(x_test), average = "binary"))


# In[94]:


f_scores = []
for prfs in precision_recall_fscore_supports:
    print(prfs)
    f_scores.append(prfs[2])
print(np.mean(np.array(f_scores)) * 100)


# In[ ]:




