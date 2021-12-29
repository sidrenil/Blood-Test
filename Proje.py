#!/usr/bin/env python
# coding: utf-8

# In[115]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
diabetes = pd.read_csv('diabetes.csv')
diabetes.columns 


# In[116]:


diabetes.head()


# In[117]:


print("Diabetes data set dimensions : {}".format(diabetes.shape))


# In[118]:


diabetes.groupby('Outcome').size()


# In[119]:


diabetes.groupby('Outcome').size()


# In[120]:


diabetes.groupby('Outcome').hist(figsize=(9, 9))


# In[121]:


diabetes.isnull().sum()
diabetes.isna().sum()


# In[122]:


print("Total : ", diabetes[diabetes.BloodPressure == 0].shape[0])


# In[123]:


print(diabetes[diabetes.BloodPressure == 0].groupby('Outcome')['Age'].count())


# In[124]:


print("Total : ", diabetes[diabetes.Glucose == 0].shape[0])


# In[125]:


print(diabetes[diabetes.Glucose == 0].groupby('Outcome')['Age'].count())


# In[126]:


print("Total : ", diabetes[diabetes.SkinThickness == 0].shape[0])


# In[127]:


print(diabetes[diabetes.SkinThickness == 0].groupby('Outcome')['Age'].count())


# In[128]:


print("Total : ", diabetes[diabetes.BMI == 0].shape[0])


# In[129]:


print(diabetes[diabetes.BMI == 0].groupby('Outcome')['Age'].count())


# In[130]:


print("Total : ", diabetes[diabetes.Insulin == 0].shape[0])


# In[131]:


print(diabetes[diabetes.Insulin == 0].groupby('Outcome')['Age'].count())


# In[132]:


diabetes_mod = diabetes[(diabetes.BloodPressure != 0) & (diabetes.BMI != 0) & (diabetes.Glucose != 0)]
print(diabetes_mod.shape)


# In[133]:


feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = diabetes_mod[feature_names]
y = diabetes_mod.Outcome


# In[134]:


from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# In[135]:


models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC(gamma='scale')))
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=4000)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier(n_estimators=100)))
models.append(('GB', GradientBoostingClassifier()))


# In[136]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = diabetes_mod.Outcome, random_state=0)


# In[137]:


names = []
scores = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score(y_test, y_pred))
    names.append(name)

tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)


# In[138]:


strat_k_fold = StratifiedKFold(n_splits=10, shuffle=True)

names = []
scores = []

for name, model in models:
    
    score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    names.append(name)
    scores.append(score)

kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)


# In[139]:


axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')

for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[140]:


from sklearn.feature_selection import RFECV


# In[141]:


logreg_model = LogisticRegression(solver='lbfgs', max_iter=4000)

rfecv = RFECV(estimator=logreg_model, step=1, cv=strat_k_fold, scoring='accuracy')
rfecv.fit(X, y)

plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


# In[142]:


feature_importance = list(zip(feature_names, rfecv.support_))

new_features = []

for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
        
print(new_features)


# In[143]:


X_new = diabetes_mod[new_features]

initial_score = cross_val_score(logreg_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))

fe_score = cross_val_score(logreg_model, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


# In[144]:


gb_model = GradientBoostingClassifier()

gb_rfecv = RFECV(estimator=gb_model, step=1, cv=strat_k_fold, scoring='accuracy')
gb_rfecv.fit(X, y)

plt.figure()
plt.title('Gradient Boost CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(gb_rfecv.grid_scores_) + 1), gb_rfecv.grid_scores_)
plt.show()


# In[145]:


feature_importance = list(zip(feature_names, gb_rfecv.support_))

new_features = []

for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
        
print(new_features)


# In[146]:


X_new_gb = diabetes_mod[new_features]

initial_score = cross_val_score(gb_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))

fe_score = cross_val_score(gb_model, X_new_gb, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))


# In[147]:


from sklearn.model_selection import GridSearchCV


# In[148]:


c_values = list(np.arange(1, 10))

param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]


# In[149]:


grid = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=4000), param_grid, cv=strat_k_fold, scoring='accuracy')
grid.fit(X_new, y)


# In[150]:


print(grid.best_params_)
print(grid.best_estimator_)


# In[151]:


logreg_new = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')


# In[152]:


initial_score = cross_val_score(logreg_new, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Final accuracy : {} ".format(initial_score))

