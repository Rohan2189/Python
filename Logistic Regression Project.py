
# coding: utf-8

# # Linear Model Practice Exercise
# 
# Logistic regression discussion in class did not include tuning of hyperparameter C. Also the model developed in the class only used l1 penalty. 
# 
# Please carry out following exercises for the same data:
# 
# <ol>
# <li> Build logistic regression model with **l1** penalty and best value of penalty C. Find out auc score on test data for the same. </li>
# <li> For **l1** penalty plot a graph showing how number of variables with coeffient 0 increase with increasing penalty and on the same graph also show how auc score on test data moves with increasing penalty</li>
# <li> Build logistic regression model with **l2** penalty and best value of penalty C. Find out auc score on test data for the same.</li>
# <li> For linear regression problem discussed in class, examine trend between your response and fico score. Is the trend linear as we assumed?</li>
# </ol>

# In[1]:


import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge


# In[2]:


data_file='R:\Data Science\Python\Data\Data\Data\Existing Base.csv'
bd=pd.read_csv(data_file)


# In[3]:


range(len(bd))


# In[4]:


bd["children"].value_counts()


# In[5]:


bd.loc[bd["children"] == "Zero","children"]=0
bd.loc[bd["children"] == "4+","children"]=4
bd["children"]=pd.to_numeric(bd["children"],errors="coerce")


# In[6]:


bd.dtypes


# In[7]:


bd["y"]=np.where(bd["Revenue Grid"]==2,0,1)
bd=bd.drop(["Revenue Grid"], 1)


# In[8]:


round(bd.groupby("children")["y"].mean(),2)


# In[9]:


round(bd.groupby("age_band")["y"].mean(),2)


# In[10]:


for i in range(len(bd)):
    if bd["age_band"][i] in ("45-50","51-55","65-70","71+"):
        bd.loc[i,"age_band"] = "ab_10"
    if bd["age_band"][i] in ("22-25","26-30","31-35","41-45","55-60"):
        bd.loc[i,"age_band"] = "ab_11"
    if bd["age_band"][i]=="36-40":
        bd.loc[i,"age_band"]="ab_13"
    if bd["age_band"][i]=="18-21":
        bd.loc[i,"age_band"]="ab_17"
    if bd["age_band"][i]=="61-65":
        bd.loc[i,"age_band"]="ab_9"
                    
ab_dummies=pd.get_dummies(bd["age_band"])


# In[11]:


bd=pd.concat([bd,ab_dummies],1)
bd=bd.drop(["age_band","Unknown"],1)


# In[12]:


round(bd.groupby("status")["y"].mean(),2)


# In[13]:


for i in range(len(bd)):
    if bd["status"][i] in ("Divorced/Separated","Widowed"):
        bd.loc[i,"status"] = "Stat_DSW"
    if bd["status"][i] in ("Partner"):
        bd.loc[i,"status"] = "Stat_P"
    if bd["status"][i] in ("Single/Never Married"):
        bd.loc[i,"status"] = "Stat_SNM"

stat_dummies=pd.get_dummies(bd["status"])


# In[14]:


bd=pd.concat([bd,stat_dummies],1)
bd=bd.drop(["Unknown","status"],1)


# In[15]:


round(bd.groupby("occupation")["y"].mean(),2)


# In[16]:


for i in range(len(bd)):
    if bd["occupation"][i] in ["Unknown","Student","Secretarial/Admin","Other","Manual Worker"]:
        bd.loc[i,"occupation"]="oc_11"
    if bd["occupation"][i] in ["Professional","Business Manager"]:
        bd.loc[i,"occupation"]="oc_12"
    if bd["occupation"][i]=="Retired":
        bd.loc[i,"occupation"]="oc_10"
oc_dummies=pd.get_dummies(bd["occupation"])

bd=pd.concat([bd,oc_dummies],1)

bd=bd.drop(["occupation","Housewife"],1)


# In[17]:


bd["occupation_partner"].value_counts()


# In[18]:


round(bd.groupby("occupation_partner")["y"].mean(),2)


# In[19]:


bd["ocp_10"]=0
bd["ocp_11"]=0
for i in range(len(bd)):
    if bd["occupation_partner"][i] in ["Unknown","Retired","Other"]:
        bd.loc[i,"ocp_10"]=1
    if bd["occupation_partner"][i] in ["Business Manager","Housewife","Manual Worker","Professional"]:
        bd.loc[i,"ocp_11"]=1


# In[20]:


bd=bd.drop(["occupation_partner"],1)

bd.drop(["post_area","post_code"],axis=1,inplace=True)


# In[21]:


bd.head()


# In[22]:


bd["TVarea"].value_counts()


# In[23]:


round(bd.groupby("TVarea")["y"].mean(),2)


# In[24]:


for i in range(len(bd)):
    if bd["TVarea"][i] in ["Meridian","TV South West","Tyne Tees"]:
        bd.loc[i,"TVarea"]="TV_9"
    if bd["TVarea"][i] in ["Grampian","HTV"]:
        bd.loc[i,"TVarea"]="TV_10"
    if bd["TVarea"][i] in ["Anglia","Carlton","Central","Scottish TV"]:
        bd.loc[i,"TVarea"]="TV_11"
    if bd["TVarea"][i] in ["Ulster","Unknown","Yorkshire","Granada"]:
        bd.loc[i,"TVarea"]="TV_12"
bd_dummies=pd.get_dummies(bd["TVarea"])


bd=pd.concat([bd,oc_dummies],axis=1)
bd.drop(["TVarea","Border"],axis=1,inplace=True)

bd.drop(['region'],axis=1,inplace=True)


# In[25]:


bd["gender"].value_counts()


# In[26]:


round(bd.groupby("gender")["y"].mean(),2)


# In[27]:


bd["home_status"].value_counts()


# In[28]:


round(bd.groupby("home_status")["y"].mean(),2)


# In[29]:


bd["self_employed"].value_counts()


# In[30]:


round(bd.groupby("self_employed")["y"].mean(),2)


# In[31]:


bd["self_employed_partner"].value_counts()


# In[32]:


round(bd.groupby("self_employed_partner")["y"].mean(),2)


# In[33]:


bd["hs_own"]=np.where(bd["home_status"]=="Own Home",1,0)
del bd["home_status"]


bd["gender_f"]=np.where(bd["gender"]=="Female",1,0)
del bd["gender"]

bd["semp_no"]=np.where(bd["self_employed"]=="No",1,0)
del bd["self_employed"]


bd["semp_part_no"]=np.where(bd["self_employed_partner"]=="No",1,0)
del bd["self_employed_partner"]


# In[34]:


bd["family_income"].value_counts()


# In[35]:


round(bd.groupby("family_income")["y"].mean(),2)


# In[36]:


bd["fi"]=4 
bd.loc[bd["family_income"]=="< 8,000, >= 4,000","fi"]=6
bd.loc[bd["family_income"]=="<10,000, >= 8,000","fi"]=9
bd.loc[bd["family_income"]=="<12,500, >=10,000","fi"]=11.25
bd.loc[bd["family_income"]=="<15,000, >=12,500","fi"]=13.75
bd.loc[bd["family_income"]=="<17,500, >=15,000","fi"]=16.25
bd.loc[bd["family_income"]=="<20,000, >=17,500","fi"]=18.75
bd.loc[bd["family_income"]=="<22,500, >=20,000","fi"]=21.25
bd.loc[bd["family_income"]=="<25,000, >=22,500","fi"]=23.75
bd.loc[bd["family_income"]=="<27,500, >=25,000","fi"]=26.25
bd.loc[bd["family_income"]=="<30,000, >=27,500","fi"]=28.75
bd.loc[bd["family_income"]==">=35,000","fi"]=35
bd=bd.drop(["family_income"],1)



bd.drop(['Tvarea'],1,inplace=True)


# In[37]:


bd.isnull().sum()


# In[38]:


bd.dropna(axis=0,inplace=True)


# In[39]:


bd_train, bd_test = train_test_split(bd, test_size = 0.2,random_state=2)

x_train=bd_train.drop(["y","REF_NO"],1)
y_train=bd_train["y"]

x_test=bd_test.drop(["y","REF_NO"],1)
y_test=bd_test["y"]


# In[40]:


x_train.head()


# In[41]:


y_train.head()


# In[42]:


x_test.head()


# In[43]:


y_test.head()


# In[44]:


x_train.index=[x for x in range(len(x_train))]


# In[45]:


x_train.index


# In[46]:


y_train.index=[x for x in range(len(x_train))]


# In[47]:


y_train.index


# In[48]:


C_param =[0.0001,0.0002,0.0005,0.001,0.005,0.01,0.05, 0.1,0.5, 1, 10, 100,500,1000] 
auc_10cv=[]
for c in C_param:
    logr=LogisticRegression(C=c,penalty="l2",class_weight="balanced",random_state=2)
    kf = KFold(len(x_train), n_folds=10)
    score_c=0
    for train, test in kf:
        logr.fit(x_train.loc[train], y_train[train])
        score_c+=roc_auc_score(y_train,logr.predict(x_train))
    auc_10cv.append(score_c/10)
    print('{:.3f}\t {:.5f}\t '.format(c,score_c/10))
C_param=np.array(C_param)
auc_10cv=np.array(auc_10cv)
best_c=C_param[auc_10cv==max(auc_10cv)][0]
print('Value of C with max AUC score:',max(auc_10cv),' for 10 fold CV is :',best_c )


# In[49]:


logr=LogisticRegression(C=best_c,penalty="l2",class_weight="balanced",random_state=2)


# In[50]:


logr.fit(x_train,y_train)


# In[51]:


roc_auc_score(y_train,logr.predict(x_train))


# In[52]:


roc_auc_score(y_test,logr.predict(x_test))


# In[53]:


prob_score=pd.Series(list(zip(*logr.predict_proba(x_train)))[1])


# In[54]:


prob_score.head()


# In[55]:


cutoffs=np.linspace(0,1,100)


# In[56]:


KS_cut=[]
for cutoff in cutoffs:
    predicted=pd.Series([0]*len(y_train))
    predicted[prob_score>cutoff]=1
    df=pd.DataFrame(list(zip(y_train,predicted)),columns=["real","predicted"])
    TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
    FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
    TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
    FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_cut.append(KS)

cutoff_data=pd.DataFrame(list(zip(cutoffs,KS_cut)),columns=["cutoff","KS"])

KS_cutoff=cutoff_data[cutoff_data["KS"]==cutoff_data["KS"].max()]["cutoff"]


# In[57]:


KS_cutoff


# In[58]:


# Performance on test data
prob_score_test=pd.Series(list(zip(*logr.predict_proba(x_test)))[1])

predicted_test=pd.Series([0]*len(y_test))
predicted_test[prob_score_test>float(KS_cutoff)]=1

df_test=pd.DataFrame(list(zip(y_test,predicted_test)),columns=["real","predicted"])

k=pd.crosstab(df_test['real'],df_test["predicted"])
print('confusion matrix :\n \n ',k)
TN=k.iloc[0,0]
TP=k.iloc[1,1]
FP=k.iloc[0,1]
FN=k.iloc[1,0]
P=TP+FN
N=TN+FP


# In[59]:


# Accuracy of test
(TP+TN)/(P+N)


# In[60]:


# Sensitivity on test
TP/P


# In[61]:


#Specificity on test
TN/N


# In[62]:


from sklearn.metrics import classification_report,confusion_matrix


# In[63]:


classification_report(y_test,logr.predict(x_test))


# In[64]:


cm=confusion_matrix(y_test, logr.predict(x_test))


# In[65]:


cm


# In[66]:


total1=sum(sum(cm))
#####from confusion matrix calculate accuracy
accuracy1=(cm[0,0]+cm[1,1])/total1
print ('Accuracy : ', accuracy1)

sensitivity1 = cm[0,0]/(cm[0,0]+cm[0,1])
print('Sensitivity : ', sensitivity1 )

specificity1 = cm[1,1]/(cm[1,0]+cm[1,1])
print('Specificity : ', specificity1)


# Logistic Regression based on different techniques

# SVM Technique :-

# In[67]:


from sklearn import svm
clf_svm=svm.LinearSVC()


# In[68]:


clf_svm.fit(x_train,y_train)


# In[69]:


y_predict=clf_svm.predict(x_test)


# In[70]:


y_predict


# In[71]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_predict)


# Naive Bayes Technique:-

# In[72]:


from sklearn.naive_bayes import BernoulliNB 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


# In[73]:


BernNB= BernoulliNB(binarize = True)
BernNB.fit(x_train,y_train)
Bern_predict=BernNB.predict(x_test)
accuracy_score(y_test,Bern_predict)


# In[74]:


GaussNB= GaussianNB()
GaussNB.fit(x_train,y_train)
Gauss_predict=GaussNB.predict(x_test)
accuracy_score(y_test,Gauss_predict)


# KNN Technique:-

# In[75]:


from sklearn.neighbors import KNeighborsClassifier


# In[76]:


clf_knn=KNeighborsClassifier()
clf_knn.fit(x_train,y_train)
Knn_predict=clf_knn.predict(x_test)
accuracy_score(y_test,Knn_predict)


# Decision Tree and Random Forest Technique :-

# In[77]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[78]:


clf_dtree=DecisionTreeClassifier()
clf_dtree.fit(x_train,y_train)
dtree_predict=clf_dtree.predict(x_test)
accuracy_score(y_test,dtree_predict)


# In[97]:


clf_rf=RandomForestClassifier(verbose=1,n_jobs=-1)
clf_rf.fit(x_train,y_train)
rf_predict=clf_rf.predict(x_test)
accuracy_score(y_test,rf_predict)


# KFOLD for all Techniques :-

# SVM Technique :-

# In[80]:


C_svm =[0.0001,0.0002,0.0005,0.001,0.005,0.01,0.05, 0.1,0.5, 1, 10, 100,500,1000] 
svm_10cv=[]
for c in C_svm:
    clf_svm=svm.LinearSVC(C=c,penalty="l2",random_state=2)
    kf_svm = KFold(len(x_train), n_folds=10)
    svm_score_c=0
    for train, test in kf_svm:
        clf_svm.fit(x_train.loc[train], y_train[train])
        svm_score_c+=accuracy_score(y_train,clf_svm.predict(x_train))
    svm_10cv.append(svm_score_c/10)
    print('{:.4f}\t {:.5f}\t '.format(c,svm_score_c/10))
C_svm=np.array(C_svm)
svm_10cv=np.array(svm_10cv)
best_c_svm=C_svm[svm_10cv==max(svm_10cv)][0]
print('Value of C with max SVM score:',max(svm_10cv),' for 10 fold CV is :',best_c_svm )


# In[81]:


clf_svm_tun=svm.LinearSVC(C=best_c_svm,penalty="l2",random_state=2)
clf_svm_tun.fit(x_train,y_train)
y_predict=clf_svm_tun.predict(x_test)
accuracy_score(y_test,y_predict)


# K-Fold Decision Tree and Random Forest

# In[82]:


from sklearn.model_selection import RandomizedSearchCV
params={'class_weight':[None,'balanced'],
       'criterion':['entropy','gini'],
       'max_depth':[None,5,10,15,20,30,50,70],
       'min_samples_leaf':[1,2,5,10,15,20],
       'min_samples_split':[2,5,10,15,20]}


# In[83]:


random_search_dtree=RandomizedSearchCV(clf_dtree,cv=10,
                                      param_distributions=params,
                                      scoring='roc_auc',
                                      n_iter=10)


# In[84]:


random_search_dtree.fit(x_train,y_train)


# In[86]:


best_tree=random_search_dtree.best_estimator_


# In[87]:


best_tree.fit(x_train,y_train)


# In[91]:


from sklearn import tree
dotfile = open("dtree.dot",'w')

tree.export_graphviz(best_tree,out_file=dotfile,
                    feature_names=x_train.columns,
                    class_names=["0","1"],
                    proportion=True)

dotfile.close()


# In[96]:


btree_predict=best_tree.predict(x_test)
accuracy_score(y_test,btree_predict)


# In[98]:


param_rf={'n_estimators':[100,200,300,500,700,1000],
          'max_features':[5,10,20,25,30,35],
          'bootstrap':[True,False],
          'class_weight':[None,'balanced'],
          'criterion':['entropy','gini'],
          'max_depth':[None,5,10,15,20,30,50,70],
          'min_samples_leaf':[1,2,5,10,15,20],
          'min_samples_split':[2,5,10,15,20]
         }


# In[99]:


random_search_rf=RandomizedSearchCV(clf_rf,param_distributions=param_rf, n_iter=10,scoring='roc_auc',cv=10)
random_search_rf.fit(x_train,y_train)


# In[102]:


rf_best=random_search_rf.best_estimator_


# In[103]:


rf_best.fit(x_train,y_train)


# In[107]:


rf_best_predict=rf_best.predict(x_test)
accuracy_score(y_test,rf_best_predict)


# In[110]:


DecisionTreeClassifier()

