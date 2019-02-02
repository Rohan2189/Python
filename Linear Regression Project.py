
# coding: utf-8

# In[1]:


data_file=r'R:\Data Science\Python\Data\Data\Data\loans data.csv'

import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.cross_validation import KFold
get_ipython().run_line_magic('matplotlib', 'inline')

ld=pd.read_csv(data_file)


# In[2]:


ld.head()


# In[3]:


for col in ["Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=ld[col].astype("str")
    ld[col]=[x.replace("%","") for x in ld[col]]


# In[4]:


ld.dtypes


# In[5]:


for col in ["Amount.Requested","Amount.Funded.By.Investors","Open.CREDIT.Lines","Revolving.CREDIT.Balance",
           "Inquiries.in.the.Last.6.Months","Interest.Rate","Debt.To.Income.Ratio"]:
    ld[col]=pd.to_numeric(ld[col],errors="coerce")


# In[6]:


ld.dtypes


# In[7]:


ld["Loan.Length"].value_counts()


# In[8]:


ll_dummies=pd.get_dummies(ld["Loan.Length"])
ll_dummies.head()


# In[9]:


ld["LL_36"]=ll_dummies["36 months"]


# In[10]:


ld.head()


# In[11]:


get_ipython().run_line_magic('reset_selective', 'll_dummies')


# In[12]:


ld=ld.drop('Loan.Length',axis=1)


# In[13]:


ld.dtypes


# In[14]:


ld["Loan.Purpose"].value_counts()


# In[15]:


round(ld.groupby("Loan.Purpose")["Interest.Rate"].mean())


# In[16]:


for i in range(len(ld.index)):
    if ld["Loan.Purpose"][i] in ["car","educational","major_purchase"]:
         ld.loc[i,"Loan.Purpose"]="cem"
    if ld["Loan.Purpose"][i] in ["home_improvement","medical","vacation","wedding"]:
        ld.loc[i,"Loan.Purpose"]="hmvw"
    if ld["Loan.Purpose"][i] in ["credit_card","house","other","small_business"]:
        ld.loc[i,"Loan.Purpose"]="chos"
    if ld["Loan.Purpose"][i] in ["debt_consolidation","moving"]:
        ld.loc[i,"Loan.Purpose"]="dm"


# In[17]:


lp_dummies=pd.get_dummies(ld["Loan.Purpose"],prefix="LP")
lp_dummies.head()


# In[18]:


ld=pd.concat([ld,lp_dummies],1)
ld=ld.drop(["Loan.Purpose","LP_renewable_energy"],axis=1)
ld.dtypes


# In[19]:


ld=ld.drop(["State"],1)


# In[20]:


ld["Home.Ownership"].value_counts()


# In[21]:


ld["ho_mort"]=np.where(ld["Home.Ownership"]=="MORTGAGE",1,0)
ld["ho_rent"]=np.where(ld["Home.Ownership"]=="RENT",1,0)
ld=ld.drop(["Home.Ownership"],1)


# In[22]:


ld["FICO.Range"].head()


# In[25]:


ld['f1'],ld['f2']=zip(*ld["FICO.Range"].apply(lambda x: x.split('-',1)))


# In[26]:


ld["fico"]=0.5*(pd.to_numeric(ld["f1"])+pd.to_numeric(ld["f2"]))

ld=ld.drop(["FICO.Range","f1","f2"],1)


# In[27]:


ld["Employment.Length"].value_counts()


# In[28]:


ld["Employment.Length"]=ld["Employment.Length"].astype("str")
ld["Employment.Length"]=[x.replace("years","") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("year","") for x in ld["Employment.Length"]]


# In[29]:


round(ld.groupby("Employment.Length")["Interest.Rate"].mean(),2)


# In[30]:


ld["Employment.Length"]=[x.replace("n/a","< 1") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("10+","10") for x in ld["Employment.Length"]]
ld["Employment.Length"]=[x.replace("< 1","0") for x in ld["Employment.Length"]]
ld["Employment.Length"]=pd.to_numeric(ld["Employment.Length"],errors="coerce")


# In[ ]:


ld.dtypes


# In[31]:


ld.dropna(axis=0,inplace=True)


# In[32]:


ld_train, ld_test = train_test_split(ld, test_size = 0.2,random_state=2)


# In[33]:


lm=LinearRegression()


# In[34]:


x_train=ld_train.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_train=ld_train["Interest.Rate"]
x_test=ld_test.drop(["Interest.Rate","ID","Amount.Funded.By.Investors"],1)
y_test=ld_test["Interest.Rate"]


# In[35]:


lm.fit(x_train,y_train)
lm_predicted=lm.predict(x_test)


# In[36]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(y_test, lm_predicted))


# In[37]:


rmse


# In[38]:


coefs=lm.coef_

features=x_train.columns

list(zip(features,coefs))


# In[39]:


# Finding best value of penalty weight with cross validation for ridge regression
alphas=np.linspace(.0001,10,100)
# We need to reset index for cross validation to work without hitch
x_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)


# In[40]:


rmse_list=[]
for a in alphas:
    ridge = Ridge(fit_intercept=True, alpha=a)

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        ridge.fit(x_train.loc[train], y_train[train])
        ridge_predicted = ridge.predict(x_train.loc[test])
        rmse_10cv = sqrt(mean_squared_error(y_train[test], ridge_predicted))
    
    # uncomment below to print rmse values for individidual alphas
    print('{:.4f}\t {:.6f}\t '.format(a,rmse_10cv))
    rmse_list.extend([rmse_10cv])
    best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )


# In[66]:


ridge=Ridge(fit_intercept=True,alpha=10)

ridge.fit(x_train,y_train)

ridge_pred=ridge.predict(x_test)


# In[68]:


ridge_rmse = sqrt(mean_squared_error(y_test, ridge_pred))
ridge_rmse


# In[71]:


list(zip(x_train.columns,ridge.coef_))


# In[43]:


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0) 
regressor.fit(x_train, y_train)


# In[44]:


y_pred = regressor.predict(x_test)


# In[69]:


dtree_rmse = sqrt(mean_squared_error(y_test, y_pred))
dtree_rmse


# In[63]:


#LASSO

alphas=np.linspace(0.0001,1,100)
rmse_list=[]
for a in alphas:
    lasso=Lasso(fit_intercept=True, alpha=a,max_iter=10000)

    # computing average RMSE across 10-fold cross validation
    kf = KFold(len(x_train), n_folds=10)
    xval_err = 0
    for train, test in kf:
        lasso.fit(x_train.loc[train], y_train[train])
        lasso_predicted = lasso.predict(x_train.loc[test])
        rmse_10cv = sqrt(mean_squared_error(y_train[test], lasso_predicted))
    
    # uncomment below to print rmse values for individidual alphas
    print('{:.4f}\t {:.6f}\t '.format(a,rmse_10cv))
    rmse_list.extend([rmse_10cv])
    best_alpha=alphas[rmse_list==min(rmse_list)]
print('Alpha with min 10cv error is : ',best_alpha )


# In[64]:


lasso=Lasso(fit_intercept=True,alpha=0.0910)

lasso.fit(x_train,y_train)

lasso_pred=lasso.predict(x_test)


# In[70]:


lasso_rmse = sqrt(mean_squared_error(y_test, lasso_pred))
lasso_rmse


# In[73]:


list(zip(x_train.columns,lasso.coef_))

