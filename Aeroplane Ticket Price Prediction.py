#!/usr/bin/env python
# coding: utf-8

# #  ----------------Predicting the ticket price of the Aeroplane--------------

# In[117]:


import numpy as np
import pandas as pd
data=pd.read_excel('C:/Users/barath raj/Downloads/Data_Train.xlsx')


# In[118]:


import seaborn as sb


# In[119]:


data.head()


# In[120]:



get_ipython().system('pip install plotly')
import plotly.express as px


# In[121]:


data.isna().sum()


# In[122]:


data.dropna(inplace=True)


# In[123]:


data.isna().sum()


# In[124]:


data.columns


# # ---------Converting the features into date-time format---------

# In[125]:


def date(a):
    data[a]=pd.to_datetime(data[a])
date('Arrival_Time')
date('Dep_Time')
date('Date_of_Journey')
data['year']=data['Date_of_Journey'].dt.year
data['month']=data['Date_of_Journey'].dt.month
data['day']=data['Date_of_Journey'].dt.day
data.drop('Date_of_Journey',axis=1,inplace=True)


# In[126]:


data.head()


# In[127]:


def hour(x):
    data[x+'_hour']=data[x].dt.hour
    data[x+'_minute']=data[x].dt.minute
def drop(x):
    data.drop(x,axis=1,inplace=True)


# In[128]:


hour('Dep_Time')
hour('Arrival_Time')


# In[129]:


data.head()


# In[130]:


drop('Dep_Time')
drop('Arrival_Time')


# In[131]:


data.head(2)


# #  -----Splitting the Duration column into hour and mins as separate column----

# In[132]:


data['Duration']


# In[133]:


data['Duration'].unique
dur=list(data['Duration'])


# In[134]:


#dur=list(data['Duration'])
for i in range(len(dur)):
    if len(dur[i].split(' '))==2:
        pass
    else:
        if 'm' in dur[i]:
            dur[i]='0h ' + dur[i]
        else:
            dur[i]=dur[i] + ' 0m'


# In[ ]:





# In[135]:


data['Duration']=dur


# In[136]:


data['Duration']


# In[137]:


data.head(2)


# In[138]:


data['Duration']


# In[139]:


'20h 60m'.split(' ')


# In[140]:


def h(x):
    return x.split(' ')[0][0:-1]
def m(x):
    return x.split(' ')[1][0:-1]    


# In[141]:


data['duration_hour']=data['Duration'].apply(h)
data['duration_mins']=data['Duration'].apply(m)


# In[142]:


data.head(2)


# In[143]:


drop('Duration')


# In[144]:


data.head()


# In[145]:


data.dtypes


# In[146]:


data['duration_hour']=data['duration_hour'].astype(int)
data['duration_mins']=data['duration_mins'].astype(int)


# In[147]:


data.dtypes


# # -------------------- Creating various INSIGHTS from the data ----------------

# In[148]:


b=data['Source'].value_counts().reset_index()


# In[149]:


b.columns=['state','count']
b


# In[150]:


data.head(2)


# #                    Number of Airline passengers  in a State

# In[151]:


px.bar(b,x=b['state'],y=b['count'],color=b['state'],hover_name=b['state'])


# In[152]:


d=data['Total_Stops'].value_counts().reset_index()
d.columns=['stop','count']
d


# # Count of Flights with number of stoppings

# In[153]:


px.bar(d,y='count',x=['1 stop','non-stop','2 stops','3 stops','4 stops'],hover_name='stop',color='stop')


# In[154]:


get_ipython().system('pip install sort-dataframeby-monthorweek')
get_ipython().system('pip install sorted-months-weekdays')


# In[155]:


data.head()


# In[156]:


data['month'].unique()


# In[157]:


x=['month','Price']
data1=data[x]
data1


# In[158]:


aa={
    1:'jan',
    2:'feb',
    3:'mar',
    4:'apr',
    5:'may',
    6:'jun',
    7:'jul',
    8:'aug',
    9:'sep',
    10:'oct',
    11:'nov',
    12:'dec'
    
}
data['month']=data.month.map(aa)


# In[159]:


final=data.groupby(['month'])['Price'].mean().reset_index()
final


# In[160]:


import sort_dataframeby_monthorweek as sdd
def sort(df,col):
    return sdd.Sort_Dataframeby_Month(df,col)
data1=sort(final,'month')


# In[161]:


data1


# # Average amount spend by  Airline passengers in a month

# In[162]:


px.bar(x=data1['month'],y=data1['Price'],color=data1['month'],hover_name=data1['month'])


# In[163]:


c=data['month'].value_counts().reset_index()
c.columns=['month','travellers']
c


# # Total number of passengers in a month

# In[164]:


px.bar(x=c['month'],y=c['travellers'],color=c['month'],hover_name=c['month'])


# In[ ]:





# In[165]:


data2=data.groupby(['Airline'])['Price'].mean().reset_index()
data2


# # Amount earned by different Airlines

# In[166]:


px.bar(y=data2['Price'],x=data2['Airline'],hover_name=data2['Airline'],color=data2['Airline'])


# In[167]:


s=data['Airline'].value_counts().reset_index()
s.columns=['airline','count']
s


# # Number of passengers in different Airlines

# In[168]:


px.funnel(x=s['airline'],y=s['count'],color=s['airline'],hover_name=s['airline'])


# In[169]:


bb={
    'jan':1,
    'feb':2,
    'mar':3,
    'apr':4,
    'may':5,
    'jun':6,
    'jul':7,
    'aug':8,
    'sep':9,
    'oct':10,
    'nov':11,
    'dec':12
    
}
data['month']=data['month'].map(bb)


# In[ ]:





# #  ------------------------------ LabelEncoding -------------------------------

# In[170]:


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
one=LabelEncoder()
def LabelEncoder(x):
    data[x]=one.fit_transform(data[x])


# In[171]:


LabelEncoder('Airline')
LabelEncoder('Source')
LabelEncoder('Destination')
LabelEncoder('Total_Stops')
LabelEncoder('Additional_Info')


# In[172]:


data['Airline'].unique()


# In[173]:


data.head()


# # ---------- splitting Route column as separate column on stopping basis -----------

# In[174]:


data['route_1']=data['Route'].str.split('→').str[0]
data['route_2']=data['Route'].str.split('→').str[1]
data['route_3']=data['Route'].str.split('→').str[2]
data['route_4']=data['Route'].str.split('→').str[3]
data['route_5']=data['Route'].str.split('→').str[4]


# In[175]:


data.head()


# In[176]:


data.isna().sum()


# In[177]:


data.columns


# In[178]:


for i in ['route_3','route_4','route_5']:
    data[i].fillna('None',inplace=True)


# In[179]:


data.isna().sum()


# In[180]:


pd.set_option('display.max_columns',21)


# In[181]:


LabelEncoder('route_1')
LabelEncoder('route_2')
LabelEncoder('route_3')
LabelEncoder('route_4')
LabelEncoder('route_5')
data.head()


# In[183]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[184]:


data.drop('year',axis=1,inplace=True)


# In[185]:


data.drop('Route',axis=1,inplace=True)


# In[ ]:





# In[ ]:





# In[186]:


import matplotlib.pyplot as plt
def plot(x):
    plt.subplot()
    sb.boxplot(data[x])


# In[187]:


plot('Price')


# In[188]:


data['Price']=np.where(data['Price']>=40000,data['Price'].median(),data['Price'])


# In[189]:


plot('Price')


# In[190]:


data.head()


# #  ---------------------------- Selecting important features------------------------

# In[191]:


from sklearn.feature_selection import mutual_info_classif
y=data['Price']
x=data.drop('Price',axis=1)


# In[192]:


mutual_info_classif(x,y)


# In[193]:


imp=pd.DataFrame(mutual_info_classif(x,y),index=x.columns)
imp.columns=['importance']


# In[194]:


imp


# In[ ]:





# In[195]:


imp.sort_values(by='importance',ascending=False)


# # ------------------------------Creating model----------------------------

# In[196]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,r2_score,mean_squared_error
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[197]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[198]:


[int(x) for x in np.linspace(start=5,stop=30,num=6)]


# In[200]:


from sklearn.ensemble import RandomForestRegressor
import pickle
def predict(model,dump):
    le=model
    le.fit(xtrain,ytrain)
    print('training score {}'.format(le.score(xtrain,ytrain)))
    
    pre=le.predict(xtest)
    print('test score {}'.format(le.score(xtest,ytest)))
    print('\n')
    
    print('cross validate score {}'.format(cross_val_score(le,x,y,cv=5)))
    print('r2_score',r2_score(pre,ytest))
    print('mse',mean_squared_error(pre,ytest))
    print('\n')

    sb.distplot(pre-ytest)
    if dump==1:
        file=open('C:/Users/barath raj/Documents/model.pkl','wb')
        pickle.dump(model,file)


# In[201]:


predict(RandomForestRegressor(),1)


# In[202]:


from sklearn.tree import DecisionTreeRegressor
predict(DecisionTreeRegressor(),0)


# # ------------ This is how the coustomer use the model to predict data -----------

# In[203]:


model=open('C:/Users/barath raj/Documents/model.pkl','rb')
forest=pickle.load(model)


# In[204]:


pre2=forest.predict(xtest)


# In[205]:


r2_score(pre2,ytest)


# In[207]:


cross_val_score(forest,x,y,cv=5)


# In[208]:


cross_val_score(forest,x,y,cv=5).mean()


# In[ ]:




