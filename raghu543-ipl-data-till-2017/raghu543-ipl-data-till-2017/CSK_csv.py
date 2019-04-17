
# coding: utf-8

# In[183]:


import os
os.getcwd()


# In[184]:


os.chdir(r"F:\BE Project\Data-set\ipl_csk_csv_new")


# In[185]:


os.getcwd()


# In[186]:


print("hii..")


# In[187]:




    
def data_process_1(df):
    tdf=df.T
    temp=0
    for i in range(len(df)):
        if tdf.iloc[1,i]==2:
            temp=i
            break
    df2=df.iloc[temp:]
    df1=df.iloc[:temp]
    df2.reset_index(inplace=True)
    df2.drop('index',axis=1,inplace=True)
    return df1,df2

def data_process_2(df,i):
    df.drop([1,10],axis=1,inplace=True)
    df=df.T
    runs=[]
    for val in df.loc['runs']:
        runs.append(int(val))
        
    df['total_wickets']=df.loc['wicket'].count()
    df['extras']=df.loc['extras'].sum()
    #if df.iloc[0,0]==2:
     #   df['first_bat']=df1['batting'][0]
    #else:
     #   df['first_bat']=df.iloc[2,0]
    df['batting']=df.iloc[2,0]
    if df.iloc[0,0]==1:
        df['bowling']=df2['batting'][0]
    else:
        df['bowling']=df1['batting'][0]
    df['inning']=df.iloc[0,0]
    df['total_runs']=np.sum(runs)
    #df['winner']=winner[i][0]
    df.drop(['inning','striker','non_striker','batting','bowler','ball','extras','wicket'],inplace=True)
    return df



# In[188]:


import pandas as pd
import numpy as np
inning1=[]
inning2=[]
for i in range(1,131):
    
    df=pd.read_csv("csk%s.csv"%(str(i)),header=-1,names=[1,"inning","ball","batting","striker","non_striker","bowler","runs","extras",10,"wicket"])
    
    df1,df2=data_process_1(df)
    
    final_df1=data_process_2(df1,i-1)
    final_df1.reset_index(inplace=True)
    final_df1.drop(['index'],axis=1,inplace=True)
    
    final_df2=data_process_2(df2,i-1)
    final_df2.reset_index(inplace=True)
    final_df2.drop(['index'],axis=1,inplace=True)
    
    inning1.append(final_df1)
    inning2.append(final_df2)


# In[189]:


len(inning1)


# In[190]:


len(inning2)


# In[191]:


inning1[50].head()


# In[192]:



winner=[]
for i in range(130):
    if inning1[i]['total_runs'].values>inning2[i]['total_runs'].values:
        winner.append(inning1[i]['batting'].values)
    if inning2[i]['total_runs'].values>inning1[i]['total_runs'].values:
        winner.append(inning2[i]['batting'].values)
    if inning2[i]['total_runs'].values==inning1[i]['total_runs'].values:
        winner.append(['tie'])
    
    


# In[193]:


len(winner)


# In[194]:


winner[0][0]


# In[195]:


inn1_df=pd.concat(inning1,ignore_index=True)


# In[196]:


inn1_df.head()


# In[197]:


teams=[]
for team in winner:
    #if team[0] not in teams:
    teams.append(team[0])
len(teams)
teams[:3]


# In[198]:


unique_teams=[]
for team in teams:
    if team not in unique_teams:
        unique_teams.append(team)
unique_teams


# In[199]:


team_dct=dict.fromkeys(unique_teams,0)


# In[200]:


for i in range(len(unique_teams)):
    team_dct[unique_teams[i]]=i


# In[201]:


team_dct


# In[202]:


for i in range(len(inn1_df)):
    inn1_df.loc[i,'batting']=team_dct[inn1_df.loc[i,'batting']]
    inn1_df.loc[i,'bowling']=team_dct[inn1_df.loc[i,'bowling']]
    #inn1_df.loc[i,'winner']=team_dct[inn1_df.loc[i,'winner']]


# In[203]:


inn1_df['batting'].head()


# In[204]:


inn1_df.iloc[0,:]


# In[205]:


inn1_df.fillna(value=-1,inplace=True)
inn1_df.iloc[0,:]


# In[206]:


team_code=[]
for team in teams:
    team_code.append(team_dct[team])


# In[207]:


team_code[:10]


# In[281]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(inn1_df,team_code)


# In[282]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import seaborn as sns

model=KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)


# In[283]:


predicted=model.predict(x_test)


# In[284]:


f1_score(y_test,predicted,average='micro')


# In[285]:


f1_score(y_test,predicted,average='macro')


# In[286]:


predicted


# In[287]:


output_compare=[]
for i in range(len(predicted)):
    output_compare.append((y_test[i],predicted[i]))
    


# In[288]:


print(output_compare)


# In[289]:


model.score(x_test,y_test)


# ## Predicting Inn2 Score

# In[46]:


inn2_df=pd.concat(inning2,ignore_index=True)


# In[47]:


inn2_df.head()


# In[48]:


inn2_runs=inn2_df['total_runs']


# In[49]:


inn2_runs[:5]


# In[50]:


"""for i in range(len(inn2_df)):
    inn2_df.loc[i,'batting']=team_dct[inn2_df.loc[i,'batting']]
    inn2_df.loc[i,'bowling']=team_dct[inn2_df.loc[i,'bowling']]
    #inn1_df.loc[i,'winner']=team_dct[inn1_df.loc[i,'winner']]"""


# In[51]:


x_train,x_test,y_train,y_test=train_test_split(inn1_df,inn2_runs)


# In[52]:


from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(x_train,y_train)
predicted=model.predict(x_test)


# In[53]:


output_compare=[]
for i in range(len(predicted)):
    output_compare.append((y_test.values[i],predicted[i]))


# In[54]:


print(output_compare)


# In[55]:


model.score(x_test,y_test)


# ## predicting inn1 score based on powerplay score

# In[225]:




    
def data_process_1(df):
    tdf=df.T
    temp=0
    for i in range(len(df)):
        if tdf.iloc[1,i]==2:
            temp=i
            break
    df2=df.iloc[temp:]
    df1=df.iloc[:temp]
    df2.reset_index(inplace=True)
    df2.drop('index',axis=1,inplace=True)
    return df1,df2

def data_process_2(df):
    df.drop([1,10],axis=1,inplace=True)
    df=df.T
    runs=[]
    for val in df.loc['runs']:
        runs.append(int(val))
        
    df['total_wickets']=df.loc['wicket'].count()
    df['extras']=df.loc['extras'].sum()
    #if df.iloc[0,0]==2:
     #   df['first_bat']=df1['batting'][0]
    #else:
     #   df['first_bat']=df.iloc[2,0]
    df['batting']=df.iloc[2,0]
    if df.iloc[0,0]==1:
        df['bowling']=df2['batting'][0]
    else:
        df['bowling']=df1['batting'][0]
    df['inning']=df.iloc[0,0]
    df['runs']=np.sum(runs)
    #df['winner']=winner[i][0]
    df.drop(['inning','striker','non_striker','batting','bowler','ball','extras','wicket'],inplace=True)
    return df



def get_pp(df):
    temp=0
    tdf=df.T
    for i in range(len(df)):
        if tdf.iloc[2,i]==6.1:
            temp=i
            break
    d=df[:temp]
    return d

def get_15(df):
    temp=0
    tdf=df.T
    for i in range(len(df)):
        if tdf.iloc[2,i]==15.1:
            temp=i
            break
    d=df[:temp]
    return d


# In[226]:


import pandas as pd
import numpy as np
inning1_pp=[]
inning1_15=[]
for i in range(1,131):
    
    df=pd.read_csv("csk%s.csv"%(str(i)),header=-1,names=[1,"inning","ball","batting","striker","non_striker","bowler","runs","extras",10,"wicket"])
    
    df1,df2=data_process_1(df)
    
    
    d1=get_pp(df1)
    d2=get_15(df1)
    ppdf=data_process_2(d1)
    df15=data_process_2(d2)
    final_df1=data_process_2(df1)
    ppdf['total_run']=final_df1['runs'][0]
    #df15['total_run']=final_df1['runs'][0]
 
    #final_df1.reset_index(inplace=True)
    #final_df1.drop(['index'],axis=1,inplace=True)
    
    #final_df2=data_process_2(df2,i-1)
    #final_df2.reset_index(inplace=True)
    #final_df2.drop(['index'],axis=1,inplace=True)
    
    
    inning1_pp.append(ppdf)
    inning1_15.append(df15)
    #inning2.append(final_df2)


# In[227]:


inn1_pp=pd.concat(inning1_pp,ignore_index=True)
inn1_pp.head()


# In[228]:


inn1_15=pd.concat(inning1_15,ignore_index=True)
inn1_15.head()


# In[229]:


total_runs=inn1_pp['total_run'].values
inn1_pp.drop(['total_run'],inplace=True,axis=1)
#inn1_15.drop(['total_run'],inplace=True,axis=1)


# In[230]:


inn1_pp.head()


# In[231]:


inn1_15.head()


# In[232]:



for i in range(len(inn1_pp)):
    inn1_pp.loc[i,'batting']=team_dct[inn1_pp.loc[i,'batting']]
    inn1_pp.loc[i,'bowling']=team_dct[inn1_pp.loc[i,'bowling']]
    #inn1_df.loc[i,'winner']=team_dct[inn1_df.loc[i,'winner']]


# In[233]:


inn1_pp.fillna(-1,inplace=True)
inn1_pp.head()


# In[234]:


team_dct


# In[235]:


inn1_15.batting.values


# In[236]:


temp=[]
for i in range(len(inn1_15.batting.values)):
    if inn1_15.batting[i]==0:
        temp.append(i)


# In[237]:


temp


# In[238]:


inn1_15.bowling[temp]


# In[239]:


inn1_15.batting[temp]='Chennai Super Kings'


# In[240]:



for i in range(len(inn1_15)):
    inn1_15.loc[i,'batting']=team_dct[inn1_15.loc[i,'batting']]
    inn1_15.loc[i,'bowling']=team_dct[inn1_15.loc[i,'bowling']]
    #inn1_df.loc[i,'winner']=team_dct[inn1_df.loc[i,'winner']]


# In[241]:


inn1_15.batting.values


# In[242]:


inn1_15.fillna(-1,inplace=True)
inn1_15.head()


# In[243]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#x_train,x_test,y_train,y_test=train_test_split(inn1_pp,total_runs)


# In[244]:


model=LinearRegression()
model.fit(inn1_pp[:100],total_runs[:100])
predicted_pp=model.predict(inn1_pp[100:])


# In[245]:


output_compare=[]
for i in range(100,100+len(predicted_pp)):
    output_compare.append((total_runs[i],predicted_pp[i-100])) 


# In[246]:


output_compare


# In[252]:


def get_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[253]:


rmse=get_rmse(predicted_pp,total_runs[100:])


# In[254]:


rmse


# In[248]:


mean_absolute_error(total_runs[100:],predicted_pp)


# In[249]:


total_runs[100:].mean()


# In[250]:


predicted_pp.mean()


# ## using 15 ovr data

# In[255]:


#x_train,x_test,y_train,y_test=train_test_split(inn1_15,total_runs)
model=LinearRegression()
model.fit(inn1_15[:100],total_runs[:100])
predicted_15=model.predict(inn1_15[100:])


# In[256]:


output_compare=[]
for i in range(100,100+len(predicted_15)):
    output_compare.append((total_runs[i],predicted_15[i-100])) 


# In[257]:


output_compare


# In[258]:


mean_absolute_error(total_runs[100:],predicted_15)


# In[260]:


get_rmse(predicted_15,total_runs[100:])


# In[261]:


total_runs[100:].mean()


# In[262]:


predicted_15.mean()


# ## avg of powerplay and 15ovr

# In[137]:


av=(predicted_15+predicted_pp)/2


# In[140]:


x=[]
for i in range(100,100+len(predicted_15)):
    x.append((total_runs[i],av[i-100])) 


# In[141]:


x

