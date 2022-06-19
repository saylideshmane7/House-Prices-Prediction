#!/usr/bin/env python
# coding: utf-8

# # House_Price_Predication_Project 
# 

# ## Project_Goal
Predict the price of a house by its features. If you are a buyer or sellor of the house, so supervised machine learning algorithms can help you to predict the price of the house just providing features of the target house. 
# In[5]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[6]:


train1=pd.read_csv(r"C:\Users\TEST\Property_Price_Train.csv")
train1


# In[7]:


train1.shape


# In[8]:


train1.describe()


# In[9]:


int_features =train1.select_dtypes(include=["int64"]).columns 
print("Total Number of integer features : ", int_features.shape[0])
print("integer features names:", int_features.tolist())
float_features =train1.select_dtypes(include=["float64"]).columns 
print("Total Number of floating  features : ", float_features.shape[0])
print("floating features names:", float_features.tolist())
cat_features =train1.select_dtypes(include=["object"]).columns 
print("Total Number of catagorical  features : ", cat_features.shape[0])
print("catagorical features names:", cat_features.tolist())


# In[10]:


train2=train1.copy()
train2.shape


# ## Visualise Null/Missing Values 

# In[12]:


pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
train2.isnull().sum()


# In[13]:


plt.figure(figsize=(16,9))
sns.heatmap(train2.isnull())
plt.savefig("TR_IM/heatmap_train2_of_nulll_values")


# In[14]:


null_var=train2.isnull().sum()/train2.shape[0]*100
null_var


# In[15]:


null_var[null_var>17].keys()


# In[16]:


train2=train2.drop(["Lane_Type","Fireplace_Quality","Pool_Quality","Fence_Quality","Miscellaneous_Feature"],axis=1)
train2


# In[35]:


train2.Lot_Extent.value_counts()


# In[36]:


train2.Lot_Extent=train2.Lot_Extent.fillna(80)


# In[37]:


train2["Lot_Extent"].isnull().sum()


# In[38]:


plt.figure(figsize=(16,9))
sns.heatmap(train2.isnull())
plt.savefig("TR_IM/heatmap_train2_of_nulll_values")


# In[20]:


train2.isnull().sum()


# In[19]:


train2.Basement_Height.value_counts()
train2.Basement_Height=train2.Basement_Height.fillna("TA") 


# In[21]:


train2.Basement_Condition.value_counts()
train2.Basement_Condition=train2.Basement_Condition.fillna("Gd")


# In[22]:


train2.Exposure_Level.value_counts()
train2.Exposure_Level=train2.Exposure_Level.fillna("Mn")


# In[23]:


train2.BsmtFinType1.value_counts()
train2.BsmtFinType1=train2.BsmtFinType1.fillna("BLQ")


# In[24]:


train2.BsmtFinType2.value_counts()
train2.BsmtFinType2 =train2.BsmtFinType2 .fillna("LwQ")


# In[25]:


train2.Garage.value_counts()
train2.Garage=train2.Garage.fillna("BuiltIn")


# In[26]:


train2.Garage_Built_Year.value_counts()
train2.Garage_Built_Year=train2.Garage_Built_Year.fillna(2005)


# In[28]:


train2.Garage_Finish_Year.value_counts()
train2.Garage_Finish_Year =train2.Garage_Finish_Year .fillna("Fin")


# In[29]:


train2.Garage_Quality.value_counts()
train2.Garage_Quality=train2.Garage_Quality.fillna("TA")


# In[30]:


train2.Garage_Condition.value_counts()
train2.Garage_Condition=train2.Garage_Condition.fillna("TA")


# In[31]:


train2.Brick_Veneer_Type.value_counts()
train2.Brick_Veneer_Type=train1.Brick_Veneer_Type.fillna("Stone")


# In[32]:


train2.Brick_Veneer_Area.value_counts()
train2.Brick_Veneer_Area=train2.Brick_Veneer_Area.fillna(119)


# In[33]:


train2.Electrical_System.value_counts()
train2.Electrical_System=train2.Electrical_System.fillna("FuseP")


# In[39]:


train2.isnull().sum()


# In[40]:


train2.isnull().sum().sum()


# In[41]:


plt.figure(figsize=(16,9))
sns.heatmap(train2.isnull())
plt.savefig("TR_IM/heatmap_train2_of_nulll_values")


# ## Change Categorical To Numerical (By Replace Method/LabelEncoder)

# In[42]:


train2.dtypes


# In[43]:


train2.Zoning_Class.value_counts()
train2.Zoning_Class.replace({"RLD":0,"RMD":1,"FVR":2,"RHD":3,"Commer":4},inplace=True)


# In[44]:


train2.Road_Type.value_counts()
train2.Road_Type.replace({"Paved":0,"Gravel":1},inplace=True)


# In[45]:


train2.Property_Shape.value_counts()
train2.Property_Shape.replace({"Reg":0,"IR1":1,"IR2":2,"IR3":4},inplace=True)


# In[46]:


train2.Land_Outline.value_counts()
train2.Land_Outline.replace({"Lvl":0,"Bnk":1,"HLS":2,"Low":3},inplace=True)


# In[47]:


train2.Utility_Type.value_counts()
train2.Utility_Type.replace({"AllPub":0,"NoSeWa":1},inplace=True)


# In[48]:


train2.Lot_Configuration.value_counts()
train2.Lot_Configuration.replace({"I":0,"C":1,"CulDSac":2,"FR2P":3,"FR3P":4},inplace=True)


# In[49]:


train2.Property_Slope.value_counts()
train2.Property_Slope.replace({"GS":0,"MS":1,"SS":2},inplace=True)


# In[50]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le


# In[59]:


train2.Neighborhood=le.fit_transform(train2.Neighborhood) 
train2.Condition1=le.fit_transform(train2.Condition1)
train2.Condition2=le.fit_transform(train2.Condition2)
train2.House_Design=le.fit_transform(train2.House_Design)
train2.Foundation_Type=le.fit_transform(train2.Foundation_Type) 


# In[52]:


train2.House_Type.value_counts()
train2.House_Type.replace({"1Fam":0,"TwnhsE":1,"Duplex":2,"Twnhs":3,"2fmCon":4},inplace=True)


# In[54]:


train2.Roof_Design.value_counts()
train2.Roof_Design.replace({"Gable":0,"Hip":1,"Flat":2,"Gambrel":3,"Mansard":4,"Shed":5},inplace=True)


# In[55]:


train2.Roof_Quality=le.fit_transform(train2.Roof_Quality) 
train2.Exterior1st=le.fit_transform(train2.Exterior1st)
train2.Exterior2nd=le.fit_transform(train2.Exterior2nd) 


# In[56]:


train2.Brick_Veneer_Type.value_counts()
train2.Brick_Veneer_Type.replace({"None":0,"BrkFace":1,"Stone":2,"BrkCmn":3},inplace=True)


# In[57]:


train2.Exterior_Material .value_counts()
train2.Exterior_Material.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[58]:


train2.Exterior_Condition.value_counts()
train2.Exterior_Condition.replace({"TA":0,"Gd":1,"Fa":2,"Ex":3,"Po":4},inplace=True)


# In[60]:


train2.Basement_Height.value_counts()
train2.Basement_Height.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[61]:


train2.Basement_Condition.value_counts()
train2.Basement_Condition.replace({"TA":0,"Gd":1,"Fa":2,"Po":3},inplace=True)


# In[62]:


train2.Exposure_Level.value_counts()
train2.Exposure_Level.replace({"No":0,"Av":1,"Gd":2,"Mn":3},inplace=True)


# In[63]:


train2.BsmtFinType1.value_counts()
train2.BsmtFinType1.replace({"Unf":0,"GLQ":1,"ALQ":2,"BLQ":3,"Rec":4,"LwQ":5},inplace=True)


# In[64]:


train2.BsmtFinType2.value_counts()
train2.BsmtFinType2.replace({"Unf":0,"Rec":1,"LwQ":2,"BLQ":3,"ALQ":4,"GLQ":5},inplace=True)


# In[65]:


train2.Heating_Type.value_counts()
train2.Heating_Type.replace({"GasA":0,"GasW":1,"Grav":2,"Wall":3,"OthW":4,"Floor":5},inplace=True)


# In[66]:


train2.Heating_Quality.value_counts()
train2.Heating_Quality.replace({"Ex":0,"TA":1,"Gd":2,"Fa":3,"Po":4},inplace=True)


# In[67]:


train2.Air_Conditioning.value_counts()
train2.Air_Conditioning.replace({"Y":0,"N":1},inplace=True)


# In[68]:


train2.Electrical_System.value_counts()
train2.Electrical_System.replace({"SBrkr":0,"FuseA":1,"FuseF":2,"FuseP":3,"Mix":4},inplace=True)


# In[69]:


train2.Kitchen_Quality.value_counts()
train2.Kitchen_Quality.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[70]:


train2.Functional_Rate=le.fit_transform(train2.Functional_Rate) 
train2.Garage=le.fit_transform(train2.Garage) 


# In[71]:


train2.Garage_Quality.value_counts()
train2.Garage_Quality.replace({"TA":0,"Fa":1,"Gd":2,"Ex":3,"Po":4},inplace=True)


# In[72]:


train2.Garage_Condition.value_counts()
train2.Garage_Condition.replace({"TA":0,"Fa":1,"Gd":2,"Po":3,"Ex":4},inplace=True)


# In[73]:


train2.Pavedd_Drive.value_counts()
train2.Pavedd_Drive.replace({"Y":0,"N":1,"P":2},inplace=True)


# In[74]:


train2.Sale_Type=le.fit_transform(train2.Sale_Type)
train2.Sale_Condition=le.fit_transform(train2.Sale_Condition) 


# In[75]:


train2.Garage_Finish_Year.value_counts()
train2.Garage_Finish_Year.replace({"Unf":0,"Fin":1,"RFn":2},inplace=True)


# In[76]:


train2.dtypes


# In[77]:


train2.duplicated()


# In[78]:


train3=train2.copy()
train3.shape


# In[79]:


train3.corr()


# 

# In[80]:


# check correlation matrix, darker means more correlation
corrmat = train3.corr()
f, aX_train3 = plt.subplots(figsize=(16, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[81]:


plt.figure(figsize=(10,8))
bar=sns.distplot(train3["Sale_Price"])
bar.legend(["Skewness:{:.2f}".format(train3["Sale_Price"].skew())])


# In[82]:



plt.figure(figsize=(25,25))
heatmap=sns.heatmap(train3.corr(),cmap = "coolwarm" , annot= True , linewidth=2)
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom + 0.5, top - 0.5)
plt.title("Heatmap using seaborn Method")
plt.show()


# In[83]:


hig_corr = train3.corr()
hig_corr_features = hig_corr.index[abs(hig_corr["Sale_Price"]) >= 0.5]
hig_corr_features


# In[84]:


plt.figure(figsize=(10,8))
ht = sns.heatmap(train3[hig_corr_features].corr(), cmap = "coolwarm", annot=True, linewidth=3)
bottom, top = ht.get_ylim()
ht.set_ylim(bottom + 0.5, top - 0.5)


# In[85]:


skewed_feature = ['Construction_Year',
        'Total_Basement_Area', 'First_Floor_Area',
       'Grade_Living_Area']


# In[86]:


plt.figure(figsize=(25,20))
for i in range(len(skewed_feature)):
    if i <= 28:
        plt.subplots()
        plt.subplots_adjust(hspace = 0.5, wspace = 0.5)
        sq = sns.distplot(train3[skewed_feature[i]])
        sq.legend(["Skeweness: {:.2f}".format(train3[skewed_feature[i]].skew())], fontsize = 'xx-large' )
        


# In[87]:


train4=train3.copy()


# ## Outlier treatment 

# In[88]:


train3.Sale_Price.hist(bins=50)


# In[89]:


train3["Sale_Price"].describe()
figure2=train3.boxplot(column="Sale_Price" )


# In[90]:


IQR=train3.Sale_Price.quantile(0.75)-train3.Sale_Price.quantile(0.25)
IQR


# In[91]:


Lower_bridge2=train3["Sale_Price"].quantile(0.25) -(IQR*3)
Upper_bridge2=train3["Sale_Price"].quantile(0.75) +(IQR*3)
print(Lower_bridge2,Upper_bridge2) 


# In[92]:


train4.loc[train4["Sale_Price"]>466150,"Sale_Price"]=466150


# In[93]:


train4.Sale_Price.hist(bins=50)


# In[94]:


figure22=train4.boxplot(column="Sale_Price" )


# In[96]:


train4["Sale_Price"].describe()
train3.Total_Basement_Area.hist( bins=50)


# In[97]:


train3["Total_Basement_Area"].describe()
figure1=train3.boxplot(column="Total_Basement_Area" )


# In[98]:


Upper_bridge2=train3["Total_Basement_Area"].mean()+3*train3["Total_Basement_Area"].std() 
lower_bridge2=train3["Total_Basement_Area"].mean()-3*train3["Total_Basement_Area"].std()
print(Upper_bridge2,lower_bridge2) 


# In[99]:


train4.loc[train4["Total_Basement_Area"] > 2373,"Total_Basement_Area"]=2373 


# In[100]:


figure2=train4.boxplot(column="Total_Basement_Area")


# In[101]:


train4.Total_Basement_Area.hist(bins=50)


# In[102]:


train4["Total_Basement_Area"].describe()


# In[103]:


train3.First_Floor_Area.hist(bins=50 )


# In[104]:


train3["First_Floor_Area"].describe()
figure3=train3.boxplot(column="First_Floor_Area" )


# In[105]:


Upper_bridge3=train3["First_Floor_Area"].mean()+3*train3["First_Floor_Area"].std() 
lower_bridge3=train3["First_Floor_Area"].mean()-3*train3["First_Floor_Area"].std()
print(Upper_bridge3,lower_bridge3) 


# In[106]:


train4.loc[train4["First_Floor_Area"] > 2322,"First_Floor_Area"]=2322 


# In[107]:


figure3=train4.boxplot(column="First_Floor_Area")


# In[109]:


train4["First_Floor_Area"].describe()
train4.First_Floor_Area.hist(bins=50)


# In[110]:


train3.Grade_Living_Area.hist(bins=50 )


# In[111]:


train3["Grade_Living_Area"].describe()
figure3=train3.boxplot(column="Grade_Living_Area" )


# In[112]:


IQR=train3.Grade_Living_Area.quantile(0.75)-train3.Grade_Living_Area.quantile(0.25)
IQR


# In[113]:


Lower_bridge4=train3["Grade_Living_Area"].quantile(0.25) -(IQR*3)
Upper_bridge4=train3["Grade_Living_Area"].quantile(0.75) +(IQR*3)
print(Lower_bridge4,Upper_bridge4) 


# In[114]:


train4.loc[train4["Grade_Living_Area"]>3723,"Grade_Living_Area"]=3723


# In[115]:


figure4=train4.boxplot(column="Grade_Living_Area")


# In[116]:


train4["Grade_Living_Area"].describe()
train4.Grade_Living_Area.hist(bins=50)


# In[117]:




plt.figure(figsize=(10,8))
bar=sns.distplot(train4["Sale_Price"])
bar.legend(["Skewness:{:.2f}".format(train4["Sale_Price"].skew())])


# In[118]:


train5=train4.copy()
train5.shape


# # Linear Regression Model

# In[270]:


train_x=train5.iloc[:,1:-1]
train_y=train5.iloc[:,-1]
test_x=test3.iloc[:,1:]


# In[271]:


train_x.head()


# In[272]:


train_y.head()


# In[273]:


train_x.shape,train_y.shape,test_x.shape


# In[274]:


import sklearn
from sklearn.model_selection import train_test_split


# In[275]:


train_xtrain,train_xtest,train_ytrain,train_ytest=train_test_split(train_x,train_y,test_size=0.3,random_state=101 )


# In[276]:


train_xtrain.shape,train_xtest.shape,train_ytrain.shape,train_ytest.shape


# In[277]:


from sklearn import linear_model
ln=linear_model.LinearRegression()


# In[278]:


ln.fit(train_xtrain,train_ytrain)


# In[351]:


predd=ln.predict(train_xtest)


# In[280]:


ln.coef_


# In[281]:


ln.intercept_


# In[282]:


rsq1=ln.score(train_xtrain,train_ytrain)
rsq1


# In[283]:


adjrr=1-(((1-rsq1)*(1021-1)/(1021-74-1)))
adjrr


# In[284]:


r=ln.predict(train_xtrain)
r


# In[285]:


mean_y=train_ytrain.mean()
mean_y


# In[286]:


SSE=np.sum(np.square(r - train_ytrain))
SSE


# In[287]:


SSR=np.sum(np.square(r - mean_y))
SSR


# In[288]:


from sklearn import metrics


# In[289]:


d1_mae=metrics.mean_absolute_error(train_ytest,predd)
d1_mae


# In[290]:


d1_mse=metrics.mean_squared_error(train_ytest,predd)
d1_mse


# In[291]:


rmse=np.sqrt(d1_mse)
rmse


# In[292]:


dd1=pd.DataFrame(predd,columns=["Sale_Price"])
dd1


# In[293]:


dd2=pd.DataFrame(test3.iloc[:,0])
dd2


# In[294]:


dd3=pd.concat([dd2,dd1],axis=1)
dd3


# In[295]:


dd3.to_csv(r"C:\Users\TEST\Final_Lin_basic_model.csv",index=False)


# #  Lasso model

# In[296]:


from sklearn.linear_model import Lasso
lasso=Lasso()


# In[297]:


lasso.fit(train_xtrain,train_ytrain)


# In[298]:


pred1=lasso.predict(train_xtest)


# In[299]:


r2=lasso.score(train_xtrain,train_ytrain)
r2


# In[300]:


adj_r2=1-(((1-r2)*(1021-1))/(1021-74-1)) 
adj_r2


# In[301]:


error1=train_ytest-pred1
error1


# In[302]:


aerror2=np.abs(error1)
aerror2


# In[303]:


mape_22=np.mean(aerror2/train_ytest)*100
mape_22


# In[304]:


MSE_22=metrics.mean_squared_error(train_ytest,pred1)
MSE_22


# # RF Reg

# In[305]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[306]:


rf.fit(train_xtrain,train_ytrain)


# In[307]:


pred3=rf.predict(train_xtest)


# In[308]:


r3=rf.score(train_xtrain,train_ytrain)
r3


# In[309]:


adj3=1-(((1-r3)*(1021-1))/(1021-74-1)) 
adj3    


# In[310]:


error3=train_ytest-pred3
error3


# In[311]:


aerror3=np.abs(error3)
aerror3


# In[312]:


mape3=np.mean(aerror3/train_ytest)*100
mape3


# In[313]:


MSE3=metrics.mean_squared_error(train_ytest,pred3)
MSE3


# # SVM reg

# In[314]:


from sklearn.svm import LinearSVR
ls = LinearSVR()


# In[315]:


ls.fit(train_xtrain,train_ytrain)


# In[316]:


pred4=ls.predict(train_xtest)


# In[317]:


r4=ls.score(train_xtrain,train_ytrain)


# In[318]:


adj4=1-(((1-r4)*(1021-1))/(1021-74-1)) 
adj4    


# In[319]:


error4=train_ytest-pred4
error4


# In[320]:


aerror4=np.abs(error4)


# In[321]:


mape4=np.mean(aerror4/train_ytest)*100
mape4


# In[322]:


MSE4=metrics.mean_squared_error(train_ytest,pred4)
MSE4


# # KNN reg

# In[323]:


from sklearn.neighbors import KNeighborsRegressor
knn =  KNeighborsRegressor(n_neighbors = 10)


# In[324]:


knn.fit(train_xtrain,train_ytrain)


# In[325]:


pred5=knn.predict(train_xtest)


# In[326]:


r25=knn.score(train_xtrain,train_ytrain)


# In[327]:


adj_r25=1-(((1-r25)*(1021-1))/(1021-74-1)) 
adj_r25   


# In[328]:


error5=train_ytest-pred5
error5


# In[329]:


aerror5=np.abs(error5)


# In[330]:


mape5=np.mean(aerror5/train_ytest)*100
mape5


# In[331]:


MSE5=metrics.mean_squared_error(train_ytest,pred5)
MSE5


# # Feature Engineering / Selection to improve accuracy 

# In[333]:


train6=train5.copy()
train6.shape


# In[334]:


plt.figure(figsize=(9,16))
corr_feat_series = pd.Series.sort_values(train6.corrwith(train6.Sale_Price))
sns.barplot(x=corr_feat_series, y=corr_feat_series.index, orient='h')


# In[335]:


list(corr_feat_series.index)


# # DROP FEATURES 

# In[338]:


train6 = train6.drop(['Road_Type',
'Year_Sold',
 'Id',
 'LowQualFinSF',
 'Underground_Half_Bathroom',
 'Miscellaneous_Value',
 'Utility_Type',
 'Open_Lobby_Area',
 'BsmtFinSF2',
 'Garage_Area',
 'Condition2',
 'Enclosed_Lobby_Area',
 'W_Deck_Area'],axis=1)


# # Linear Regression Model

# In[339]:


train_x=train6.iloc[:,1:-1]
train_y=train6.iloc[:,-1]
test_x=test3.iloc[:,1:]


# In[340]:


train_x.head()


# In[341]:


train_y.head()


# In[342]:


train_x.shape,train_y.shape,test_x.shape


# In[343]:


import sklearn
from sklearn.model_selection import train_test_split


# In[344]:


train_xtrain,train_xtest,train_ytrain,train_ytest=train_test_split(train_x,train_y,test_size=0.3,random_state=101 )


# In[345]:


train_xtrain.shape,train_xtest.shape,train_ytrain.shape,train_ytest.shape


# In[346]:


from sklearn import linear_model
ln=linear_model.LinearRegression()


# In[347]:


ln.fit(train_xtrain,train_ytrain)


# In[349]:


preddd=ln.predict(train_xtest)


# In[352]:


ln.intercept_


# In[353]:


rsqq=ln.score(train_xtrain,train_ytrain)
rsqq


# In[354]:


adjrrr=1-(((1-rsqq)*(1021-1)/(1021-61-1)))
adjrrr


# In[355]:


rr=ln.predict(train_xtrain)
rr


# In[356]:


mean_y=train_ytrain.mean()
mean_y


# In[357]:


SSR=np.sum(np.square(r - mean_y))
SSR


# In[358]:


from sklearn import metrics


# In[359]:


d1_mae=metrics.mean_absolute_error(train_ytest,preddd)
d1_mae


# In[360]:


d1_mse=metrics.mean_squared_error(train_ytest,preddd)
d1_mse


# In[361]:


rmse=np.sqrt(d1_mse)
rmse


# In[362]:


ddd1=pd.DataFrame(preddd,columns=["Sale_Price"])
ddd1


# In[363]:


ddd2=pd.DataFrame(test3.iloc[:,0])
ddd2


# In[364]:


ddd3=pd.concat([dd2,dd1],axis=1)
ddd3


# In[365]:


ddd3.to_csv(r"C:\Users\TEST\features _Final_Lin_model.csv",index=False)


# # Lasso model

# In[366]:


lasso.fit(train_xtrain,train_ytrain)


# In[367]:


pred1=lasso.predict(train_xtest)


# In[368]:


rr2=lasso.score(train_xtrain,train_ytrain)
rr2


# In[369]:


adj_rr2=1-(((1-r2)*(1021-1))/(1021-61-1)) 
adj_rr2


# In[370]:


errorr1=train_ytest-pred1
errorr1


# In[371]:


aerrorr2=np.abs(errorr1)
aerrorr2


# In[372]:


mape_222=np.mean(aerrorr2/train_ytest)*100
mape_222


# In[373]:


MSE_222=metrics.mean_squared_error(train_ytest,pred1)
MSE_222


# # RF Reg

# In[374]:


rf.fit(train_xtrain,train_ytrain)


# In[376]:


pred3=rf.predict(train_xtest)


# In[377]:


r3=rf.score(train_xtrain,train_ytrain)
r3


# In[378]:


adj3=1-(((1-r3)*(1021-1))/(1021-61-1)) 
adj3    


# In[379]:


error3=train_ytest-pred3
error3


# In[380]:


aerror3=np.abs(error3)
aerror3


# In[381]:


mape3=np.mean(aerror3/train_ytest)*100
mape3


# In[382]:


MSE3=metrics.mean_squared_error(train_ytest,pred3)
MSE3


# # SVM reg

# In[383]:


ls.fit(train_xtrain,train_ytrain)


# In[384]:


pred4=ls.predict(train_xtest)


# In[385]:


r4=ls.score(train_xtrain,train_ytrain)


# In[386]:


adj4=1-(((1-r4)*(1021-1))/(1021-61-1)) 
adj4    


# In[387]:


error4=train_ytest-pred4
error4


# In[388]:


aerror4=np.abs(error4)


# In[389]:


mape4=np.mean(aerror4/train_ytest)*100
mape4


# In[390]:


MSE4=metrics.mean_squared_error(train_ytest,pred4)
MSE4


# # KNN reg

# In[391]:


knn.fit(train_xtrain,train_ytrain)


# In[392]:


pred5=knn.predict(train_xtest)


# In[393]:


r25=knn.score(train_xtrain,train_ytrain)


# In[394]:


adj_r25=1-(((1-r25)*(1021-1))/(1021-61-1)) 
adj_r25   


# In[395]:


error5=train_ytest-pred5
error5


# In[396]:


aerror5=np.abs(error5)


# In[397]:


mape5=np.mean(aerror5/train_ytest)*100
mape5


# In[398]:


MSE5=metrics.mean_squared_error(train_ytest,pred5)
MSE5


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Test Data

# In[134]:


test1=pd.read_csv(r"C:\Users\TEST\Property_Price_Test.csv")
test1


# In[135]:


test1.info()


# In[136]:


test1.shape


# In[137]:


test1.describe() 


# In[138]:


test1.shape


# In[139]:


int_features =test1.select_dtypes(include=["int64"]).columns 
print("Total Number of integer features : ", int_features.shape[0])
print("integer features names:", int_features.tolist())

float_features =test1.select_dtypes(include=["float64"]).columns 
print("Total Number of floating  features : ", float_features.shape[0])
print("floating features names:", float_features.tolist())

cat_features =test1.select_dtypes(include=["object"]).columns 
print("Total Number of catagorical  features : ", cat_features.shape[0])
print("catagorical features names:", cat_features.tolist())


# In[140]:


test2=test1.copy()
test2.shape


# ## Null Values

# In[141]:


test2.isnull().sum()


# In[142]:


null_varr=test2.isnull().sum()/test2.shape[0]*100
null_varr


# In[143]:


null_varr[null_varr>17].keys()


# In[144]:


test2=test2.drop(["Lane_Type","Fireplace_Quality","Pool_Quality","Fence_Quality","Miscellaneous_Feature"],axis=1)
#test2


# In[145]:


test2.Lot_Extent.value_counts()
test2.Lot_Extent=test2.Lot_Extent.fillna(test2.Lot_Extent.median())


# In[146]:


test2.Zoning_Class.value_counts()
test2.Zoning_Class=test2.Zoning_Class.fillna("RHD")


# In[147]:


test2.Utility_Type.value_counts()
test2.Utility_Type=test2.Utility_Type.fillna("AllPub")


# In[148]:


test2.Exterior1st.value_counts()
test2.Exterior1st=test2.Exterior1st.fillna("CB")


# In[240]:


test2.Exterior1st=test2.Exterior1st.fillna("CB")


# In[149]:


test2.Exterior2nd.value_counts()
test2.Exterior2nd=test2.Exterior2nd.fillna("Stone")


# In[150]:


test2.Brick_Veneer_Type.value_counts()
test2.Brick_Veneer_Type=test2.Brick_Veneer_Type.fillna("Stone")


# In[151]:


test2.Brick_Veneer_Area.value_counts()
test2.Brick_Veneer_Area=test2.Brick_Veneer_Area.fillna(test2.Brick_Veneer_Area.median())


# In[152]:


test2.Basement_Height.value_counts()
test2.Basement_Height=test2.Basement_Height.fillna("Fa")


# In[153]:


test2.Basement_Condition.value_counts()
test2.Basement_Condition=test2.Basement_Condition.fillna("Fa")


# In[154]:


test2.Exposure_Level.value_counts()
test2.Exposure_Level=test2.Exposure_Level.fillna("Mn")


# In[155]:


test2.BsmtFinType1.value_counts()
test2.BsmtFinType1=test2.BsmtFinType1.fillna("LwQ")


# In[156]:


test2.BsmtFinSF1.value_counts()
test2.BsmtFinSF1=test2.BsmtFinSF1.fillna(test2.BsmtFinSF1.median())


# In[157]:


test2.BsmtFinType2.value_counts()
test2.BsmtFinType2=test2.BsmtFinType2.fillna("Rec")


# In[158]:


test2.BsmtFinSF2.value_counts()
test2.BsmtFinSF2=test2.BsmtFinSF2.fillna(test2.BsmtFinSF2.median())


# In[159]:


test2.BsmtUnfSF.value_counts()
test2.BsmtUnfSF=test2.BsmtUnfSF.fillna(test2.BsmtUnfSF.median())


# In[160]:


test2.Total_Basement_Area.value_counts()
test2.Total_Basement_Area=test2.Total_Basement_Area.fillna(test2.Total_Basement_Area.median())


# In[161]:


test2.Underground_Full_Bathroom.value_counts()
test2.Underground_Full_Bathroom=test2.Underground_Full_Bathroom.fillna(test2.Underground_Full_Bathroom.median())


# In[162]:


plt.figure(figsize=(16,9))
sns.heatmap(test2.isnull())
plt.savefig("TR_IM/heatmap_test2_of_nulll_values")


# In[163]:


test2.Kitchen_Quality.value_counts()
test2.Kitchen_Quality=test2.Kitchen_Quality.fillna("Fa")


# In[164]:


test2.Functional_Rate.value_counts()
test2.Functional_Rate=test2.Functional_Rate.fillna("MD2")


# In[165]:


test2.Garage.value_counts()
test2.Garage=test2.Garage.fillna("BuiltIn")


# In[166]:


test2.Garage_Built_Year.value_counts()
test2.Garage_Built_Year=test2.Garage_Built_Year.fillna(test2.Garage_Built_Year.median())


# In[167]:


test2.Garage_Finish_Year.value_counts()
test2.Garage_Finish_Year=test2.Garage_Finish_Year.fillna("Fin")


# In[168]:


test2.Garage_Size.value_counts()
test2.Garage_Size=test2.Garage_Size.fillna(test2.Garage_Size.median())


# In[169]:


test2.Garage_Area.value_counts()
test2.Garage_Area=test2.Garage_Area.fillna(test2.Garage_Area.median())


# In[170]:


test2.Garage_Quality.value_counts()
test2.Garage_Quality=test2.Garage_Quality.fillna("Fa")


# In[171]:


test2.Garage_Condition.value_counts()
test2.Garage_Condition=test2.Garage_Condition.fillna("TA")


# In[172]:


test2.Sale_Type.value_counts()
test2.Sale_Type=test2.Sale_Type.fillna("ConLD")


# In[173]:


test2.Underground_Half_Bathroom.value_counts()
test2.Underground_Half_Bathroom=test2.Underground_Half_Bathroom.fillna(test2.Underground_Half_Bathroom.median())


# In[174]:


test2.isnull().sum()


# In[175]:


plt.figure(figsize=(16,9))
sns.heatmap(test2.isnull())
plt.savefig("TR_IM/heatmap_test2_of_nulll_values")


# In[176]:


test2.dtypes


# In[177]:


from sklearn.preprocessing import LabelEncoder 
le=LabelEncoder()
le


# In[206]:


test2.Road_Type=le.fit_transform(test2.Road_Type) 
test2.Utility_Type=le.fit_transform(test2.Utility_Type)
test2.Neighborhood=le.fit_transform(test2.Neighborhood)
test2.Condition1=le.fit_transform(test2.Condition1)
test2.House_Design=le.fit_transform(test2.House_Design)
test2.Exterior1st=le.fit_transform(test2.Exterior1st) 
test2.Exterior2nd=le.fit_transform(test2.Exterior2nd) 
test2.Air_Conditioning=le.fit_transform(test2.Air_Conditioning) 
test2.Functional_Rate=le.fit_transform(test2.Functional_Rate)
test2.Garage=le.fit_transform(test2.Garage) 
test2.Sale_Type=le.fit_transform(test2.Sale_Type) 
test2.Sale_Condition=le.fit_transform(test2.Sale_Condition)   


# In[179]:


test2.Property_Shape.value_counts() 
test2.Property_Shape.replace({"Reg":0,"IR1":1,"IR2":2,"IR3":3},inplace=True)


# In[180]:


test2.Land_Outline.value_counts() 
test2.Land_Outline.replace({"Lvl":0,"HLS":1,"Bnk":2,"Low":3},inplace=True)


# In[181]:


test2.Lot_Configuration.value_counts()
test2.Lot_Configuration.replace({"I":0,"C":1,"CulDSac":2,"FR2P":3,"FR3P":4},inplace=True)


# In[182]:


test2.Property_Slope.value_counts() 
test2.Property_Slope.replace({"GS":0,"MS":1,"SS":2},inplace=True)


# In[184]:


test2.Condition2.value_counts()
test2.Condition2.replace({"Norm":0,"NoRMD":1,"Feedr":2,"PosA":3,"Artery":4,"PosN":5},inplace=True)


# In[185]:


test2.House_Type.value_counts()
test2.House_Type.replace({"1Fam":0,"TwnhsE":1,"Duplex":2,"Twnhs":3,"2fmCon":4},inplace=True)


# In[187]:


test2.Roof_Design.value_counts() 
test2.Roof_Design.replace({"Gable":0,"Hip":1,"Gambrel":2,"Flat":3,"Mansard":4,"Shed":5},inplace=True)


# In[188]:


test2.Roof_Quality.value_counts() 
test2.Roof_Quality.replace({"SS":0,"TG":1,"WS":2,"WSh":3},inplace=True)


# In[191]:


test2.Brick_Veneer_Type.value_counts() 
test2.Brick_Veneer_Type.replace({"None":0,"BrkFace":1,"Stone":2,"BrkCmn":3},inplace=True)


# In[192]:


test2.Exterior_Material.value_counts()
test2.Exterior_Material.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[193]:


test2.Exterior_Condition.value_counts() 
test2.Exterior_Condition.replace({"TA":0,"Gd":1,"Fa":2,"Ex":3,"Po":4},inplace=True)


# In[194]:


test2.Foundation_Type.value_counts() 
test2.Foundation_Type.replace({"PC":0,"CB":1,"BT":2,"SL":3,"S":4,"W":5},inplace=True)


# In[195]:


test2.Basement_Height.value_counts()
test2.Basement_Height.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[196]:


test2.Basement_Condition.value_counts()
test2.Basement_Condition.replace({"TA":0,"Fa":1,"Gd":2,"Po":3},inplace=True)


# In[197]:


test2.Exposure_Level.value_counts()
test2.Exposure_Level.replace({"No":0,"Av":1,"Mn":2,"Gd":3},inplace=True)


# In[198]:


test2.BsmtFinType1.value_counts() 
test2.BsmtFinType1.replace({"GLQ":0,"Unf":1,"ALQ":2,"Rec":3,"LwQ":4,"BLQ":5},inplace=True)


# In[199]:


test2.BsmtFinType2.value_counts() 
test2.BsmtFinType2.replace({"Unf":0,"Rec":1,"LwQ":2,"BLQ":3,"ALQ":4,"GLQ":5},inplace=True)


# In[200]:


test2.Heating_Type.value_counts() 
test2.Heating_Type.replace({"GasA":0,"GasW":1,"Grav":2,"Wall":3},inplace=True)


# In[201]:


test2.Heating_Quality.value_counts() 
test2.Heating_Quality.replace({"Ex":0,"TA":1,"Gd":2,"Fa":3,"Po":4},inplace=True)


# In[203]:


test2.Electrical_System.value_counts() 
test2.Electrical_System.replace({"SBrkr":0,"FuseA":1,"FuseF":2,"FuseP":3},inplace=True)


# In[204]:


test2.Kitchen_Quality.value_counts() 
test2.Kitchen_Quality.replace({"TA":0,"Gd":1,"Ex":2,"Fa":3},inplace=True)


# In[207]:


test2.Garage_Finish_Year.value_counts() 
test2.Garage_Finish_Year.replace({"Unf":0,"Fin":1,"RFn":2},inplace=True)


# In[208]:


test2.Garage_Quality.value_counts() 
test2.Garage_Quality.replace({"TA":0,"Fa":1,"Gd":2,"Po":3},inplace=True)


# In[209]:


test2.Garage_Condition.value_counts() 
test2.Garage_Condition.replace({"TA":0,"Fa":1,"Po":2,"Gd":3,"Ex":4},inplace=True)


# In[210]:


test2.Pavedd_Drive.value_counts() 
test2.Pavedd_Drive.replace({"Y":0,"N":1,"P":2},inplace=True)


# In[211]:


test2.Zoning_Class.value_counts() 
test2.Zoning_Class.replace({"RLD":0,"RMD":1,"FVR":2,"Commer":3,"RHD":4},inplace=True)


# In[212]:


test2.duplicated() 


# In[372]:


test2.dtypes


# In[213]:


test3=test2.copy()
test3.shape


# ## Linear Regression Model

# In[214]:


train_x=train3.iloc[:,1:-1]
train_y=train3.iloc[:,-1]
test_x=test3.iloc[:,1:]


# In[215]:


train_x.head()


# In[216]:


train_y.head()


# In[377]:


test_x.head()


# In[217]:


train_x.shape,train_y.shape,test_x.shape


# In[218]:


import sklearn
from sklearn.model_selection import train_test_split


# In[219]:


train_xtrain,train_xtest,train_ytrain,train_ytest=train_test_split(train_x,train_y,test_size=0.3,random_state=101 )


# In[221]:


train_xtrain.shape,train_xtest.shape,train_ytrain.shape,train_ytest.shape


# In[222]:


from sklearn import linear_model
ln=linear_model.LinearRegression()


# In[223]:


ln.fit(train_xtrain,train_ytrain)


# In[224]:


predd=ln.predict(train_xtest)
predd


# In[225]:


ln.coef_


# In[226]:


ln.intercept_


# In[227]:


rsq1=ln.score(train_xtrain,train_ytrain)
rsq1


# In[228]:


adjrr=1-(((1-rsq1)*(1021-1)/(1021-74-1)))
adjrr


# In[229]:


r=ln.predict(train_xtrain)


# In[230]:


mean_y=train_ytrain.mean()


# In[231]:


SSE=np.sum(np.square(r - train_ytrain))
SSE


# In[232]:


SSR=np.sum(np.square(r - mean_y))
SSR


# In[233]:


from sklearn import metrics


# In[234]:


d1_mae=metrics.mean_absolute_error(train_ytest,predd)
d1_mae


# In[235]:


d1_mse=metrics.mean_squared_error(train_ytest,predd)
d1_mse


# In[236]:


rmse=np.sqrt(d1_mse)
rmse


# In[237]:


r=ln.predict(test_x)


# In[238]:


mean_y=test_x.mean()

SSE=np.sum(np.square(r - test_x))
SSE
# In[239]:


d1=pd.DataFrame(predd,columns=["Sale_Price"])
d1


# In[240]:


#d2=pd.DataFrame(test3.iloc[:,0])
#d2

d3=pd.concat([d2,d1],axis=1)
d3d3.to_csv(r"C:\Users\TEST\Lin_basic_model.csv",index=False)
# ## KNN

# In[241]:


from sklearn.neighbors import KNeighborsClassifier
km=KNeighborsClassifier(n_neighbors=17)


# In[242]:


km.fit(train_xtrain,train_ytrain)


# In[243]:


predd=km.predict(train_xtest)


# In[244]:


r22=km.score(train_xtrain,train_ytrain)
r22


# In[245]:


adj_r22=1-(((1-r22)*(1021-1))/(1021-74-1)) 
adj_r22   


# In[246]:


error9=train_ytest-predd
error9


# In[247]:


aerror9=np.abs(error9)


# In[248]:


mape_10=np.mean(aerror9/train_ytest)*100
mape_10


# In[249]:


MSE_10=metrics.mean_squared_error(train_ytest,predd)
MSE_10


# # RF

# In[250]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()


# In[251]:


rf.fit(train_xtrain,train_ytrain)


# In[252]:


pred2=rf.predict(train_xtest)


# In[253]:


r2_6=rf.score(train_xtrain,train_ytrain)
r2_6


# In[254]:


adj_r2_6=1-(((1-r2_6)*(1021-1))/(1021-74-1)) 
adj_r2_6    


# In[255]:


error=train_ytest-pred2
error


# In[256]:


aerror=np.abs(error)
aerror


# In[257]:


mape_6=np.mean(aerror/train_ytest)*100
mape_6


# In[258]:


MSE_6=metrics.mean_squared_error(train_ytest,pred2)
MSE_6


# ## Lasso

# In[259]:


from sklearn.linear_model import Lasso
la=Lasso()


# In[260]:


la.fit(train_xtrain,train_ytrain)


# In[261]:


pred3=la.predict(train_xtest)
pred3


# In[262]:


la.coef_


# In[263]:


rq=la.score(train_xtrain,train_ytrain)
rq


# In[264]:


adr=1-(((1-rq)*(1021-1)/(1021-74-1)))
adr


# In[265]:


error3=train_ytest-pred3
error3


# In[266]:


aerror3=np.abs(error3)
aerror3


# In[267]:


mape_3=np.mean(aerror3/train_ytest)*100
mape_3


# In[268]:


MSE_3=metrics.mean_squared_error(train_ytest,pred3)
MSE_3

