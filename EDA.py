#!/usr/bin/env python
# coding: utf-8

# # Dataset [link](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


data = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
data.head()


# In[3]:


data["attrition"] = data["Attrition"]
data.head()


# In[4]:


data.drop(columns = ["Attrition"], inplace = True)
data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


print("{:.2f}% of the employees resigned/retired".format(data["attrition"].value_counts()["Yes"]/len(data["attrition"])*100))
sns.displot(data = data, x = "attrition")


# In[9]:


groupby_Gender = data.groupby("Gender")["attrition"]
print(groupby_Gender.value_counts())

print("\nOf the total employees resigned/retired, {:.2f}% are Female".format(groupby_Gender.value_counts()["Female"]["Yes"]/(groupby_Gender.value_counts()["Female"]["Yes"] + groupby_Gender.value_counts()["Male"]["Yes"])*100))
print("Of the total employees resigned/retired, {:.2f}% are Male".format(groupby_Gender.value_counts()["Male"]["Yes"]/(groupby_Gender.value_counts()["Female"]["Yes"] + groupby_Gender.value_counts()["Male"]["Yes"])*100))

print("Of the total Male employees, {:.2f}% resigned/retired".format(groupby_Gender.value_counts()["Male"]["Yes"]/data["Gender"].value_counts()["Male"]*100))
print("Of the total Female employees, {:.2f}% resigned/retired".format(groupby_Gender.value_counts()["Female"]["Yes"]/data["Gender"].value_counts()["Female"]*100))

sns.displot(data = data, x = "Gender", hue = "attrition")


# In[10]:


groupby_BusinessTravel = data.groupby("attrition")["BusinessTravel"]
print(groupby_BusinessTravel.value_counts())

sns.displot(data = data, x = "BusinessTravel", hue = "attrition")


# In[11]:


print("minimum age of employee is {}".format(pd.Series(data["Age"]).min()))
print("maximum age of employee is {}".format(pd.Series(data["Age"]).max()))

plt.figure()
sns.histplot(data = data, x = "Age", bins = 7)
plt.figure()
sns.histplot(data = data, x = "Age", bins = 3, hue = "attrition")
plt.figure()
sns.histplot(data = data, x = "Age", bins = 7, hue = "attrition")


# In[12]:


gropuby_Education = data.groupby("attrition")["Education"]
print(100*gropuby_Education.value_counts()["Yes"]/sum(gropuby_Education.value_counts()["Yes"]))

print("\nOf the total employees, {:.2f}% belong to Education = 3".format(100*(gropuby_Education.value_counts()[0] + gropuby_Education.value_counts()[5])/sum(gropuby_Education.value_counts())))
print("Of the total employees resigned/retired, {:.2f}% were with Education = 3".format(gropuby_Education.value_counts()["Yes"][3]/sum(gropuby_Education.value_counts()["Yes"])*100))

sns.displot(data = data, x = "Education", hue = "attrition")
sns.displot(data = data, x = "Education", hue = "attrition", col = "Gender")
sns.displot(data = data, x = "Education", hue = "attrition", col = "MaritalStatus")


# In[13]:


print("minimum DailyRate of employee is {}".format(pd.Series(data["DailyRate"]).min()))
print("maximum DailyRate of employee is {}".format(pd.Series(data["DailyRate"]).max()))

plt.figure()
sns.histplot(data = data, x = "DailyRate", bins = 7, hue = "attrition")


# In[14]:


groupby_Department = data.groupby("attrition")["Department"]
print(groupby_Department.value_counts())

sns.displot(data = data, x = "Department", hue = "attrition", height = 6.5)


# In[15]:


groupby_EducationField = data.groupby("attrition")["EducationField"]
print(groupby_EducationField.value_counts())

sns.displot(data=data, x="EducationField", hue = "attrition", height = 10)


# In[16]:


sns.displot(data=data, x="EnvironmentSatisfaction", hue = "attrition")


# In[17]:


print("minimum HourlyRate of employee is {}".format(pd.Series(data["HourlyRate"]).min()))
print("maximum HourlyRate of employee is {}".format(pd.Series(data["HourlyRate"]).max()))

plt.figure()
sns.histplot(data = data, x = "HourlyRate", bins = 7, hue = "attrition")


# In[18]:


sns.displot(data=data, x="JobInvolvement", hue = "attrition")


# In[19]:


sns.displot(data = data, x = "JobLevel", hue = "attrition")


# In[20]:


sns.displot(data=data, x="JobRole", hue = "attrition", height = 10, aspect = 1.9)


# In[21]:


groupby_JobSatisfaction = data.groupby("attrition")["JobSatisfaction"]
print(groupby_JobSatisfaction.value_counts())

sns.displot(data=data, x="JobSatisfaction", hue = "attrition")


# In[22]:


print("minimum MonthlyIncome of employee is {}".format(pd.Series(data["MonthlyIncome"]).min()))
print("maximum MonthlyIncome of employee is {}".format(pd.Series(data["MonthlyIncome"]).max()))

plt.figure()
sns.histplot(data = data, x = "MonthlyIncome", bins = 5, hue = "attrition")


# In[23]:


sns.displot(data = data, x = "MaritalStatus", hue = "attrition")


# In[24]:


sns.displot(data = data, x = "NumCompaniesWorked", hue = "attrition")


# In[25]:


sns.displot(data = data, x = "OverTime", hue = "attrition")


# In[26]:


print("minimum PercentSalaryHike of employee is {}".format(pd.Series(data["PercentSalaryHike"]).min()))
print("maximum PercentSalaryHike of employee is {}\n".format(pd.Series(data["PercentSalaryHike"]).max()))

groupby_PercentSalaryHike = data.groupby("attrition")["PercentSalaryHike"]
print(groupby_PercentSalaryHike.value_counts())

sns.displot(data = data, x = "PercentSalaryHike", hue = "attrition")
sns.displot(data = data, x = "PercentSalaryHike", hue = "attrition", col = "OverTime")


# In[27]:


print("minimum PerformanceRating of employee is {}".format(pd.Series(data["PerformanceRating"]).min()))
print("maximum PerformanceRating of employee is {}\n".format(pd.Series(data["PerformanceRating"]).max()))

groupby_PerformanceRating = data.groupby(by = ["attrition"])["PerformanceRating"]
print(groupby_PerformanceRating.value_counts(), "\n")

groupby_PerformanceRating_OverTime = data.groupby(by = ["attrition", "PerformanceRating"])["OverTime"]
print(groupby_PerformanceRating_OverTime.value_counts())

sns.displot(data = data, x = "PerformanceRating", hue = "attrition")
sns.displot(data = data, x = "PerformanceRating", hue = "attrition", col = "OverTime")


# In[28]:


print("minimum RelationshipSatisfaction of employee is {}".format(pd.Series(data["RelationshipSatisfaction"]).min()))
print("maximum RelationshipSatisfaction of employee is {}\n".format(pd.Series(data["RelationshipSatisfaction"]).max()))

groupby_RelationshipSatisfaction = data.groupby(by = ["attrition"])["RelationshipSatisfaction"]
print(groupby_RelationshipSatisfaction.value_counts(), "\n")

sns.displot(data = data, x = "RelationshipSatisfaction", hue = "attrition")


# In[29]:


print("minimum StockOptionLevel of employee is {}".format(pd.Series(data["StockOptionLevel"]).min()))
print("maximum StockOptionLevel of employee is {}\n".format(pd.Series(data["StockOptionLevel"]).max()))

groupby_StockOptionLevel = data.groupby(by = ["attrition"])["StockOptionLevel"]
print(groupby_StockOptionLevel.value_counts(), "\n")

sns.displot(data = data, x = "StockOptionLevel", hue = "attrition")
sns.displot(data = data, x = "StockOptionLevel", hue = "attrition", col = "OverTime")


# In[30]:


print("minimum TotalWorkingYears of employee is {}".format(pd.Series(data["TotalWorkingYears"]).min()))
print("maximum TotalWorkingYears of employee is {}".format(pd.Series(data["TotalWorkingYears"]).max()))

plt.figure()
sns.histplot(data = data, x = "TotalWorkingYears", bins = 4, hue = "attrition")
plt.figure()
sns.histplot(data = data, x = "TotalWorkingYears", bins = 8, hue = "attrition")


# In[31]:


print("minimum WorkLifeBalance of employee is {}".format(pd.Series(data["WorkLifeBalance"]).min()))
print("maximum WorkLifeBalance of employee is {}\n".format(pd.Series(data["WorkLifeBalance"]).max()))

groupby_WorkLifeBalance = data.groupby(by = ["attrition"])["WorkLifeBalance"]
print(groupby_WorkLifeBalance.value_counts())

groupby_WorkLifeBalance = data.groupby(by = ["attrition", "WorkLifeBalance"])["OverTime"]
print(groupby_WorkLifeBalance.value_counts())

sns.displot(data = data, x = "WorkLifeBalance", hue = "attrition")
sns.displot(data = data, x = "WorkLifeBalance", hue = "attrition", col = "OverTime")


# In[32]:


print("minimum YearsInCurrentRole of employee is {}".format(pd.Series(data["YearsInCurrentRole"]).min()))
print("maximum YearsInCurrentRole of employee is {}".format(pd.Series(data["YearsInCurrentRole"]).max()))

plt.figure()
sns.histplot(data = data, x = "YearsInCurrentRole", bins = 3, hue = "attrition")
plt.figure()
sns.histplot(data = data, x = "YearsInCurrentRole", bins = 6, hue = "attrition")


# In[33]:


print("minimum YearsSinceLastPromotion of employee is {}".format(pd.Series(data["YearsSinceLastPromotion"]).min()))
print("maximum YearsSinceLastPromotion of employee is {}".format(pd.Series(data["YearsSinceLastPromotion"]).max()))

plt.figure()
sns.histplot(data = data, x = "YearsSinceLastPromotion", bins = 5, hue = "attrition")


# In[34]:


data["attrition"] = data["attrition"].astype('category').cat.codes
data["BusinessTravel"] = data["BusinessTravel"].astype('category').cat.codes
data["Department"] = data["Department"].astype('category').cat.codes
data["EducationField"] = data["EducationField"].astype('category').cat.codes
data["Gender"] = data["Gender"].astype('category').cat.codes
data["JobRole"] = data["JobRole"].astype('category').cat.codes

data["MaritalStatus"] = data["MaritalStatus"].astype('category').cat.codes
data["Over18"] = data["Over18"].astype('category').cat.codes
data["OverTime"] = data["OverTime"].astype('category').cat.codes
data


# In[35]:


data.drop(columns = ["StandardHours", "Over18", "EmployeeCount"], inplace = True)


# In[36]:


corr = data.corr("pearson")
plt.figure(figsize = (10,10))
sns.heatmap(corr, linewidth = 1)
corr


# In[37]:


plt.figure(figsize = (10,10))
sns.heatmap(abs(corr) > 0.75, linewidth = 1)


# In[ ]:




