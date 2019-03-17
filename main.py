
# coding: utf-8

# In[23]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import re

from scipy import stats
from functools import reduce

get_ipython().magic(u'matplotlib inline')


# In[24]:


# ustawienie wyświetlania wszystkich kolumn i wczytanie danych
pd.set_option('display.max_columns', 40)
pd.set_option('float_format', '{:.3f}'.format)
data = pd.read_csv('ibm_hr_data.csv', low_memory=False)


# In[25]:


# ustawienie czytelnej kolejności kolumn 
col_core = ['Gender', 'Age', 'MaritalStatus']
col_exp = ['Education', 'EducationField', 'NumCompaniesWorked', 'TotalWorkingYears']
col_job = ['Department', 'JobRole', 'JobLevel', 'JobInvolvement']
col_job_general = ['Employee Source', 'DistanceFromHome', 'DailyRate', 'BusinessTravel', 'Attrition']
col_years = ['YearsAtCompany', 'YearsSinceLastPromotion', 'YearsInCurrentRole', 'YearsWithCurrManager', ]
col_opinion = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'WorkLifeBalance']
col_more_money = ['MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'PerformanceRating', 'StockOptionLevel', 'TrainingTimesLastYear']
col_not_important = ['StandardHours', 'OverTime', 'Over18', 'HourlyRate']
col_weird = ['Application ID', 'EmployeeCount', 'EmployeeNumber']

new_col_order = col_core + col_exp + col_job + col_job_general + col_years + col_opinion + col_more_money + col_not_important + col_weird

set(data.columns) == set(new_col_order)

data = data[new_col_order]
data_befsize = data.shape
data.head()


# In[26]:


# podsumowanie - ile cech w bazie zawiera puste komórki

summary = pd.DataFrame(data.dtypes, columns=['Data_Type'])
summary['Nulls'] = pd.DataFrame(data.isnull().any())
summary['Sum_of_Nulls'] = pd.DataFrame(data.isnull().sum())
summary['Per_of_Nulls'] = round((data.apply(pd.isnull).mean()*100),2)
summary.Data_Type = summary.Data_Type.astype(str)
print(summary)


# In[27]:


# tablica duplikatów - wiersze różnią się pojedynczymi komórkami
# trzeba dokładnie sprawdzić do ilu elementów nie powtarzających się uznaje grupe wierszy za duplikaty

data[data.duplicated(keep=False)]


# In[28]:


# szukanie kolumn z pustymi komórkami
data.isnull().sum()
data.isnull().sum().sum()

# sprawdzenie ile pustych komórek przypada na każdy wiersz
dd = data.isnull().sum(axis=1)
dd.sort_values(ascending=False)

# usunięcie wierszy z pustymi komórkami
# ich liczba jest tak mała, że nie zaszkodzi w dalszych badaniach
# usuwanie do 5%
data.dropna(axis=0, inplace=True)

# wyszukiwanie duplikatów tej samej informacji
# keep - czy zachować dane, np. first duplicate, last albo usunąć wszystkie
data[data.duplicated(keep=False)].shape

# 18 x 37 - można również usunąć
data.drop_duplicates(inplace=True)

# rozmiar końcowy do dalszych badań 
print(data.shape)

# różnica w rozmiarze
diff = data_befsize[0] - data.shape[0]
data_befsize[0] - data.shape[0]

# ile procent zostało usunięte
print(str(100 * diff/data_befsize[0]) + "% of whole dataset are duplicates and empty spaces!")


# In[29]:


data.columns


# In[30]:


# delete not needed columns
del_cols = ['EmployeeNumber', 'EmployeeCount', 'Application ID', 'Over18', 'StandardHours', 'StockOptionLevel']
data = data.drop(del_cols, axis=1)

# print all values for all columns ( to check if columns consists of only 1 value) 
# seems that NOO
for col in data.columns.tolist():
    data[col].value_counts()
    
# check types to be sure that everything seems right
data.dtypes


# In[31]:


# not right - object type to columns, which are supposed to be float
checktype_cols = ['JobSatisfaction', 'HourlyRate', 'PercentSalaryHike', 'MonthlyIncome', 'DistanceFromHome']

print(data.JobSatisfaction.size)
print(data.JobSatisfaction.value_counts())
print(data.JobSatisfaction.value_counts().sum())

# suma się zgadza. Dlaczego type object?
data.JobSatisfaction.unique()


# In[32]:


data[checktype_cols] = data[checktype_cols].apply(np.float64)
data.dtypes


# In[33]:


# to chyba nie jest rozkład normalny - lepiej zastosować inną korelację

data.describe()

# spróbować korelację rang Spearmana - może nasza baza również wykazuje skośność rozkładu


# In[34]:


# opis rozkładu danych typu object

data.select_dtypes(include = ['object']).describe()


# In[35]:


data.to_csv('clean_ibm_hr_data.csv',index=False)

# create copy for heatmap

data_copy = data.copy()
data_copy


# In[36]:


# now we are ready to start checking correlations between cols
# zamiana wszystkich object danych na numeryczne

object_data = data.select_dtypes(include=[np.object])
for col in object_data:
    print(col)
    print(data[col].unique())

gender = {'Female': 0, 'Male': 1}
marstatus = {'Single': 0, 'Married': 1, 'Divorced': 2}
edufield = {'Test': 0, 'Life Sciences': 1, 'Human Resources': 2, 'Marketing': 3, 'Technical Degree': 4, 'Medical': 5, 'Other': 6}
department = {'Sales': 0, 'Human Resources': 1, 'Research & Development': 2}
jobrole = {'Sales Executive': 0, 'Manager': 1, 'Human Resources': 2, 'Research Scientist': 3,
 'Manufacturing Director': 4, 'Laboratory Technician': 5,
 'Healthcare Representative': 6, 'Sales Representative': 7, 'Research Director': 8}
employeesrc = {'Referral': 1, 'Company Website': 2, 'Indeed': 3, 'Seek': 4, 'Adzuna': 5, 'Recruit.net': 6,
 'GlassDoor': 7, 'Jora': 8, 'LinkedIn': 9,'Test': 0}
busstravel = {'Travel_Rarely': 0, 'Travel_Frequently': 1, 'Non-Travel': 2}
attrition = {'Voluntary Resignation': 0, 'Current employee': 1, 'Termination': 0}
overtime = {'Yes': 1, 'No': 0}

# if its already changed, it won't work

data_copy['Gender']          = [gender[attr] for attr in data['Gender']]
data_copy['MaritalStatus']   = [marstatus[attr] for attr in data['MaritalStatus']]
data_copy['EducationField']  = [edufield[attr] for attr in data['EducationField']]
data_copy['Department']      = [department[attr] for attr in data['Department']]
data_copy['JobRole']         = [jobrole[attr] for attr in data['JobRole']]
data_copy['Employee Source'] = [employeesrc[attr] for attr in data['Employee Source']]
data_copy['BusinessTravel']  = [busstravel[attr] for attr in data['BusinessTravel']]
data_copy['Attrition']       = [attrition[attr] for attr in data['Attrition']]
data_copy['OverTime']        = [overtime[attr] for attr in data['OverTime']]


# In[37]:


# wyszukuje te dane, które są wartościami liczbowymi
numeric_data = data_copy.select_dtypes(include=[np.number])

# obliczenie korelacji między wszystkimi wartościami numerycznymi
# wnioski np. mapy korelacji poszczególnych atrybutów

# korelacja Pearsona
corr_matrix = data_copy[numeric_data.columns].corr(method="pearson")

fig, ax = plt.subplots(figsize = (15,15))

cmap = sns.diverging_palette(220, 180, as_cmap=True)
# Generate a custom diverging colormap

sns.heatmap(corr_matrix, center=0.0, vmax=1, square=True, linewidths=.5, ax=ax, cmap=cmap)


# In[38]:


# ANALIZA ZMIENNEJ CELU

print(data['Attrition'].value_counts())
print()
print(data['Attrition'].value_counts(normalize = True))


plt.figure(figsize=(10,7))
sns.countplot(data['Attrition'], palette = ['#eb6c6a', '#f0918f']).set(title = 'Wykres częstości Attrition', xlabel = 'Default', ylabel = 'Counts of observation')
plt.show()


# In[39]:


# korelacja dla Attrition

corr_matrix['Attrition']
# porównanie Attrition w departamentach

plt.figure(figsize=(50,15))
plt.title("Correlation between Attrition and all other variables")
plt.plot(corr_matrix['Attrition'].drop('Attrition'))
plt.axhline(0, color="k", linestyle="--");


# In[40]:


# do zaobserwowania: AGE, OVERTIME, TOTALWORKINGYEARS, DEPARTMENT, YEARSINCURRENTROLE, JOBLEVEL
# największa korelacja: AGE, OVERTIME
# zamiana danych na tych co pracują i co odeszli - uogólnienie

data.Attrition = data.Attrition.replace('Termination', 'Voluntary Resignation')
data.Attrition = data.Attrition.replace('Voluntary Resignation', 'Former Employees')
data.Attrition = data.Attrition.replace('Current employee', 'Current Employees')


# In[87]:


# Plot the distribution of Years at Company by Attrition


plt.figure(figsize=(15,10))
plt.title('Rozkład lat przepracowanych przez pracowników')
sns.distplot(data.YearsAtCompany[data.Attrition == 'Former Employees'], bins = np.linspace(0,40,40))
sns.distplot(data.YearsAtCompany[data.Attrition == 'Current Employees'], bins = np.linspace(0,40,40))
plt.legend(['Former Emploees','Current Employees'])


# In[46]:


# ANALIZA ZMIENNYCH POWIĄZANYCH Z CELEM

# AGE 

plt.figure(figsize=(13,13))
plt.title('Rozkład wieku')
sns.distplot(data.Age, bins = np.linspace(10,70,35))


# In[112]:


plt.figure(figsize=(13,10))
plt.title('Rozkład wieku byłych pracowników')
sns.distplot(data.Age[data.Attrition == 'Former Employees'], bins = np.linspace(10,70,35))
sns.distplot(data.Age[data.Attrition == 'Current Employees'], bins = np.linspace(10,70,35))
plt.legend(['Former Emploees','Current Employees'])


# In[48]:


def count_percentage(data, name1, name2):
    datasum = data[name1].value_counts()
    xdata = {}
    for index in datasum.index:
        valuessum = dict(data[name2][data[name1] == index].value_counts().divide(datasum[index]/100))
        xdata[index] = valuessum
    mdata = {}
    for index in xdata:
        for name in xdata[index].keys():
            if name not in mdata.keys():
                mdata[name] = {}
            mdata[name][index] = xdata[index][name]
    return mdata


# In[92]:


# porównanie Attrition w departamentach

# procentowe
# sales

print("Ile procent pracowników odchodzi z danego sektora firmy")
sales = data.Attrition[data.Department == 'Sales'].value_counts()
print("Sales: " + str(100 * sales['Former Employees'] / (sales.sum())))

# HR
hr = data.Attrition[data.Department == 'Human Resources'].value_counts()
print("HR: " + str(100 * hr['Former Employees'] / (hr.sum())))

# RnD
rnd = data.Attrition[data.Department == 'Research & Development'].value_counts()
print("RnD: " + str(100 * rnd['Former Employees'] / (rnd.sum())))

# na wykresie
pd.DataFrame(count_percentage(data, 'Department', 'Attrition')).plot(title='Attrition in departments',figsize=(15,8), kind='bar')


# In[93]:


# ilość szczebli w danym sektorze
# wykres

pd.DataFrame(count_percentage(data, 'Department', 'JobLevel')).plot(title='Percentage of people on specified job levels in departments',figsize=(15,8), kind='bar')


# In[114]:


plt.figure(figsize=(15,10))
plt.title('Rozkład płac byłych pracowników')
sns.distplot(data.MonthlyIncome[data.Department == 'Sales'], bins = np.linspace(10,25000,100))
sns.distplot(data.MonthlyIncome[data.Department == 'Human Resources'], bins = np.linspace(10,25000,100))
sns.distplot(data.MonthlyIncome[data.Department == 'Research & Development'], bins = np.linspace(10,25000,100))
plt.legend(['Sales','Human Resources', 'Research & Development'])


# In[117]:


plt.figure(figsize=(15,10))
plt.title('Rozkład płac byłych pracowników')
sns.distplot(data.DailyRate[data.Department == 'Sales'], bins = np.linspace(10,2000,100))
sns.distplot(data.DailyRate[data.Department == 'Human Resources'], bins = np.linspace(10,2000,100))
sns.distplot(data.DailyRate[data.Department == 'Research & Development'], bins = np.linspace(10,2000,100))
plt.legend(['Sales','Human Resources', 'Research & Development'])


# In[118]:


pd.DataFrame(count_percentage(data[data['Attrition'] == 'Former Employees'], 'Department', 'Gender')).plot(title='Gender who left per Department',figsize=(15,8), kind='bar')


# In[94]:


# attrition w zależnosci od nadgodzin


pd.DataFrame(count_percentage(data, 'OverTime', 'Attrition')).plot(title='Attrition by overtime work',figsize=(15,8), kind='bar')

# Given this assoication and that of age: perhaps over worked employees are more likely to be under 30?


# In[95]:


# check one more thing - Marital Status

pd.DataFrame(count_percentage(data, 'MaritalStatus', 'OverTime')).plot(title='Marital Status by overtime work',figsize=(15,8), kind='bar')


# In[56]:


####################################################################

# classification


data.Attrition.value_counts()


# In[57]:


data_copy.Attrition.value_counts()
data_copy.dtypes
data_copy.head()


# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn_pandas import DataFrameMapper, gen_features, cross_val_score


# In[59]:


# metoda 1
# random forest classifier

data_mod1 = data_copy.copy()
data_mod1 = data_mod1.sort_values('Attrition', ascending=False)


# In[60]:


from math import floor
testsize = floor(0.2*data_mod1.shape[0])
data_test = data_mod1[:testsize + 1].copy()
data_pract = data_mod1[testsize+1:].copy()


# In[61]:


rf = RandomForestClassifier(class_weight="balanced", n_estimators=500) 


# In[62]:


rf.fit(data_pract.drop('Attrition',axis=1), data_pract.Attrition)


# In[63]:


# część uczenia się

dats = rf.predict(data_pract.drop('Attrition',axis=1))
dats


# In[64]:


# sprawdzenie 

rf.score(data_pract.drop('Attrition',axis=1), data_pract.Attrition)


# In[65]:


# część testowa

rf.score(data_test.drop('Attrition', axis=1), data_test.Attrition)


# In[66]:


test = rf.predict(data_test.drop('Attrition', axis=1))


# In[67]:


importances = rf.feature_importances_


# In[68]:


names = data_pract.columns


# In[69]:


importances, names = zip(*sorted(zip(importances, names)))


# In[70]:


plt.figure(figsize=(15,15))
plt.barh(range(len(names)), importances, align = 'center')
plt.yticks(range(len(names)), names)
plt.xlabel('Importance of features')
plt.ylabel('Features')
plt.title('Importance of each feature')
plt.show()


# In[71]:


# Make predictions using 10-K-Fold-CV
# https://mateuszgrzyb.pl/klasyfikacja-wnioskow-o-wydanie-karty-kredytowej/
# o walidacji krzyżowej , dlaczego warto używać


# Baseline:
print((data_copy.Attrition.value_counts()/(data_copy.shape[0]))*100)


# Accuracy
# Accuracy jako główna miara jakości modelu i współczynnik zmienności jako miara jego stabilności.

scores = cross_val_score(rf, data_copy.drop(['Attrition'],axis=1), data_copy.Attrition, cv=10, scoring='accuracy')
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Średnie Accuracy: ' + str(cv.mean().round(3)))
print('Stabilność: ' + str((cv.std()*100/cv.mean()).round(3)) + '%')


# # ROC
# scores = cross_val_score(rf, data_copy.drop(['Attrition'],axis=1), data_copy.Attrition, cv=10, scoring='roc_auc')
# print(scores)
# print("ROC_AUC: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# print('Średnie Accuracy: ' + str(cv.mean().round(3)))
# print('Stabilność: ' + str((cv.std()*100/cv.mean()).round(3)) + '%')


# In[72]:


# metoda 2
# klasteryzacja

data_mod2 = data_copy[['Attrition', 'Age', 'JobLevel', 'OverTime', 'Department', 'TotalWorkingYears']].copy()
data_mod2.head()


# In[73]:


from sklearn.cluster import KMeans


# In[74]:


kmeansdata = data_mod2[:].copy()
kmeans = KMeans(n_clusters=2, random_state=0).fit(kmeansdata)
labels = kmeans.labels_


# In[75]:


kmeansdata['clusters'] = labels
kmeansdata


# In[76]:


print(kmeansdata[kmeansdata.columns].groupby(['clusters']).mean())


# In[99]:


plt.rcParams['figure.figsize']=(15,10)
sns.lmplot('Age', 'JobLevel', 
           data=kmeansdata, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", "s": 1})
plt.title('Clusters Age vs JobLevel')
plt.xlabel('Age')
plt.ylabel('JobLevel')
plt.show()


# In[78]:


# metoda 3
# klasteryzacja aglomeracyjna

from sklearn.cluster import AgglomerativeClustering


# In[103]:


data_mod3 = data_copy[['Attrition', 'Age', 'JobLevel', 'OverTime', 'Department', 'TotalWorkingYears']].copy()
agglomerative = AgglomerativeClustering(n_clusters=2, affinity='euclidean').fit(data_mod3)


# In[104]:


agglomerative


# In[105]:


labels = agglomerative.labels_


# In[106]:


data_mod3['clusters'] = labels
print(data_mod3[data_mod3.columns].groupby(['clusters']).mean())


# In[107]:


plt.rcParams['figure.figsize']=(10,10)
sns.lmplot('Age', 'JobLevel', 
           data=data_mod3, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", "s": 1})
plt.title('Clusters Age vs JobLevel')
plt.xlabel('Age')
plt.ylabel('JobLevel')
plt.show()

