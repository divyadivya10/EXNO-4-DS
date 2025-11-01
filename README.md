# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:


```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```

<img width="1390" height="380" alt="image" src="https://github.com/user-attachments/assets/115a4197-167e-4989-9a59-346b86c84bac" />


```
 data.isnull().sum()
```
<img width="230" height="310" alt="image" src="https://github.com/user-attachments/assets/296a08de-cebe-47e4-9d69-4d412b1d8fda" />

```
 missing=data[data.isnull().any(axis=1)]
 missing
```
<img width="1382" height="378" alt="image" src="https://github.com/user-attachments/assets/6434428c-f231-4655-ac9c-6f6583b25fa1" />

```
 data2=data.dropna(axis=0)
 data2
````
<img width="1369" height="381" alt="image" src="https://github.com/user-attachments/assets/f3454ad3-5baf-4aaf-ac6d-b8777904d160" />

```
 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```
<img width="440" height="256" alt="image" src="https://github.com/user-attachments/assets/92a08f4f-907f-4688-b6ce-4c3a03385e4b" />

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
<img width="412" height="517" alt="image" src="https://github.com/user-attachments/assets/d21527d3-2919-4ee3-83a3-6ae48ff15218" />


```
 data2
```
<img width="1374" height="369" alt="image" src="https://github.com/user-attachments/assets/f6ff31d9-3ebf-4c31-991e-1c406a4048bd" />


```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```

<img width="1402" height="331" alt="image" src="https://github.com/user-attachments/assets/79be7e22-c99a-440b-ae62-82ded1845fcf" />

```
 columns_list=list(new_data.columns)
 print(columns_list)
```
<img width="1377" height="218" alt="image" src="https://github.com/user-attachments/assets/67ee332d-5c4f-4515-a757-dd176b0f6b28" />

```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```

<img width="1397" height="157" alt="image" src="https://github.com/user-attachments/assets/3849cbb3-eda1-42ca-b5e9-98dfa6de4f48" />

```
 y=new_data['SalStat'].values
 print(y)
```
<img width="174" height="32" alt="image" src="https://github.com/user-attachments/assets/bd1287c0-14c1-4800-b037-884284e048fd" />

```
 x=new_data[features].values
 print(x)
```
<img width="445" height="171" alt="image" src="https://github.com/user-attachments/assets/5c9f850b-201f-4990-9c38-38dc1b6d8e53" />

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
<img width="306" height="105" alt="image" src="https://github.com/user-attachments/assets/54809507-bd88-4d28-a5c5-b82dc601497c" />


```
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```
<img width="150" height="52" alt="image" src="https://github.com/user-attachments/assets/7db496ae-c80f-48fe-8463-8c51c9e49863" />

```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```
<img width="199" height="36" alt="image" src="https://github.com/user-attachments/assets/34fbdfcb-031e-4a76-a4e8-ab7f9b9b9884" />

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
<img width="295" height="38" alt="image" src="https://github.com/user-attachments/assets/5035bcbb-5802-48a8-860b-247bc66d0bf2" />

```
 data.shape
```

<img width="144" height="44" alt="image" src="https://github.com/user-attachments/assets/268cfb25-a2e9-4114-b20a-da7e30acd456" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)
```

<img width="343" height="56" alt="image" src="https://github.com/user-attachments/assets/147e025d-dd53-40ba-acb4-bb60f6831d22" />


```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```
<img width="484" height="237" alt="image" src="https://github.com/user-attachments/assets/bd6b7b46-824d-4e22-b8e6-7c5c511bd26a" />


```
 tips.time.unique()
```
<img width="428" height="62" alt="image" src="https://github.com/user-attachments/assets/f3037ba9-5422-423d-9b24-44ece350ab57" />


```
 contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```
<img width="234" height="102" alt="image" src="https://github.com/user-attachments/assets/146f5006-1882-4c88-9d4b-fc0705e84fa4" />

```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```
<img width="377" height="60" alt="image" src="https://github.com/user-attachments/assets/a0d7f000-e391-4bf1-94e3-49cdea64c63f" />








# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
save the data to a file is been executed
