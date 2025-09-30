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

# CODING
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

# output
<img width="1347" height="701" alt="image" src="https://github.com/user-attachments/assets/e32fcec5-a2fa-4030-8b13-bc3047be9a92" />


```
data.isnull().sum()

```
# output
<img width="296" height="651" alt="image" src="https://github.com/user-attachments/assets/6d27febe-b0f6-4c94-93dd-38744019bc62" />


```
missing=data[data.isnull().any(axis=1)]
missing
```


# output
<img width="1352" height="762" alt="image" src="https://github.com/user-attachments/assets/cddf8773-0a81-490c-b198-3e01ecc5089f" />



```
 data2=data.dropna(axis=0)
 data2
```


# output

<img width="1374" height="773" alt="Screenshot 2025-09-30 111018" src="https://github.com/user-attachments/assets/54a61ee1-d05f-4851-9b57-8ffda1c3df8e" />



```
 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])
```


# output

<img width="1351" height="416" alt="image" src="https://github.com/user-attachments/assets/cc461001-b698-45fe-9f76-cf6159ddd0f1" />


```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```


# output

<img width="442" height="512" alt="image" src="https://github.com/user-attachments/assets/dbb923d0-443f-446c-a0ba-900f5c0c0c9c" />


```
 new_data=pd.get_dummies(data2, drop_first=True)
 new_data
```


# output

<img width="1407" height="676" alt="image" src="https://github.com/user-attachments/assets/eb481f20-752f-4192-9bc2-a77902a9d160" />


```
columns_list=list(new_data.columns)
print(columns_list)
```



# output

<img width="1380" height="41" alt="image" src="https://github.com/user-attachments/assets/e35fdf0d-9a88-407d-822a-319cefb25756" />


```
 features=list(set(columns_list)-set(['SalStat']))
 print(features)
```


# output

<img width="1362" height="37" alt="image" src="https://github.com/user-attachments/assets/2616d739-2d2e-4fe9-a8b2-5d6139f9f667" />


```
x=new_data[features].values
print(x)
```


# output

<img width="466" height="171" alt="image" src="https://github.com/user-attachments/assets/31064398-8458-4976-9d07-bc2a5c2e5639" />


```
 KNN_classifier.fit(train_x,train_y)
```


# output
<img width="358" height="86" alt="image" src="https://github.com/user-attachments/assets/9ee5bbaa-ed55-4714-9a81-6122f0ee4c13" />



```
 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)
```



# output

<img width="195" height="65" alt="image" src="https://github.com/user-attachments/assets/c5d72dac-eff3-4949-a218-4c6d0e792c37" />



```
 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)
```



# output

<img width="214" height="31" alt="Screenshot 2025-09-30 113253" src="https://github.com/user-attachments/assets/c7a843f8-4f77-4961-b638-455741ee869f" />



```
 print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```


# output

<img width="296" height="32" alt="image" src="https://github.com/user-attachments/assets/49c9c00e-aa8b-4cdc-93d6-be3717b1ebf6" />



```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data
```



# output

<img width="1318" height="660" alt="image" src="https://github.com/user-attachments/assets/387ebbf9-ed33-4f5a-8a0b-c94ab3910a39" />



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



# output

<img width="1372" height="112" alt="image" src="https://github.com/user-attachments/assets/41165eb2-697c-489a-ae0a-8099283dd8fd" />



```
 import pandas as pd
 import numpy as np
 from scipy.stats import chi2_contingency
 import seaborn as sns
 tips=sns.load_dataset('tips')
 tips.head()
```



# output

<img width="540" height="257" alt="image" src="https://github.com/user-attachments/assets/fbae70e3-7889-400f-bb50-dccd28299ecc" />



```
tips.time.unique()
```
# output

<img width="458" height="62" alt="image" src="https://github.com/user-attachments/assets/46ffa7a5-5f46-461f-a466-f7b9240cf2ee" />   


  
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
 print(contingency_table)
```



# output

<img width="242" height="96" alt="image" src="https://github.com/user-attachments/assets/cff50140-a932-4c90-9c32-eb61d9ad9de9" />




```
 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")
```


# output

<img width="402" height="47" alt="image" src="https://github.com/user-attachments/assets/20db1507-5301-4130-a035-c854de4efab3" />


# RESULT:

 Thus the program to read the given data and perform Feature Scaling and Feature Selection process and
 save the data to a file is been executed.
