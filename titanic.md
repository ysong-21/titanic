
Liying Lu

Titanic: Machine Learning from Disaster

Start: June 3, 2019


```python
# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib as mpl
import matplotlib.pylab as pylab
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score

# Modeling helpers 
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV
```

# Data Processing
1) Change all the character values to numbers  

2) Fill the NA values from the appropriate mean values

3) Combine columns to make a more informed column

4) Remove the unused columns as needed


```python
# import the dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# change the character values to numbers
# sex
def getGender(str):
    if str == 'male':
        return 1
    else:
        return 0
train['sex']=train['Sex'].apply(getGender)

# Embarked
def getEmbarked(str):
    if str == 'S':
        return 1
    elif str == 'C':
        return 2
    else:
        return 3
train['embarked'] = train['Embarked'].apply(getEmbarked)

# title
def getTitle(str):
    list = str.split()
    for i in list:
        if i[-1] == '.':
            return i
# separates the title from the name
train['title'] = train['Name'].apply(getTitle) 
# check the different titles
train['title'].unique()
# assign the numbers to the classified group of titles
# male:     Mr. Don.(Spanish Mr.) Sir.
# female:   married:[Mrs. Mme.(Madame)], unmarried:[Miss. Ms. Lady. Mlle.(French Miss.)]
# Military: Col. Capt. Major. 
# Paster:   Rev. 
# Academic: Dr. 
# Aritocrat:female:[Countess.] male:[Jonkheer. Master.]
def getTitleRank(str):
    if str=='Mr.' or str=='Don.' or str=='Sir.':
        return 1
    elif str=='Mrs.' or  str=='Mme.':
        return 2
    elif str=='Miss.' or str=='Ms.' or str=='Lady.' or str=='Mlle.':
        return 3
    elif str=='Col.' or str=='Capt.' or str=='Major.':
        return 4
    elif str=='Rev.':
        return 5
    elif str=='Dr.':
        return 6
    elif str=='Countess.' or str=='Jonkheer.' or str=='Master.':
        return 7
train['titleRank'] = train['title'].apply(getTitleRank)

train.head(10)
    
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>sex</th>
      <th>embarked</th>
      <th>title</th>
      <th>titleRank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>Mr.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>2</td>
      <td>Mrs.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>Miss.</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>Mrs.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>Mr.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
      <td>1</td>
      <td>3</td>
      <td>Mr.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>Mr.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>1</td>
      <td>Master.</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>1</td>
      <td>Mrs.</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
      <td>0</td>
      <td>2</td>
      <td>Mrs.</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
# fill the NA values
# remove unused columns and check where the null values are
train.isnull().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    sex              0
    embarked         0
    title            0
    titleRank        0
    dtype: int64




```python
# fill the missing ages according to the Pcalss_titleRank
meanAges = np.zeros(shape=(3,7)) # a matrix to hold the ages
for p in range(1,4): # pclass
    for tr in range(1,8): # titleRank
        meanAges[p-1][tr-1] = round(train[(train.Pclass==p) & (train.titleRank==tr)].Age.mean(),1)
for p in range(1,4):
    for tr in range(1,8):
        train['Age'] = np.where((train.Age.isnull()) & (train['Pclass']==p) & (train['titleRank']==tr), meanAges[p-1][tr-1], train['Age'])

```


```python
# add a family column that totals the number of Sibsp and Parch.
train['family'] = train['SibSp'] + train['Parch'] +1
# add a isalone column that determines if the passenger is alone
train['isalone'] = np.where(train['family']==1, 1, 0) 
```


```python
# remove the unused columns SibSp and Parch
train = train.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked','title','SibSp','Parch'], axis=1)
train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>sex</th>
      <th>embarked</th>
      <th>titleRank</th>
      <th>family</th>
      <th>isalone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>7.2500</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>71.2833</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Machine Learning 

The machine learning portion includes the following steps:

1) Seperate the train dataset into X_all and Y_all by 7:3 for training and testing respectively.

2) Use RandomForestClassifier as the testing model.

3) Use KFold to test the accuracy of the results.


```python
# Seperate the train dataset intp training and testing dataset
X_all = train.drop(['Survived'], axis=1)
Y_all = train['Survived']

train_valid_X = X_all
train_valid_Y = Y_all
train_X, valid_X, train_Y, valid_Y = train_test_split(train_valid_X, train_valid_Y, train_size=0.7, test_size=0.3)

X_all.shape, train_X.shape, valid_X.shape, Y_all.shape, train_Y.shape, valid_Y.shape
```




    ((891, 8), (623, 8), (268, 8), (891,), (623,), (268,))




```python
# load the model RandomForestClassifier
model = RandomForestClassifier(max_leaf_nodes=100)
model.fit(train_X, train_Y)

# load KFold to test the accuracy of the model
def run_kFold(RFC):
    kf = KFold(n_splits = 10, random_state=None, shuffle=False)
    outcome = []
    fold = 0
    for train_index, test_index in kf.split(train):
        fold += 1
        train_X, valid_X = X_all.values[train_index], X_all.values[test_index]
        train_Y, valid_Y = Y_all.values[train_index], Y_all.values[test_index]
        RFC.fit(train_X, train_Y)
        prediction = RFC.predict(valid_X)
        accuracy = accuracy_score(valid_Y, prediction)
        outcome.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))
    mean_outcome = np.mean(outcome)
    print("Mean Accuracy: {0}".format(mean_outcome))

run_kFold(model)
```

    Fold 1 accuracy: 0.7777777777777778
    Fold 2 accuracy: 0.8764044943820225
    Fold 3 accuracy: 0.7752808988764045
    Fold 4 accuracy: 0.8314606741573034
    Fold 5 accuracy: 0.8539325842696629
    Fold 6 accuracy: 0.8314606741573034
    Fold 7 accuracy: 0.7865168539325843
    Fold 8 accuracy: 0.7752808988764045
    Fold 9 accuracy: 0.8539325842696629
    Fold 10 accuracy: 0.8651685393258427
    Mean Accuracy: 0.8227215980024969
    

    C:\ProgramData\Anaconda3\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    

# Actual prediction on the test dataset

1) Clean the test dataset 
    *  change the character values to numerical values

    *  fill the null values

    *  create new columns

    *  remove unused columns

2) Apply RandomForestClassifier

3) save the results into a csv file


```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# change the character values to numerical values
test['sex']=test['Sex'].apply(getGender) # sex
test['embarked'] = test['Embarked'].apply(getEmbarked) # embarked
test['title'] = test['Name'].apply(getTitle) # separate title from names
test['titleRank'] = test['title'].apply(getTitleRank) # title rank
# age
for p in range(1,4):
    for tr in range(1,8):
        test['Age'] = np.where((test.Age.isnull()) & (test['Pclass']==p) & (test['titleRank']==tr), meanAges[p-1][tr-1], test['Age'])

# add new columns
test['family'] = test['SibSp'] + test['Parch'] +1 # family
test['isalone'] = np.where(test['family']==1, 1, 0) # isalone
        
# remove the unused columns
test = test.drop(['Name','Sex','Ticket','Cabin','Embarked','title','SibSp','Parch'], axis=1)

test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>Fare</th>
      <th>sex</th>
      <th>embarked</th>
      <th>titleRank</th>
      <th>family</th>
      <th>isalone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>34.5</td>
      <td>7.8292</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>47.0</td>
      <td>7.0000</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>62.0</td>
      <td>9.6875</td>
      <td>1</td>
      <td>3</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>27.0</td>
      <td>8.6625</td>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>22.0</td>
      <td>12.2875</td>
      <td>0</td>
      <td>1</td>
      <td>2.0</td>
      <td>3</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# check for null values in test
print(test.isnull().sum())
```

    PassengerId    0
    Pclass         0
    Age            0
    Fare           1
    sex            0
    embarked       0
    titleRank      1
    family         0
    isalone        0
    dtype: int64
    


```python
# null value in titleRank
print(test[test['titleRank'].isnull()])
# give the title of Aristocrat because of the high fare and no. 1 pclass 
test.loc[test.titleRank.isnull(), "titleRank"] = 7
```

         PassengerId  Pclass   Age   Fare  sex  embarked  titleRank  family  \
    414         1306       1  39.0  108.9    0         2        NaN       1   
    
         isalone  
    414        1  
    


```python
# null value in Fare
print(test[test.Fare.isnull()])
# take the average fare of the no. 3 Pclass
avgFare = train[train.Pclass==3].Fare.mean()
test.loc[test.Fare.isnull(), "Fare"] = avgFare
```

         PassengerId  Pclass   Age  Fare  sex  embarked  titleRank  family  \
    152         1044       3  60.5   NaN    1         1        1.0       1   
    
         isalone  
    152        1  
    


```python
# last check on the null values in test
test.isnull().sum()
```




    PassengerId    0
    Pclass         0
    Age            0
    Fare           0
    sex            0
    embarked       0
    titleRank      0
    family         0
    isalone        0
    dtype: int64




```python
# apply RFC
testPrediction = model.predict(test.drop(['PassengerId'], axis=1))
outcome = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': testPrediction})
outcome.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# save the outcome into a csv
outcome.to_csv('titanic_predictions1.csv', index=False)
```
