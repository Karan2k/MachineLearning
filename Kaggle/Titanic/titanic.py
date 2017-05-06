# remove warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn import tree
from sklearn import ensemble
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# Set the global default size of matplotlib figures
plt.rc('figure', figsize=(10, 5))

# Size of matplotlib figures that contain subplots
FIGSIZE_WITH_SUBPLOTS = (7, 7)

# Size of matplotlib histogram bins
BIN_SIZE = 10

# VARIABLE DESCRIPTIONS:
# survival        Survival
#                 (0 = No; 1 = Yes)
# pclass          Passenger Class
#                 (1 = 1st; 2 = 2nd; 3 = 3rd)
# name            Name
# sex             Sex
# age             Age
# sibsp           Number of Siblings/Spouses Aboard
# parch           Number of Parents/Children Aboard
# ticket          Ticket Number
# fare            Passenger Fare
# cabin           Cabin
# embarked        Port of Embarkation
#                 (C = Cherbourg; Q = Queenstown; S = Southampton)

# SPECIAL NOTES:
# Pclass is a proxy for socio-economic status (SES)
#  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

# Age is in Years; Fractional if Age less than One (1)
#  If the Age is Estimated, it is in the form xx.5

# With respect to the family relation variables (i.e. sibsp and parch)
# some relations were ignored.  The following are the definitions used
# for sibsp and parch.

# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
# Parent:   Mother or Father of Passenger Aboard Titanic
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

# Other family relatives excluded from this study include cousins,
# nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
# only with a nanny, therefore parch=0 for them.  As well, some
# travelled with very close friends or neighbors in a village, however,
# the definitions do not support such relations.

# load the dataset using pandas
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print the first few rows to get an idea of the dataset
print train.head()
print test.head()

# print the data types of the training dataset
print train.dtypes
print test.dtypes

# print the basic information of the dataset
print train.info()
print test.info()

# generate various descriptive statistics of the dataset
print train.describe()
print test.describe()

# setup a grid of plots
figure = plt.figure(figsize = FIGSIZE_WITH_SUBPLOTS)
fig_dims = (3, 2)

# Plot death and survival counts
plt.subplot2grid(fig_dims, (0, 0))
train["Survived"].value_counts().plot(kind='bar', title = 'Death and Survival Counts')

# Plot Pclass counts
plt.subplot2grid(fig_dims, (0, 1))
train['Pclass'].value_counts().plot(kind='bar', title='Passenger Class Counts')

# Plot Sex counts
plt.subplot2grid(fig_dims, (1, 0))
train['Sex'].value_counts().plot(kind='bar', title='Gender Counts')
plt.xticks(rotation=0)

# Plot Embarked counts
plt.subplot2grid(fig_dims, (1, 1))
train['Embarked'].value_counts().plot(kind='bar', title='Ports of Embarkation Counts')

# Plot the Age histogram
plt.subplot2grid(fig_dims, (2, 0))
train['Age'].hist()
plt.title('Age Histogram')

plt.show()

pclass_xt = pd.crosstab(train['Pclass'], train['Survived'])
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)
pclass_xt_pct.plot(kind = 'bar', stacked = True, title = 'Survival Rate by Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()

# mapping sex from string format to integer format
sexes = sorted(train['Sex'].unique())
genders_mapping = dict(zip(sexes, range(0, len(sexes) + 1)))
train['Sex_Val'] = train['Sex'].map(genders_mapping).astype(int)

sex_val_xt = pd.crosstab(train['Sex_Val'], train['Survived'])
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
sex_val_xt_pct.plot(kind='bar', stacked=True, title='Survival Rate by Gender')

plt.show()

# Plot survival rate by Sex
females_df = train[train['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis=0)
females_xt_pct.plot(kind='bar', 
                    stacked=True, 
                    title='Female Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()

# Plot survival rate by Pclass
males_df = train[train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
males_xt_pct.plot(kind='bar', 
                  stacked=True, 
                  title='Male Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')

plt.show()

# Get the unique values of Embarked
train['Embarked'] = train['Embarked'].fillna('S')
embarked_locs = sorted(train['Embarked'].unique())
embarked_locs_mapping = dict(zip(embarked_locs, range(0, len(embarked_locs) + 1)))
train['Embarked_Val'] = train['Embarked'] \
                               .map(embarked_locs_mapping) \
                               .astype(int)
train['Embarked_Val'].hist(bins=len(embarked_locs), range=(0, 3))
plt.title('Port of Embarkation Histogram')
plt.xlabel('Port of Embarkation')
plt.ylabel('Count')
plt.show()

embarked_val_xt = pd.crosstab(train['Embarked_Val'], train['Survived'])
embarked_val_xt_pct = \
    embarked_val_xt.div(embarked_val_xt.sum(1).astype(float), axis=0)
embarked_val_xt_pct.plot(kind='bar', stacked=True)
plt.title('Survival Rate by Port of Embarkation')
plt.xlabel('Port of Embarkation')
plt.ylabel('Survival Rate')

plt.show()

# Impute the missing Age values
train['Age'] = train['Age'].fillna(train['Age'].median())

# Adding a new feature for Child which is 1 if Age is <= 10 else 0
train['Child'] = float('NaN')
train['Child'][train['Age'] <= 10] = 1
train['Child'][train['Age'] > 10] = 0
 
# Adding a new feature for family size
train['FamilySize'] = train['SibSp'] + train['Parch']

# Cleaning the testing data and mapping the feature values to required type
test['Age'] = test['Age'].fillna(test['Age'].median())

# Adding Child feature to test dataset
test['Child'] = float('NaN')
test['Child'][test['Age'] <= 10] = 1
test['Child'][test['Age'] > 10] = 0

# Mapping Sex values from string to int
test['Sex_Val'] = test['Sex'].map(genders_mapping).astype(int)

# Mapping Embarkation location from string to int
test['Embarked_Val'] = test['Embarked'].map(embarked_locs_mapping).astype(int)

# Imputing the missing Fare value in test dataset
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Adding new feature for family size
test['FamilySize'] = test['SibSp'] + test['Parch']

# Extracting the required features from the given dataset
train_features = train[['Fare', 'Child', 'Sex_Val', 'FamilySize', 'Pclass', 'Age', 'SibSp', 'Parch']].values
test_features = test[['Fare', 'Child', 'Sex_Val', 'FamilySize', 'Pclass', 'Age', 'SibSp', 'Parch']].values

target = train['Survived'].values

# Fit Decision tree with given training features and target value
model = tree.DecisionTreeClassifier(max_depth = 9, min_samples_split = 5, random_state = 1)
model.fit(train_features, target)

forest = ensemble.RandomForestClassifier(max_depth = 7, min_samples_split = 15, n_estimators = 80)
forest.fit(train_features, target)

# Make predictions using the test dataset
prediction = forest.predict(test_features)

# Create a data frame with two columns : PassengerId and Survived(prediction)
PassengerId = np.array(test["PassengerId"]).astype(int)
solution = pd.DataFrame(prediction, index = PassengerId, columns = ["Survived"])

# Write solution to a csv file
solution.to_csv("output.csv", index_label = ["PassengerId"])

