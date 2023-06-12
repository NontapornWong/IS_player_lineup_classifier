# Independent Study
###  ⚽ Enhancing Player Selection Strategies: Machine Learning for Manchester United's Lineup Prediction
In this project, I conducted experiments utilizing a combination of traditional machine learning classifiers and neural network models implemented using TensorFlow. The primary objective was to predict the optimal player lineup for Manchester United based on historical statistics. To obtain the necessary data, I performed web scraping from "https://fbref.com/", extracting relevant information such as player performance metrics and match data.

Through extensive experimentation, I tested multiple traditional machine learning models. However, after careful evaluation, three models demonstrated the most potential in effectively working with this dataset. These models included Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). Additionally, I implemented neural network models using TensorFlow to leverage the power of deep learning for improved predictions.

#### Import Data, Cleansing, EDA
In the initial phase of this project, I performed web scraping on the [2022-2023 Manchester United Stats (Premier League)](https://fbref.com/en/players/b853e0ad/matchlogs/2022-2023/Fred-Match-Logs) to collect relevant data. I focused on extracting statistics specifically for the midfield position. To ensure data quality, I conducted cleansing procedures by removing unnecessary columns and handling missing values. To address missing data, data interpolation was employed.

Data interpolation is essential in this context as it allows for the estimation or filling in of missing values, ensuring a complete dataset for analysis. By utilizing interpolation techniques, we can infer and estimate values based on existing data patterns, thereby preserving the integrity of the dataset. This step enhances the reliability and comprehensiveness of subsequent analyses and predictions.

After performing data interpolation, I proceeded with exploratory data analysis (EDA) to gain insights into the dataset. This involved checking correlations between different variables and examining their distributions.

By conducting correlation analysis, I aimed to identify any significant relationships between variables such as goals, assists, tackles, and passing statistics. This helped in understanding the interplay between different performance metrics and their potential impact on player selection.

Additionally, analyzing the distribution of variables provided valuable information about their underlying patterns and variations. This enabled me to identify outliers, understand the central tendencies of the data, and assess the overall consistency of the dataset.

Overall, the EDA stage allowed me to uncover meaningful insights and patterns within the dataset, paving the way for informed decision-making in the subsequent phases of the project

# Manchester United Midfielder 2022-2023 Match Logs (Summary)

The dataset contains match logs for the midfielders of Manchester United for the 2022-2023 Premier League season. The provided data includes various performance metrics, expected goals (xG), shot-creating actions (SCA), goal-creating actions (GCA), passing statistics, carrying statistics, and take-ons.

## Performance
- Goals (Gls): Number of goals scored
- Assists (Ast): Number of assists
- Penalty kicks made (PK): Number of penalty kicks converted
- Penalty kicks attempted (PKatt): Number of penalty kicks taken
- Shot totals (Sh): Total number of shots
- Shots on target (SoT): Number of shots on target
- Yellow cards (CrdY): Number of yellow cards received
- Red cards (CrdR): Number of red cards received
- Touches: Number of times the player touched the ball
- Tackles (Tkl): Number of successful tackles
- Interceptions (Int): Number of interceptions
- Blocks: Number of times the player blocked the ball

## Expected Goals (xG)
- Expected Goals (xG): Expected goals, including penalty kicks
- Non-Penalty xG (npxG): Expected goals excluding penalty kicks
- Expected Assisted Goals (xAG): Expected goals following a pass that assists a shot

## Shot-Creating Actions (SCA) and Goal-Creating Actions (GCA)
- Shot-Creating Actions (SCA): Offensive actions directly leading to a shot
- Goal-Creating Actions (GCA): Offensive actions directly leading to a goal

## Passes
- Passes Completed (Cmp): Number of passes completed
- Passes Attempted (Att): Number of passes attempted
- Pass Completion Percentage (Cmp%): Percentage of passes completed
- Progressive Passes (PrgP): Passes that move the ball towards the opponent's goal line

## Carries
- Carries: Number of times the player controlled the ball with their feet
- Progressive Carries (PrgC): Carries that move the ball towards the opponent's goal line

## Take-Ons
- Take-Ons Attempted (Att.1): Number of attempted take-ons
- Successful Take-Ons (Succ): Number of successful take-ons

# Machine Learning Implementation
### Required Libraries

The following Python libraries are required to run the code:

- scikit-learn (version X.X.X)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, classification_report
```
### Train, Validation, Test Dataset
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
from imblearn.over_sampling import RandomOverSampler
train, valid, test = np.split(df_train.sample(frac=1), [int(0.6*len(df_train)), int(0.8*len(df_train))])

def scale(df, oversample=False):
  X = df[df.columns[:-1]].values
  y = df[df.columns[-1]].values
  
  mms = MinMaxScaler()
  X = mms.fit_transform(X)

  if oversample:
    ros = RandomOverSampler()
    X, y = ros.fit_resample(X, y)

  data = np.hstack((X, np.reshape(y, (-1,1))))

  return data, X, y
  
train, X_train, y_train = scale(train, oversample=True)
valid, X_val, y_val = scale(valid, oversample=False)
test, X_test, y_test = scale(test, oversample=False)
```
## Traditional Machine learning 
### Plot learning curve 
```python
def plot_learning_curve(estimator, X, y, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X, y, train_sizes=train_sizes, cv=cv, scoring='accuracy'
    )
    
    # Calculate mean and standard deviation of training and validation scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)
    
    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Score', color='blue')
    plt.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
        alpha=0.1, color='blue'
    )
    plt.plot(train_sizes, validation_scores_mean, label='Validation Score', color='orange')
    plt.fill_between(
        train_sizes, validation_scores_mean - validation_scores_std,
        validation_scores_mean + validation_scores_std, alpha=0.1, color='orange'
    )
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
```
```python
accuracy_train=[]
accuracy_val=[]
accuracy_cross=[]

def data_model(model, k=5):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print("Training Accuracy:", train_accuracy)
    print("Validation Accuracy:", val_accuracy)
    accuracy_train.append(train_accuracy)
    accuracy_val.append(val_accuracy)

    scores = cross_val_score(model, X_train, y_train, cv=k)
    print("Cross-Validation Scores:", scores)
    print("Average Cross-Validation Score:", np.mean(scores))
    accuracy_cross.append(np.mean(scores))

    train_sizes = np.linspace(0.1, 1.0, 10)  
    plot_learning_curve(model, X_train, y_train, train_sizes, cv=k)
```
```python
Algorithm=['LogisticRegression','GradientBoostingClassifier','KNeighborsClassifier','RandomForestClassifier',
           'DecisionTreeClassifie',
          'GaussianNB','SVC']
```
#### Logistic regression
```python
Logistic_model=LogisticRegression()
data_model(Logistic_model)
```
#### Random Forest Classifier
```python
RFC_model=RandomForestClassifier()
data_model(RFC_model)
```
#### Gradient Boosting Classifier
```python
GBC_model=GradientBoostingClassifier()
data_model(GBC_model)
```
#### Decision Tree Classifier
```python
DTC_model=DecisionTreeClassifier()
data_model(DTC_model)
```
#### KNeighbors Classifier
```python
KNC_model=KNeighborsClassifier()
data_model(KNC_model)
```
#### GaussianNB
```python
GNB_model=GaussianNB()
data_model(GNB_model)
```
#### SVC
```python
SVC_model=SVC()
data_model(SVC_model)
```
### Accuracy results from 7 models 
The table below shows the training accuracy, validation accuracy, and average cross-validation score for different algorithms:

| Algorithm                   | Training Accuracy | Validation Accuracy | Average Cross-Validation Score |
|-----------------------------|-------------------|---------------------|-------------------------------|
| LogisticRegression          | 0.812500          | 0.843137            | 0.802323                      |
| GradientBoostingClassifier  | 1.000000          | 0.803922            | 0.932636                      |
| KNeighborsClassifier        | 1.000000          | 0.745098            | 0.922997                      |
| RandomForestClassifier      | 1.000000          | 0.803922            | 0.846574                      |
| DecisionTreeClassifier      | 0.836538          | 0.764706            | 0.759466                      |
| GaussianNB                  | 0.615385          | 0.607843            | 0.615215                      |
| SVC                         | 0.855769          | 0.843137            | 0.787921                      |
