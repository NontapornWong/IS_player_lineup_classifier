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

## 🚩Performance
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

## 🚩Expected Goals (xG)
- Expected Goals (xG): Expected goals, including penalty kicks
- Non-Penalty xG (npxG): Expected goals excluding penalty kicks
- Expected Assisted Goals (xAG): Expected goals following a pass that assists a shot

## 🚩Shot-Creating Actions (SCA) and Goal-Creating Actions (GCA)
- Shot-Creating Actions (SCA): Offensive actions directly leading to a shot
- Goal-Creating Actions (GCA): Offensive actions directly leading to a goal

## 🚩Passes
- Passes Completed (Cmp): Number of passes completed
- Passes Attempted (Att): Number of passes attempted
- Pass Completion Percentage (Cmp%): Percentage of passes completed
- Progressive Passes (PrgP): Passes that move the ball towards the opponent's goal line

## 🚩Carries
- Carries: Number of times the player controlled the ball with their feet
- Progressive Carries (PrgC): Carries that move the ball towards the opponent's goal line

## 🚩Take-Ons
- Take-Ons Attempted (Att.1): Number of attempted take-ons
- Successful Take-Ons (Succ): Number of successful take-ons

# 📍 Machine Learning Implementation
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
## 🗼Traditional Machine learning🗼
#### Plot learning curve 
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
![lm](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/f76c34fa-b206-4f7f-bbf8-1a79b8cc8e43)

#### Random Forest Classifier
```python
RFC_model=RandomForestClassifier()
data_model(RFC_model)
```
![randomforest](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/3fd9dea8-21b9-43de-88bd-29018f7e863b)

#### Gradient Boosting Classifier
```python
GBC_model=GradientBoostingClassifier()
data_model(GBC_model)
```
![GBC](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/ebd276c2-39fb-4824-b1e2-c5bafc3d073f)

#### Decision Tree Classifier
```python
DTC_model=DecisionTreeClassifier()
data_model(DTC_model)
```
![DTC](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/fd9942e8-725f-45d9-a7f7-4296cd9fe23d)

#### KNeighbors Classifier
```python
KNC_model=KNeighborsClassifier()
data_model(KNC_model)
```
![KNN](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/3aaebaca-0080-465a-be54-9ab43d7d1ca9)

#### GaussianNB
```python
GNB_model=GaussianNB()
data_model(GNB_model)
```
![GNB](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/416fae17-d242-4911-82d3-979cb2affff0)

#### SVC
```python
SVC_model=SVC()
data_model(SVC_model)
```
![SVC](https://github.com/NontapornWong/IS_player_lineup_classifier/assets/97610480/82eeb064-9831-4073-a310-4463e79adb8b)

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

### Fine Tuning
```python
#Logistic regression
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.01, 0.1, 1, 10, 100]
}
logistic_model = LogisticRegression()
grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_logistic_model = LogisticRegression(**best_params)
best_logistic_model.fit(X_train, y_train)
```
```python
log_predict = best_logistic_model.predict(X_test)
print(classification_report(y_test, log_predict))
```
##### Logistic regression report 

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   0.92    |  0.63  |   0.75   |   19    |
|     1     |   0.82    |  0.97  |   0.89   |   32    |
|  Accuracy |           |        |   0.84   |   51    |
| Macro Avg |   0.87    |  0.80  |   0.82   |   51    |
| Weighted Avg |  0.86   |  0.84  |   0.84   |   51    |

```python
#KNN
param_grid = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

knn_model = KNeighborsClassifier()
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_knn_model = KNeighborsClassifier(**best_params)
best_knn_model.fit(X_train, y_train)
```
```python
KNN_predict = best_knn_model.predict(X_test)
print(classification_report(y_test, KNN_predict))
```
##### KNN report 

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   0.65    |  0.58  |   0.61   |   19    |
|     1     |   0.76    |  0.81  |   0.79   |   32    |
|  Accuracy |           |        |   0.73   |   51    |
| Macro Avg |   0.71    |  0.70  |   0.70   |   51    |
| Weighted Avg |  0.72   |  0.73  |   0.72   |   51    |

```python
#SVM
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
SVC_model=SVC()
grid_search = GridSearchCV(SVC_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_svc_model = SVC(**best_params)
best_svc_model.fit(X_train, y_train)
```
```python
SVC_predict = best_svc_model.predict(X_test)
print(classification_report(y_test, SVC_predict))
```
##### SVC report 

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   0.80    |  0.42  |   0.55   |   19    |
|     1     |   0.73    |  0.94  |   0.82   |   32    |
|  Accuracy |           |        |   0.75   |   51    |
| Macro Avg |   0.77    |  0.68  |   0.69   |   51    |
| Weighted Avg |  0.76   |  0.75  |   0.72   |   51    |

## 🗼Implement Neural Network with Tensorflow🗼
```python
import tensorflow as tf
```
```python
def plot_hist(history):
  fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
  ax1.plot(history.history['loss'], label='loss')
  ax1.plot(history.history['val_loss'], label='val_loss')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Binary crossentropy')
  ax1.legend()
  ax1.grid(True)

  ax2.plot(history.history['accuracy'], label='accuracy')
  ax2.plot(history.history['val_accuracy'], label='val_accuracy')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.legend()
  ax2.grid(True)

  plt.show()
```
```python
def train_model(X_train, y_train, num_nodes, dropout, learning_rate, batch_size, epochs):
  nn_model = tf.keras.Sequential([
      tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(24,)),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(num_nodes, activation='relu'),
      tf.keras.layers.Dropout(dropout),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

  nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="binary_crossentropy", 
                    metrics=['accuracy'])
  history = nn_model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
  
  return nn_model, history
```
```python
least_val_loss = float('inf')
least_loss_model = None
epochs = 100

for num_nodes in [16,32,64]:
  for dropout in [0,0.2]:
    for learning_rate in [0.005, 0.001, 0.01]:
      for batch_size in [32, 64, 128]:
        print(f"{num_nodes} nodes, dropout: {dropout}, learning_rate: {learning_rate}, batch_size: {batch_size}")
        model, history = train_model(X_train, y_train, num_nodes, dropout, learning_rate, batch_size, epochs)
        plot_hist(history)
        val_loss, val_accuracy = model.evaluate(X_val, y_val)
        print("Validation Loss:", val_loss)
        print("Validation Accuracy:", val_accuracy)
        if val_loss < least_val_loss:
          least_val_loss = val_loss
          least_loss_model = model
```
```python
y_predict = least_loss_model.predict(X_test)
y_predict = (y_predict > 0.5).astype(int).reshape(-1,)
```
```python
print(classification_report(y_test, y_predict))
```
##### Neural Network Report 

|           | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
|     0     |   1.00    |  0.58  |   0.73   |   19    |
|     1     |   0.80    |  1.00  |   0.89   |   32    |
|  Accuracy |           |        |   0.84   |   51    |
| Macro Avg |   0.90    |  0.79  |   0.81   |   51    |
| Weighted Avg |  0.87   |  0.84  |   0.83   |   51    |
