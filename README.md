# Independent Study
###  ⚽Enhancing Player Selection Strategies: Machine Learning for Manchester United's Lineup Prediction
In this project, I conducted experiments utilizing a combination of traditional machine learning classifiers and neural network models implemented using TensorFlow. The primary objective was to predict the optimal player lineup for Manchester United based on historical statistics. To obtain the necessary data, I performed web scraping from "https://fbref.com/", extracting relevant information such as player performance metrics and match data.

Through extensive experimentation, I tested multiple traditional machine learning models. However, after careful evaluation, three models demonstrated the most potential in effectively working with this dataset. These models included Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). Additionally, I implemented neural network models using TensorFlow to leverage the power of deep learning for improved predictions.

#### Data preparation
In the initial phase of this project, I performed web scraping on the [2022-2023 Manchester United Stats (Premier League)](https://fbref.com/en/players/b853e0ad/matchlogs/2022-2023/Fred-Match-Logs) to collect relevant data. I focused on extracting statistics specifically for the midfield position. To ensure data quality, I conducted cleansing procedures by removing unnecessary columns and handling missing values. To address missing data, data interpolation was employed.

Data interpolation is essential in this context as it allows for the estimation or filling in of missing values, ensuring a complete dataset for analysis. By utilizing interpolation techniques, we can infer and estimate values based on existing data patterns, thereby preserving the integrity of the dataset. This step enhances the reliability and comprehensiveness of subsequent analyses and predictions.
