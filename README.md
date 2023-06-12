# Independent Study
###  ⚽Enhancing Player Selection Strategies: Machine Learning for Manchester United's Lineup Prediction
In this project, I conducted experiments utilizing a combination of traditional machine learning classifiers and neural network models implemented using TensorFlow. The primary objective was to predict the optimal player lineup for Manchester United based on historical statistics. To obtain the necessary data, I performed web scraping from "https://fbref.com/", extracting relevant information such as player performance metrics and match data.

Through extensive experimentation, I tested multiple traditional machine learning models. However, after careful evaluation, three models demonstrated the most potential in effectively working with this dataset. These models included Logistic Regression, K-Nearest Neighbors (KNN), and Support Vector Machines (SVM). Additionally, I implemented neural network models using TensorFlow to leverage the power of deep learning for improved predictions.

#### Data preparation
In the initial phase of this project, I performed web scraping on the [2022-2023 Manchester United Stats (Premier League)](https://fbref.com/en/players/b853e0ad/matchlogs/2022-2023/Fred-Match-Logs) to collect relevant data. I focused on extracting statistics specifically for the midfield position. To ensure data quality, I conducted cleansing procedures by removing unnecessary columns and handling missing values. To address missing data, data interpolation was employed.

Data interpolation is essential in this context as it allows for the estimation or filling in of missing values, ensuring a complete dataset for analysis. By utilizing interpolation techniques, we can infer and estimate values based on existing data patterns, thereby preserving the integrity of the dataset. This step enhances the reliability and comprehensiveness of subsequent analyses and predictions.

# Manchester United Midfielder 2022-2023 Match Logs (Summary)

This repository contains match logs for the midfielders of Manchester United for the 2022-2023 Premier League season. The provided data includes various performance metrics, expected goals (xG), shot-creating actions (SCA), goal-creating actions (GCA), passing statistics, carrying statistics, and take-ons.

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

