V1.0

🔍 Context : 

    As a tech enthusiast and passionate NBA follower, I wanted to create a project that blends both worlds. This project explores the potential of machine learning to simulate the complex, human-driven decision of selecting the NBA’s Most Valuable Player (MVP) each season.

🎯 Objective :

    Using historical data from the 1980 to 2024 NBA seasons, this project aims to predict the MVP based solely on player and team statistics. The goal is to understand the key performance indicators that influence MVP voting and to evaluate whether a model can reasonably identify the league’s top performer.

📊 Data : 

    All data was collected from basketball-reference.com, and includes:

    1 - Player basic and advanced statistics (e.g. points, assists, efficiency, usage, BPM, win shares…)
    2 - Team performance metrics (e.g. win percentage, standing)
    3 - Seasons from 1980 onward (data prior to 1980 was excluded due to differences in MVP voting methodology)

⚙️ Project Structure & Pipeline :

    NBA_MVP_Project/
    │
    ├── 01_Data_Scraping/
    │   ├── 01_players_stats.py
    │   ├── 1.1_Advanced_players_stats.py
    │   └── 02_teams_ranking.py
    │
    ├── 02_Data_preprocessing/
    │   ├── 01_Merging_data_to_main_df.py
    │   ├── 02_Main_df_data_cleaning.py
    │   └── 03_feature_engineering.py
    │
    ├── 03_Models/
    │   └── XGB/
    │       └── 01_xgboost_training.py
    │
    ├── 04_Evaluation/
    │   └── XGB/
    │       └── 01_evaluate_xgboost_model.py
    │
    ├── 05_Predictions/
    │   └── 01_predict.py

🔍 Key Steps in the Pipeline:

    1 - Scraping: Retrieve and clean raw player and team data from the web
    2 - Preprocessing: Merge datasets, clean missing values, standardize features
    3 - Feature Engineering: Create meaningful variables (e.g. advanced stats, usage rates)
    4 - Model Training: Use XGBoost for regression-based MVP scoring
    5 - Evaluation: Analyze correlation, SHAP values, and prediction rankings
    6 - Prediction: Generate a ranked list of top MVP candidates per season

🧠 Model : 

    The chosen algorithm is XGBoost, a powerful gradient boosting framework well-suited for tabular data and feature importance analysis.

📈 Evaluation : 

    The model does not attempt to classify a single MVP per season, but instead predicts an MVP score for each player, which allows us to rank players by their likelihood of being chosen MVP.

    To assess the performance and relevance of the MVP prediction model, several evaluation metrics and analytical tools were used :

        1 Regression Metrics :

        Although the task is to rank MVP candidates (rather than predict a binary label), the model was evaluated using regression metrics to quantify the difference between predicted scores and actual MVP labels:
            •	Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.
            •	Root Mean Squared Error (RMSE): Penalizes larger errors more than MAE, giving a better sense of the model’s precision.
            •	R² Score: Indicates how well the model explains the variability of the target.
            •	AUC-ROC: Despite the regression approach, ROC-AUC was also calculated to measure the ability of the model to separate MVPs from non-MVPs.

        2 MVP Prediction Quality :

        For a qualitative analysis, the model was tested on the 10 most recent seasons in the validation set. For each season:
            •	The top 5 predicted players were displayed alongside the actual MVP.
            •	If the true MVP wasn’t in the top 5, their rank and predicted score were shown.

        Across the last 10 seasons, the average predicted rank of the actual MVP was computed to provide a global view of the model’s ability to identify MVP-level players. This metric gives insight into how well the model distinguishes elite performances.

        3 Feature Importance & Interpretability =

        Two approaches were used to understand and refine feature importance:
            •	SHAP (SHapley Additive exPlanations) values were computed to assess the contribution of each feature to the predictions on the validation set.
            •	A combined analysis of feature correlation and SHAP importance was conducted to identify redundant features. Highly correlated feature pairs (correlation ≥ 0.85) were compared, and the less important one (according to SHAP) was flagged for potential removal.

        A bar chart was generated to visualize the most influential features, providing interpretability and helping guide future feature engineering.

⚠️ Limitations & Future Improvements :

    1 - Feature engineering could be further optimized
    2 - Social/media dynamics (popularity, narrative buzz) are not modeled
    3 - Voter fatigue (e.g., multiple consecutive MVPs) is not directly integrated
    4 - Model does not simulate vote counts but instead extrapolates from stats
    5 - Consider using neural networks to capture non-linear or contextual patterns