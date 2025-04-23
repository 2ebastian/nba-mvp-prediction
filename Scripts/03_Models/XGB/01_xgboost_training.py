import pandas as pd
import pickle
from xgboost import XGBRegressor
from NBA_modules import split_train_val_seasons
from datetime import datetime

def train_mvp_model(input_path, output_path):

    """
    Train an XGBoost regression model to predict NBA MVP scores based on player statistics.

    This function reads a preprocessed dataset from a CSV file, selects relevant features,
    splits the data into training and validation sets based on season years, trains an 
    XGBoost Regressor with early stopping, and saves the trained model as a pickle file.

    Args:
        input_path (str): Path to the CSV file containing the processed dataset.
        output_path (str): Directory path where the trained model will be saved.

    Returns:
        None. The trained model is saved to disk as a pickle (.pkl) file.

    Notes:
        - The train/validation split is based on custom logic implemented in the
          `split_train_val_seasons` function, which defines which seasons are used for 
          training and validation.
        - The trained model is saved with a timestamp in its filename to avoid overwriting.
        - The model uses early stopping based on validation set performance to prevent overfitting.
    """
    
    # Load preprocessed dataset
    data = pd.read_csv(input_path)
    
    # Define the list of features to be used for training
    features = [
        'position_C', 'position_PF', 'position_PG', 'position_SF', 'position_SG', 'age', 'conf', 'team_standing',
        'game_played', 'field_goal_attempts', 'field_goal_percentage', 'three_points_attempts',
        'three_points_percentage', 'two_points_percentage',
        'free_throws_attempts', 'free_throws_percentage', 'total_rebonds', 'assists', 'steals', 'blocks', 'turnovers',
        'total_points', 'triple_double', 'true_shooting_%', '3pt_attempt_rate', 'FT_attempt_rate', 'off_reb_%',
        'def_reb_%', 'total_reb_%', 'assist_%', 'steal_%', 'blk_%', 'turnover_%', 'usage_%',
        'total_win_shares', 'ws_per_48', 'def_box_+/-', 'box_+/-', 'value_over_replacement',

        # Per game stats
        'field_goal_made_per_game', 'field_goal_attempts_per_game', 'three_points_made_per_game',
        'three_points_attempts_per_game', 'two_points_made_per_game', 'two_points_attempts_per_game',
        'free_throws_made_per_game', 'free_throws_attempts_per_game', 'offensive_rebonds_per_game',
        'defensive_rebonds_per_game', 'total_rebonds_per_game', 'assists_per_game', 'steals_per_game',
        'blocks_per_game', 'turnovers_per_game', 'personal_fouls_per_game', 'total_points_per_game'
    ]

    # Split the dataset into training and validation seasons using a custom function
    val_seasons, train_seasons = split_train_val_seasons(input_path)

    # Create boolean masks for train and validation sets based on season year
    train_mask = data['season_year'].isin(train_seasons)
    val_mask = data['season_year'].isin(val_seasons)

    # Split the data into features (X) and target (y) for training and validation sets
    train_X = data.loc[train_mask, features]
    train_y = data.loc[train_mask, 'is_MVP']
    val_X = data.loc[val_mask, features]
    val_y = data.loc[val_mask, 'is_MVP']

    # Initialize and train the XGBoost Regressor with early stopping on validation RMSE
    model = XGBRegressor(n_estimators=3000, learning_rate=0.01, eval_metric="rmse", early_stopping_rounds=50)
    model.fit(train_X, train_y, eval_set=[(val_X, val_y)], verbose=False)

    # Generate a timestamp to uniquely identify model file and save it
    timestamp = datetime.today().strftime('%Y-%m-%d')
    model_filename = f"{output_path}/mvp_xgb_advancedfeatures_model{timestamp}.pkl"
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)

    # Confirmation message
    print(f"Modèle entraîné et sauvegardé dans {model_filename}") 

if __name__ == "__main__":
    input_path = "/path/to/intput/engineered_dataframe"
    output_path = "/path/to/output/Model_folder"
    train_mvp_model(input_path, output_path)