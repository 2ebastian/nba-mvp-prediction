import pandas as pd
import pickle
import numpy as np


import pandas as pd
import pickle
import numpy as np

# Load the trained model from a pickle file
def load_model(model_path):
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def predict_the_mvp(dataframe, season, features, top_n=3, display=True):
    """
    Predict MVP candidates for a given season and display the top N predictions.
    
    Parameters:
    - dataframe: full dataset containing all seasons
    - season: season year to predict (e.g. 2025)
    - features: list of features used by the model
    - top_n: number of top candidates to display
    - display: whether to print the results
    
    Returns:
    - A DataFrame sorted by predicted MVP score for the season
    """
    season_data = dataframe[dataframe['season_year'] == season].copy()

    if season_data.empty:
        raise ValueError(f"Aucune donnée trouvée pour la saison {season}.")

    # Results prediction
    season_data['predicted_score'] = model.predict(season_data[features])
    
    # Players ranking
    season_data = season_data.sort_values(by='predicted_score', ascending=False)

    # Focus on top N
    top_players = season_data.head(top_n).copy()

    # Normalize over top N
    top_scores = top_players['predicted_score'].values
    total_top_scores = np.sum(top_scores)
    top_players['predicted_probability'] = (top_scores / total_top_scores * 100).round(2)

    if display:
        print(f"\nTop {top_n} MVP predictions for the {season} - {season + 1} season:\n")
        for i, (_, row) in enumerate(top_players.iterrows(), 1):
            print(f"{i}. {row['player_name']}: {row['predicted_probability']:.2f}%")
    
    return season_data


if __name__ == "__main__":
    model_path = "/Users/sebastianestephe/Desktop/Python_-_Projet_Perso/03-ML_NBA_MVP/Models/mvp_xgb_advancedfeatures_model2025-04-22.pkl"
    data_path = "/Users/sebastianestephe/Desktop/Python_-_Projet_Perso/03-ML_NBA_MVP/Data/03_df_2024_ready_for_prediction2025-04-22.csv"

    # Load model and dataframe
    model = load_model(model_path)
    dataframe = pd.read_csv(data_path)

    season = 2024

    features = [
        'position_C', 'position_PF', 'position_PG', 'position_SF', 'position_SG', 'age', 'conf', 'team_standing',
        'game_played', 'field_goal_attempts', 'field_goal_percentage', 'three_points_attempts',
        'three_points_percentage', 'two_points_percentage',
        'free_throws_attempts', 'free_throws_percentage', 'total_rebonds', 'assists', 'steals', 'blocks', 'turnovers',
        'total_points', 'triple_double', 'true_shooting_%', '3pt_attempt_rate', 'FT_attempt_rate', 'off_reb_%',
        'def_reb_%', 'total_reb_%', 'assist_%', 'steal_%', 'blk_%', 'turnover_%', 'usage_%',
        'total_win_shares', 'ws_per_48', 'def_box_+/-', 'box_+/-', 'value_over_replacement',
        'field_goal_made_per_game', 'field_goal_attempts_per_game', 'three_points_made_per_game',
        'three_points_attempts_per_game', 'two_points_made_per_game', 'two_points_attempts_per_game',
        'free_throws_made_per_game', 'free_throws_attempts_per_game', 'offensive_rebonds_per_game',
        'defensive_rebonds_per_game', 'total_rebonds_per_game', 'assists_per_game', 'steals_per_game',
        'blocks_per_game', 'turnovers_per_game', 'personal_fouls_per_game', 'total_points_per_game'
    ]

    predict_the_mvp(dataframe, season, features, top_n=3, display=True)