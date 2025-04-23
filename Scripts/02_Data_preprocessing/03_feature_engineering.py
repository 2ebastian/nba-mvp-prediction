import pandas as pd
from datetime import datetime
import numpy as np

def engineering(df_cleaned, output_path):

    """
    Perform advanced feature engineering on cleaned NBA player data.

    This function applies multiple transformations and dimensionality 
    reductions on a cleaned dataset, including:
    
    - Ranking teams by conference and season based on win percentage.
    - Computing per-game statistics to reduce season length bias (e.g. due to lockdowns).
    - Organizing features into logical groups for readability and consistency.
    - Dropping highly correlated or redundant features based on corr-SHAP analysis.
    - Saving the final processed DataFrame to a timestamped CSV file.

    Args:
        df_cleaned (str): Path to the cleaned CSV file.
        output_path (str): Directory where the final CSV file will be saved.

    Returns:
        pd.DataFrame: The engineered DataFrame ready for model training.
        str: The path to the saved CSV file.
    """

    try:
        # Load cleaned DataFrame
        df_final = pd.read_csv(df_cleaned)

        # Feature engineering : ranking teams by conference and seasons
        df_final["team_standing"] = (df_final.groupby(["season_year", "conf"])["win_pct"]
                                     .rank(method="dense", ascending=False)
                                     .astype(int))
        
        # Averaging stats per game to reduce season lockdowns bias 
        stats_to_avg = ['field_goal_made', 'field_goal_attempts', 'three_points_made', 'three_points_attempts', 
                        'two_points_made', 'two_points_attempts', 'free_throws_made', 'free_throws_attempts',
                        'offensive_rebonds', 'defensive_rebonds', 'total_rebonds', 'assists', 'steals', 'blocks', 
                        'turnovers', 'personal_fouls', 'total_points']

        for stat in stats_to_avg:
            df_final[f'{stat}_per_game'] = (df_final[stat] / df_final['game_played'].replace(0, np.nan)).fillna(0)

        # Reorganizing columns 
        names_cols = ['player_name']
        player_info = [
            'position_C', 'position_PF', 'position_PG', 'position_SF', 'position_SG', 'age', 
            'conf', 'win_pct', 'team_standing', 'game_played', 'game_starter', 'minutes_played'
            ]
        shooting_stats = [
            'field_goal_made', 'field_goal_attempts', 'field_goal_percentage', 'three_points_made', 
            'three_points_attempts', 'three_points_percentage', 'two_points_made', 'two_points_attempts', 
            'two_points_percentage', 'effective_fg_percentage', 'free_throws_made','free_throws_attempts', 
            'free_throws_percentage'
            ] 
        performance_stats = [
            'offensive_rebonds', 'defensive_rebonds', 'total_rebonds', 'assists', 'steals', 'blocks', 
            'turnovers', 'personal_fouls', 'total_points', 'triple_double'
            ] 
        
        per_game_stats = [f'{stat}_per_game' for stat in stats_to_avg]

        advanced_stats = [
            "efficiency_rating", "true_shooting_%", "3pt_attempt_rate", "FT_attempt_rate", "off_reb_%", 
            "def_reb_%", "total_reb_%", "assist_%", "steal_%", "blk_%", "turnover_%", "usage_%", 
            "off_win_shares", "def_win_shares", "total_win_shares", "ws_per_48", "off_box_+/-", "def_box_+/-",
            "box_+/-", "value_over_replacement"
            ]
        meta_cols = ['season_year', 'is_MVP']

        df_final = df_final[names_cols + player_info + shooting_stats + performance_stats + per_game_stats + advanced_stats+ meta_cols]     
        
        # Dimensionality reduction : Drop highly correlated features based on corr-SHAP analysis
        df_final = df_final.drop(columns=["field_goal_made", "three_points_made", "two_points_made", "free_throws_made",
                                            "offensive_rebonds", "defensive_rebonds", "win_pct", "minutes_played", 
                                            "personal_fouls", "game_starter", "off_box_+/-", "off_win_shares", "def_win_shares",
                                            "two_points_attempts", "effective_fg_percentage", "efficiency_rating"])           

        # Saving new dataframe
        timestamp = datetime.today().strftime('%Y-%m-%d')
        output_file = f"{output_path}/03_df_advances_ready_for_training{timestamp}.csv"
        df_final.to_csv(output_file, index=False)
        print("🎉 Dataframe processed and ready for training! File saved as CSV.")   

        return df_final, output_file

    except Exception as e:
        print(f"An error occurred during data engineering: {e}")
        raise

if __name__ == "__main__": 
    clean_df_path = "/path/to/intput/cleaned_dataframe"
    output_path = "/path/to/output/folder"
    engineering(clean_df_path, output_path)