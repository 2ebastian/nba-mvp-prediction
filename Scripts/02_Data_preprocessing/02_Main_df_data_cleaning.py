import pandas as pd

from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


def clean_data(raw_df, output_path):

    """
    Clean and preprocess NBA player statistics data for machine learning.

    This function performs a series of data cleaning and preparation steps 
    on a raw CSV file containing NBA player statistics, including:
    - Removing league average rows and aggregated multi-team rows.
    - Encoding categorical variables (conference and position).
    - Reordering columns into logical feature groups.
    - Setting percentage columns from 0-100% to 0-1 scale.
    - Handling division-by-zero cases in shooting percentages.
    - Imputing missing values with the median of each column.
    - Saving the cleaned DataFrame as a timestamped CSV file.

    Args:
        raw_df (str): Path to the raw CSV data file.
        output_path (str): Directory where the cleaned CSV file will be saved.

    Returns:
        tuple:
        pd.DataFrame: The cleaned and preprocessed DataFrame ready for feature engineering.
        str: The path to the saved CSV file containing the cleaned data.

    Raises:
        Exception: If any error occurs during the data cleaning process, it is printed and re-raised.
    """

    try :

        df = pd.read_csv(raw_df)

        # Remove "League Average" lines
        df = df.drop(df[df["player_name"] == "League Average"].index)

        # Remove lines with value 2TM, 3TM... under team serie
        multiple_teams = ["2TM", "3TM", "4TM", "5TM"]
        df = df.drop(df[df["team"].isin(multiple_teams)].index)

        # Drop team serie
        df = df.drop(columns=["team"])

        # Encoding string values (conf, position, team)
        # conf (binary : E = 0 ; W = 1)
        df["conf"] = df["conf"].map({"E": 0, "W": 1})

        # position (OneHotEncoder)
        ohc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohc_position = ohc.fit_transform(df["position"].values.reshape(-1, 1))
        position_columns = ohc.get_feature_names_out(['position'])
        position_df = pd.DataFrame(ohc_position, columns=position_columns, index=df.index)
        df = pd.concat([df, position_df], axis=1)
        df = df.drop("position", axis=1)

        # Repositionning Encoded positions
        names_cols = ['player_name']

        player_info = [
            'position_C', 'position_PF', 'position_PG', 'position_SF', 'position_SG', 'age', 
            'conf', 'win_pct', 'game_played', 'game_starter', 'minutes_played'
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
        advanced_stats = [
            "efficiency_rating", "true_shooting_%", "3pt_attempt_rate", "FT_attempt_rate", "off_reb_%", 
            "def_reb_%", "total_reb_%", "assist_%", "steal_%", "blk_%", "turnover_%", "usage_%", 
            "off_win_shares", "def_win_shares", "total_win_shares", "ws_per_48", "off_box_+/-", "def_box_+/-",
            "box_+/-", "value_over_replacement"
            ]
        meta_cols = ['season_year', 'is_MVP']

        df = df[names_cols + player_info + shooting_stats + performance_stats + advanced_stats + meta_cols]

        #df = df.drop(columns=["position_nan"])

        # if data in shooting attempts = 0, pct = 0
        df.loc[df["three_points_attempts"] == 0, "three_points_percentage"] = 0.0
        df.loc[df["two_points_attempts"] == 0, "two_points_percentage"] = 0.0
        df.loc[df["free_throws_attempts"] == 0, "free_throws_percentage"] = 0.0

        # Rescale 0-100% cols to 0-1 format
        rescale_perc_cols = ["off_reb_%", "def_reb_%", "total_reb_%", "assist_%", "steal_%", "blk_%", "turnover_%", "usage_%"]
        df[rescale_perc_cols] = (df[rescale_perc_cols] / 100).round(4)

        # Find columns with missing values and replace it with column's mediane
        my_imputer = SimpleImputer(strategy="median")
        df[df.columns[df.isnull().any()]] = my_imputer.fit_transform(df[df.columns[df.isnull().any()]])

        # Save to CSV at the end
        timestamp = datetime.today().strftime('%Y-%m-%d')
        output_file = f"{output_path}/02_df_cleaned_ready_for_engineering{timestamp}.csv"
        df.to_csv(output_file, index=False)
        print("🎉 Dataframe processed and ready for engineering! File saved as CSV.")
        
        # Return the processed dataframe
        return df, output_file

    except Exception as e :
        print(f"An error occurred during data cleaning: {e}")
        raise

# Execute the function
if __name__ == "__main__": 
    raw_df = "/path/to/intput/dataframe"
    output_path = "/path/to/output/folder"
    clean_data(raw_df, output_path)