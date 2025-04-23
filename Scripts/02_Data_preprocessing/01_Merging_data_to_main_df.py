import pandas as pd 
from datetime import datetime

def main_dataframe(teams_ranking_path, players_stats_path, advanced_stats_path, output_path): 

    """
    Builds the main NBA players dataframe by merging player statistics, team standings, and advanced player metrics.

    This function:
    - Loads team rankings data and maps full team names to abbreviations.
    - Loads player statistics data and merges it with team rankings based on team and season.
    - Loads advanced player statistics and merges it with the combined dataset based on player name, team, and season.
    - Selects a specific list of statistical and advanced metrics columns.
    - Exports the final dataframe to a CSV file with a timestamp in the filename.

    Parameters:
    -----------
    teams_ranking_path : str
        Path to the CSV file containing team standings data.
    
    players_stats_path : str
        Path to the CSV file containing player season totals data.
    
    advanced_stats_path : str
        Path to the CSV file containing advanced player statistics data.

    Returns:
    --------
    None
        The function saves the final merged dataframe as a CSV file and prints a success message.

    Notes:
    ------
    - The generated CSV file is saved in the same directory as the input files, with a timestamp in its filename.
    - The function assumes specific column names in the input CSV files for proper merging.
    - A mapping dictionary is used internally to convert team full names to abbreviations for consistency.
    """

    # Dictionnaire de mapping des noms d'équipes
    team_name_to_abbr = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BRK",
        "Chicago Bulls": "CHI",
        "Charlotte Hornets": "CHO",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHO",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
        "Charlotte Bobcats": "CHA",
        "Kansas City Kings": "KCK",
        "New Jersey Nets": "NJN",
        "San Diego Clippers": "SDC",
        "Seattle SuperSonics": "SEA",
        "Washington Bullets": "WSB",
        "Vancouver Grizzlies": "VAN",
        "New Orleans Hornets": "NOH",
        "Charlotte Hornets": "CHH",
        "New Orleans/Oklahoma City Hornets": "NOK"
    }

    try:
        # Load team rankings data and Convert team full names into abbreviations for merging
        df_ranking = pd.read_csv(teams_ranking_path)
        df_ranking["team"] = df_ranking["team"].map(team_name_to_abbr)

        # Load player statistics data and Merge player stats with team rankings based on team and season
        df_players = pd.read_csv(players_stats_path)
        df_augmented = pd.merge(df_players, df_ranking, on=['team', 'season_year'], how='left')

        # Load advanced player statistics data and Merge with advanced stats using player name, team, and season
        df_advanced_stats = pd.read_csv(advanced_stats_path)
        df_final = pd.merge(df_augmented, df_advanced_stats, on=['player_name','team', 'season_year'], how='left', suffixes=('', '_adv'))

        # Select columns for the final dataset
        cols = [
            'player_name', 'position', 'age', 'team', 'conf', 'win_pct', 'game_played', 'game_starter', 'minutes_played', 
            'field_goal_made', 'field_goal_attempts', 'field_goal_percentage', 'three_points_made', 'three_points_attempts', 
            'three_points_percentage', 'two_points_made', 'two_points_attempts', 'two_points_percentage', 
            'effective_fg_percentage', 'free_throws_made', 'free_throws_attempts', 'free_throws_percentage', 'offensive_rebonds', 
            'defensive_rebonds', 'total_rebonds', 'assists', 'steals', 'blocks', 'turnovers', 'personal_fouls', 'total_points', 
            'triple_double'
        ]
        
        advanced_cols = [
            "efficiency_rating", "true_shooting_%", "3pt_attempt_rate", "FT_attempt_rate", "off_reb_%", 
            "def_reb_%", "total_reb_%", "assist_%", "steal_%", "blk_%", "turnover_%", "usage_%", 
            "off_win_shares", "def_win_shares", "total_win_shares", "ws_per_48", "off_box_+/-", "def_box_+/-",
            "box_+/-", "value_over_replacement"
        ]           
        
        meta_cols = ['season_year', 'is_MVP']

        # Keep only the selected columns to create the final dataframe
        df_final = df_final[cols + advanced_cols + meta_cols]

        # Generate timestamped filename and save
        timestamp = datetime.today().strftime('%Y-%m-%d')
        output_file = f"{output_path}/01_df_row_{timestamp}.csv"
        df_final.to_csv(output_file, index=False)

    finally:
        print("Main dataframe created successfully.")

# ===================================
# Execute the function
# ===================================

if __name__ == "__main__":
    teams_ranking = "/path/to/intput/file"
    players_stats = "/path/to/intput/file"
    advanced_stats = "/path/to/intput/file"
    output_path = "/path/to/output/folder"

    main_dataframe(teams_ranking, players_stats, advanced_stats, output_path)
