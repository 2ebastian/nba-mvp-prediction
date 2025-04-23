from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime

def scrap_all_seasons_advanced_stats(geckodriver_path, output_path, start_year: int, end_year: int):

    """
    Scrapes NBA players' advanced stats for each season within the given range 
    from Basketball-Reference.com and saves the data into a CSV file.

    This function uses Selenium with a headless Firefox browser to navigate through 
    the advanced stats pages for each NBA season between `start_year` and `end_year`.
    It extracts advanced player statistics (such as PER, True Shooting %, Usage %, 
    Win Shares, Box Plus/Minus, etc.) from regular seasons only.

    ⚠️ Important note:
    -----------------
    In this project, NBA seasons are referred to by the year when the season starts, 
    **not the year when it ends**. 
    For example:
    - `1980` refers to the **1980/1981** NBA season.
    - `2024` refers to the **2024/2025** NBA season.

    This differs from the convention often used by Basketball-Reference and the NBA 
    which name seasons based on the year when the season ends.

    For each season:
    - Scrapes player stats from the advanced stats table.
    - Identifies the season's MVP.
    - Adds columns indicating the MVP status and season year.
    - Combines all the data into a single DataFrame.
    - Saves the final combined dataset as a CSV file.

    Parameters:
    ----------
    geckodriver_path : str
        The file path to the `geckodriver` executable for controlling the Firefox browser.
    save_path : str
        The file path (including file name) where the final CSV file will be saved.
    start_year : int
        The starting year of the NBA seasons to scrape (inclusive, year when the season starts).
    end_year : int
        The ending year of the NBA seasons to scrape (exclusive, year when the season starts).

    Notes:
    -----
    - Requires `geckodriver` installed and available at the specified `geckodriver` path.
    - The function adapts to changes in the Basketball-Reference.com HTML structure 
      (specifically different IDs for standings tables before and after the 2015/2016 season).
    - The script is designed to be used as a standalone module or called programmatically.

    Example:
    --------
    geckodriver_path = "/opt/homebrew/bin/geckodriver"
    save_path = "/path/to/save/nba_players_AdvancedData_totals.csv"
    scrap_all_seasons_advanced_stats(geckodriver_path, save_path, 1980, 2025)
    # This will scrape advanced stats from the 1980/1981 season up to (but not including) the 2025/2026 season.
    """

    # Configure Firefox with Selenium
    options = Options()
    options.headless = True
    service = Service(geckodriver_path)
    driver = webdriver.Firefox(service=service, options=options)
    
    # Define column names outside the loop
    stat_names = [
        "player_name", "age", "team", "position", "game_played", "game_starter", "minutes_played",
        "efficiency_rating", "true_shooting_%", "3pt_attempt_rate", "FT_attempt_rate", "off_reb_%", 
        "def_reb_%", "total_reb_%", "assist_%", "steal_%", "blk_%", "turnover_%", "usage_%", 
        "off_win_shares", "def_win_shares", "total_win_shares", "ws_per_48", "off_box_+/-", "def_box_+/-",
        "box_+/-", "value_over_replacement"
    ]

    # Create an empty DataFrame to store all seasons
    all_seasons_advanced_df = pd.DataFrame()
    
    try:
        for year in range(start_year, end_year):
            year += 1
            print(f"📡 Scraping year {year}...")
            
            # Load the page
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
            driver.get(url)
            time.sleep(3)
            
            # Get HTML and parse with BeautifulSoup
            page_content = driver.page_source
            parser = BeautifulSoup(page_content, "html.parser")
            
            # Get player stats
            advanced_stats = parser.select("#advanced tr")
            player_stats = []
            
            for row in advanced_stats:
                cells = row.find_all('td')
                if cells:  # Check if we have valid cells
                    stats = {stat_name: cell.get_text(strip=True) 
                            for stat_name, cell in zip(stat_names, cells)}
                    if stats:
                        player_stats.append(stats)
            
            # # Get MVP
            find_mvp = parser.find_all("p")
            season_mvp = []
            for p in find_mvp:
                if "Most Valuable Player" in p.get_text():
                    season_mvp.append(p.find("a").get_text(strip=True))
                    break
            
            # Get season year
            season_year = parser.select("#info h1")[0].get_text()
            season_year_cleaned = season_year[1:5]
            
            # Create DataFrame for current season
            df = pd.DataFrame(player_stats)
            
            # Add MVP and season columns
            df["is_MVP"] = df["player_name"].apply(lambda x: 1 if season_mvp and x == season_mvp[0] else 0)
            df["season_year"] = season_year_cleaned.strip()
            
            # Ensure all columns are present
            for col in stat_names + ["is_MVP", "season_year"]:
                if col not in df.columns:
                    df[col] = None
            
            # Reorder columns to match stat_names
            df = df[stat_names + ["is_MVP", "season_year"]]
            
            # Append to main DataFrame
            all_seasons_advanced_df = pd.concat([all_seasons_advanced_df, df], ignore_index=True)
            
            print(f"✅ {len(df)} players recorded for season {year}")
        
        # Save all data to CSV at once
        timestamp = datetime.today().strftime('%Y-%m-%d')
        output_file = f"{output_path}/players_data_advanced{timestamp}.csv"
        all_seasons_advanced_df.to_csv(output_file, index=False)
        print("\nDataFrame info:")
        print(all_seasons_advanced_df.info())
        print("\nFirst few rows:")
        print(all_seasons_advanced_df.head())
        
    finally:
        driver.quit()
        print("🎉 Scraping completed! All seasons saved.")

# ===================================
# Execute the function
# ===================================

if __name__ == "__main__":
    geckodriver_path = "/path/to/geckodriver"
    output_path = "/path/to/output/folder"
    scrap_all_seasons_advanced_stats(geckodriver_path, output_path, 1980, 2024)
