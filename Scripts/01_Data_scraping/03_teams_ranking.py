from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime

def scrap_all_seasons_standings(geckodriver_path, output_path, start_year: int, end_year: int):

    """
    Scrapes NBA team standings (team name, conference, and win percentage) for each season 
    within the given range from Basketball-Reference.com, and saves the data into a CSV file.

    This function uses Selenium with a headless Firefox browser to navigate through 
    the standings pages for each NBA season between `start_year` and `end_year`.
    It extracts for each team:
    - The team name
    - Its conference (Eastern 'E' or Western 'W')
    - Its regular season win percentage

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
    - Scrapes standings separately for both Eastern and Western Conferences.
    - Extracts the season's year, team names, win percentages, and conference labels.
    - Merges both conferences' data into a single DataFrame.
    - Appends each season's data to a list.
    - Concatenates all seasons into one final DataFrame.
    - Saves the final combined dataset to a CSV file.

    Parameters:
    -----------
    geckodriver : str
        The file path to the `geckodriver` executable for controlling the Firefox browser.
    save_path : str
        The file path (including file name) where the final CSV file will be saved.
    start_year : int
        The starting year of the NBA seasons to scrape (inclusive, year when the season starts).
    end_year : int
        The ending year of the NBA seasons to scrape (exclusive, year when the season starts).

    Notes:
    ------
    - Requires `geckodriver` installed and available at the specified `geckodriver` path.
    - The function adapts to changes in the Basketball-Reference.com HTML structure 
      (specifically different IDs for standings tables before and after the 2015/2016 season).
    - The script is designed to be used as a standalone module or called programmatically.

    Example:
    --------
    geckodriver_path = "/path/to/geckodriver"
    save_path = "/path/to/save/teams_standings.csv"
    scrap_all_seasons_standings(geckodriver_path, save_path, 1980, 2025)
    # This will scrape team standings from the 1980/1981 season up to (but not including) the 2025/2026 season.
    """

    # Configure Firefox with Selenium
    options = Options()
    options.headless = True
    service = Service(geckodriver_path)
    driver = webdriver.Firefox(service=service, options=options)
    
    all_seasons_data = []

    try : 
        for year in range(start_year, end_year):
            year += 1
            print(f"📡 Scraping year {year}...")
            
            # Load the page
            url = f"https://www.basketball-reference.com/leagues/NBA_{year}_standings.html"
            driver.get(url)
            time.sleep(3)
            
            # Get HTML and parse with BeautifulSoup
            page_content = driver.page_source
            parser = BeautifulSoup(page_content, "html.parser")
            
            # Get teams standings
            # For East
            if year in range(1981, 2016) :
                eastern_conf = parser.select("#divs_standings_E .full_table")
            else:
                eastern_conf = parser.select("#confs_standings_E .full_table")

            eastern_conf_teams = []
            for row in eastern_conf:
                teams_cells = row.find('a').text
                wins_cells = row.find(attrs={"data-stat": "win_loss_pct"}).text
                #conf_cell = row.find(attrs={"aria-label":"Eastern Conference"}).text
                if teams_cells and wins_cells:
                    east_team_data = {
                        "team": teams_cells,
                        "conf" : "E",
                        "win_pct": wins_cells
                    }
                    eastern_conf_teams.append(east_team_data)

            # For West
            if year in range(1981, 2016) :
                west_conf = parser.select("#divs_standings_W .full_table")
            else:
                west_conf = parser.select("#confs_standings_W .full_table")

            western_conf_teams = []
            for row in west_conf:
                teams_cells = row.find('a').text
                wins_cells = row.find(attrs={"data-stat": "win_loss_pct"}).text
                #conf_cell = row.find(attrs={"aria-label":"Western Conference"}).text
                if teams_cells and wins_cells:
                    west_team_data = {
                        "team": teams_cells,
                        "conf" : "W",
                        "win_pct": wins_cells
                    }
                    western_conf_teams.append(west_team_data)

            # Merging all the teams together 
            all_teams = western_conf_teams + eastern_conf_teams 

            # Get season year
            season_year = parser.select("#info h1")[0].get_text()
            season_year_cleaned = season_year[1:5]
            
            df = pd.DataFrame(all_teams)
            df["season_year"] = season_year_cleaned.strip()

            # Save every season in a list outside of the loop
            all_seasons_data.append(df)

            print(f"✅ {len(df)} teams recorded for season {year}")

        final_df = pd.concat(all_seasons_data, ignore_index=True)

        # Save all data to CSV at once
        timestamp = datetime.today().strftime('%Y-%m-%d')
        output_file = f"{output_path}/teams_standings{timestamp}.csv"
        final_df.to_csv(output_file, index=False)
        print("\nDataFrame info:")
        print(final_df.info())
        print("\nFirst few rows:")
        print(final_df.head())

    finally: 
        driver.quit()
        print("🎉 Scraping completed! All seasons saved.")    
        
# ===================================
# Execute the function
# ===================================

if __name__ == "__main__":
    geckodriver_path = "/path/to/geckodriver"
    output_path = "/path/to/output/folder"
    scrap_all_seasons_standings(geckodriver_path, output_path, 1980, 2024)


