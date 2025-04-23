import pandas as pd

file_path = "/path/to/intput/engineered_dataframe"

def split_train_val_seasons(file_path):
    """
    Split NBA seasons into training and validation sets for MVP prediction modeling.
    
    This function loads NBA data from a CSV file and divides the seasons into separate
    training and validation sets. Instead of using random splitting from sklearn, 
    this approach gives precise control over which seasons are used for training
    and which for validation, ensuring temporal consistency in the evaluation.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing NBA data with a 'season_year' column
        
    Returns:
    --------
    tuple
        A tuple containing two sorted lists:
        - val_set: List of season years used for validation (odd years from 2005-2023)
        - train_set: List of remaining season years used for training (excluding 2024)
        
    Raises:
    -------
    Exception
        If there's an error reading the file or processing the data
        
    Notes:
    ------
    - Validation set includes every other year starting from 2005 up to 2023
    - 2024 season is explicitly excluded from both training and validation sets
    - This approach ensures chronological separation between training and validation
    """

    try:
        data = pd.read_csv(file_path)
        seasons = set(map(int, data["season_year"].unique()))  # Convert all to int immediately

        val_set = {2023, 2021, 2019, 2017, 2015, 2013, 2011, 2009, 2007, 2005}  # Set for easy subtraction
        train_set = seasons - val_set - {2024}  # Exclude validation seasons and 2024

        return sorted(list(val_set)), sorted(list(train_set))  # Return sorted lists for clarity

    except Exception as e:
        print(f"An error occurred during validation or training season splits: {e}")
        raise

val_set, train_set = split_train_val_seasons(file_path)
print("Validation Seasons:", val_set)
print("Training Seasons:", train_set)

if __name__ == "__main__": 
    split_train_val_seasons(file_path)