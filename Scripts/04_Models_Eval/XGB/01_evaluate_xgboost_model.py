import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score, roc_auc_score
from NBA_modules import split_train_val_seasons
import shap
import seaborn as sns
import numpy as np
from datetime import datetime

shap.initjs()

def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as file:
        return pickle.load(file)

def define_features():
    """Return the list of features used for prediction."""
    return [
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

def evaluate_model(model, val_X, val_y):
    """Print evaluation metrics for the model."""
    predictions = model.predict(val_X)
    mae = mean_absolute_error(val_y, predictions)
    rmse = root_mean_squared_error(val_y, predictions)
    r2 = r2_score(val_y, predictions)
    auc = roc_auc_score(val_y, predictions)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    return predictions

def display_mvp_predictions(val_seasons, model, data, features):
    """Display MVP predictions for the last 10 validation seasons."""
    print("\nTop MVP predictions for the last 10 seasons:\n")
    for season in val_seasons[-10:]:
        analyze_season(season, model, data, features)

    # Compute average predicted MVP rank
    ranks = []
    for season in val_seasons[-10:]:
        season_data = data[data['season_year'] == season].copy()
        season_data['predicted_score'] = model.predict(season_data[features])
        actual_mvp = season_data[season_data['is_MVP'] == 1]
        actual_rank = season_data['predicted_score'].rank(ascending=False)[actual_mvp.index[0]]
        ranks.append(actual_rank)
    print(ranks)
    print(f"Average rank of the actual MVP: {sum(ranks) / len(ranks):.2f}")

def analyze_season(season_num, model, data, features):
    """
    Analyze model predictions for a given season.
    Display top candidates and actual MVP rank if not in top 5.
    """
    season_data = data[data['season_year'] == season_num].copy()
    season_data['predicted_score'] = model.predict(season_data[features])
    top_candidates = season_data.sort_values(by='predicted_score', ascending=False).head(5)

    print(f"\nTop 5 MVP Candidates for {season_num} Season:")
    for _, row in top_candidates.iterrows():
        actual = "🏆 MVP" if row['is_MVP'] == 1 else ""
        print(f"{row['player_name']}: {row['predicted_score']:.4f} {actual}")

    if 1 in season_data['is_MVP'].values and not any(top_candidates['is_MVP'] == 1):
        actual_mvp = season_data[season_data['is_MVP'] == 1]
        actual_rank = season_data['predicted_score'].rank(ascending=False)[actual_mvp.index[0]]
        print(f"Actual MVP {actual_mvp['player_name'].values[0]} ranked {int(actual_rank)} with score {actual_mvp['predicted_score'].values[0]:.4f}")

def analyze_corr_and_shap(val_X, shap_values, features, output_path, corr_threshold=0.85):
    """
    Combine correlation analysis and SHAP feature importance to suggest
    redundant features to remove.
    """
    corr_matrix = val_X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    correlated_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={'level_0': 'Feature_1', 'level_1': 'Feature_2', 0: 'Correlation'})
    )
    correlated_pairs = correlated_pairs[correlated_pairs['Correlation'] >= corr_threshold]

    shap_df = pd.DataFrame(shap_values.values, columns=features)
    shap_importance = shap_df.abs().mean().rename('SHAP_mean_importance')

    correlated_pairs['SHAP_Feature_1'] = correlated_pairs['Feature_1'].map(shap_importance)
    correlated_pairs['SHAP_Feature_2'] = correlated_pairs['Feature_2'].map(shap_importance)

    def choose_to_keep(row):
        return (row['Feature_1'], row['Feature_2']) if row['SHAP_Feature_1'] > row['SHAP_Feature_2'] else (row['Feature_2'], row['Feature_1'])

    correlated_pairs[['To_Keep', 'To_Consider_Removing']] = correlated_pairs.apply(choose_to_keep, axis=1, result_type='expand')

    result_df = correlated_pairs[['Feature_1', 'Feature_2', 'Correlation', 'SHAP_Feature_1', 'SHAP_Feature_2', 'To_Keep', 'To_Consider_Removing']]
    timestamp = datetime.today().strftime('%Y-%m-%d')
    result_df.to_csv(f"{output_path}/corr_SHAP{timestamp}.csv", index=False)

    return result_df.sort_values(by='Correlation', ascending=False)

def plot_feature_importance(model, features):
    """Plot bar chart of model's feature importances."""
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(model.feature_importances_)), model.feature_importances_, align='center')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance for MVP Prediction')
    plt.tight_layout()
    plt.show()

def main():
    """Main pipeline to load data/model, evaluate, explain and display predictions."""
    model_path = "/path/to/intput/model"
    data_path = "/path/to/intput/dataframe"
    output_path = "/path/to/output/folder"

    model = load_model(model_path)
    data = pd.read_csv(data_path)
    features = define_features()

    val_seasons, _ = split_train_val_seasons(data_path)
    val_mask = data['season_year'].isin(val_seasons)
    val_X = data.loc[val_mask, features]
    val_y = data.loc[val_mask, 'is_MVP']

    evaluate_model(model, val_X, val_y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(val_X)

    display_mvp_predictions(val_seasons, model, data, features)

    print("\nHighly correlated features and SHAP-based suggestions:")
    corr_shap_df = analyze_corr_and_shap(val_X, shap_values, features, output_path)
    print(corr_shap_df.head(10))

    corr = val_X.corr().unstack().sort_values(ascending=False).drop_duplicates()
    print(corr[corr >= .85])

    plot_feature_importance(model, features)

if __name__ == "__main__":
    main()