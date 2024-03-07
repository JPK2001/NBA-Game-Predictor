import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Load data from CSV files
team1_data = pd.read_csv("Raptors.csv")
team2_data = pd.read_csv("Suns.csv")

# Convert non-numeric columns to numeric
team1_data = team1_data.apply(pd.to_numeric, errors='coerce').fillna(0)
team2_data = team2_data.apply(pd.to_numeric, errors='coerce').fillna(0)

# Analyze recent performance stats
def calculate_stats(team_data):
    stats = {}
    stats['PTS/G'] = team_data['TM_PTS'].mean()
    stats['REB/G'] = team_data['REB'].mean()
    stats['AST/G'] = team_data['AST'].mean()
    stats['FG%'] = (team_data['FGM'] / team_data['FGA']).mean()
    stats['3P%'] = (team_data['3PM'] / team_data['3PA']).mean()
    stats['FT%'] = (team_data['FTM'] / team_data['FTA']).mean()
    return stats

team1_stats = calculate_stats(team1_data)
team2_stats = calculate_stats(team2_data)

# Visualize stats
def visualize_stats(team1_stats, team2_stats):
    stats = list(team1_stats.keys())
    team1_values = list(team1_stats.values())
    team2_values = list(team2_stats.values())

    bar_width = 0.35
    index = np.arange(len(stats))

    plt.figure(figsize=(12, 8))
    plt.bar(index, team1_values, bar_width, label='Team 1', color='skyblue')
    plt.bar(index + bar_width, team2_values, bar_width, label='Team 2', color='orange')

    plt.xlabel('Stats')
    plt.ylabel('Values')
    plt.title('Recent Performance Stats')
    plt.xticks(index + bar_width / 2, stats)
    plt.legend()
    plt.show()

visualize_stats(team1_stats, team2_stats)

# Predict upcoming opposition stats
def prepare_data(team1_stats, team2_stats):
    X = team1_stats
    y = team2_stats
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(team1_data, team2_data)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), verbose = 0)

# Predict upcoming game stats
def predict_stats(team_data, model):
    scaled_data = StandardScaler().fit_transform(team_data)
    predictions = model.predict(scaled_data)
    return predictions.mean()

team1_predicted_stats = predict_stats(team1_data, model)
team2_predicted_stats = predict_stats(team2_data, model)

# Compare predicted stats and determine the winner
def determine_winner(team1_stats, team2_stats):
    if team1_stats > team2_stats:
        return "Team 1"
    elif team1_stats < team2_stats:
        return "Team 2"
    else:
        return "Tie"

winner = determine_winner(team1_predicted_stats, team2_predicted_stats)
print(f"The predicted winner is: {winner}")

# Visualize predicted stats
def visualize_predicted_stats(team1_predicted_stats, team2_predicted_stats):
    teams = ['Team 1', 'Team 2']
    predicted_stats = [team1_predicted_stats, team2_predicted_stats]
    plt.figure(figsize=(8, 5))
    plt.bar(teams, predicted_stats, color=['blue', 'orange'])
    plt.xlabel('Teams')
    plt.ylabel('Predicted Stats')
    plt.title('Predicted Stats for the Upcoming Game')
    plt.show()

visualize_predicted_stats(team1_predicted_stats, team2_predicted_stats)
