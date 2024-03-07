import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# Load team-specific CSV files
raptors_data = pd.read_csv("Raptors.csv")
suns_data = pd.read_csv("Suns.csv")

# Select relevant features for prediction
features = ["REB", "AST", "STL", "BLK", "TOV", "FG%", "3P%", "FT%"]

# Extract stats for each team
raptors_stats = raptors_data[features].values
suns_stats = suns_data[features].values

# Standardize features
scaler = StandardScaler()
raptors_stats_scaled = scaler.fit_transform(raptors_stats)
suns_stats_scaled = scaler.transform(suns_stats)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(len(features),)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model (you can use historical data if available)
# For demonstration purposes, let's assume the Raptors have better recent performance
model.fit(raptors_stats_scaled, np.ones(len(raptors_stats)), epochs=10, verbose=0)
model.fit(suns_stats_scaled, np.zeros(len(suns_stats)), epochs=10, verbose=0)

# Make predictions for the hypothetical game
hypothetical_game_stats = np.array([[40, 25, 8, 5, 12, 45.0, 35.0, 80.0]])  # Example stats for the hypothetical game
hypothetical_game_stats_scaled = scaler.transform(hypothetical_game_stats)
predicted_result = model.predict(hypothetical_game_stats_scaled)[0][0]

# Display predicted result
if predicted_result > 0.5:
    winner = "Toronto Raptors"
else:
    winner = "Phoenix Suns"

# Plot team stats
plt.figure(figsize=(10, 6))
plt.bar(features, raptors_stats_scaled.mean(axis=0), label="Toronto Raptors", alpha=0.7)
plt.bar(features, suns_stats_scaled.mean(axis=0), label="Phoenix Suns", alpha=0.7)
plt.xlabel("Statistics")
plt.ylabel("Scaled Value")
plt.title("Team Statistics Comparison")
plt.legend()
plt.grid(axis="y")
plt.show()

print(f"Predicted Winner: {winner}")
