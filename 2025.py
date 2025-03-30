import pandas as pd
import joblib

model = joblib.load("f1_prediction_model.pkl")

# Load 2025 dataset
data_2025 = pd.read_csv("f1_2025.csv")

# Define features
features = ["grid", "averageFinish", "careerWins", "podiums", "totalRaces", 
            "constructorAvgPoints", "constructorReliability", "trackPerformance", 
            "gridEffect", "fastestLapSpeed", "laps"]

# Predict race winners
data_2025["predicted_winner"] = model.predict(data_2025[features])

# Determine winners for each Grand Prix
race_winners = data_2025.loc[data_2025.groupby("name_y")["predicted_winner"].idxmax(), ["name_y", "driverRef"]]
race_winners = race_winners.rename(columns={"name_y": "Grand Prix", "driverRef": "Predicted Winner"})

print("\n🏆 Predicted Grand Prix Winners:")
print(race_winners)

# Predict World Drivers' Champion (WDC)
wdc = data_2025.groupby("driverRef")["predicted_winner"].sum().idxmax()
print(f"\n🏁 Predicted World Drivers' Champion (WDC) for 2025: {wdc}")
