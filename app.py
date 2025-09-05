import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np

# ------------------------------
# Load scaler and model
# ------------------------------
scaler = joblib.load("scaler_dl.pkl")

# Define the exact same model architecture used in training
class FireNN_DL(nn.Module):
    def __init__(self, input_dim):
        super(FireNN_DL, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Feature order must match training
feature_names = ["PRECIPITATION", "MAX_TEMP", "MIN_TEMP", "TEMP_RANGE", 
                 "AVG_WIND_SPEED", "WIND_TEMP_RATIO", 
                 "LAGGED_PRECIPITATION", "LAGGED_AVG_WIND_SPEED", 
                 "SEASON_Fall", "SEASON_Spring", "SEASON_Summer", "SEASON_Winter"]

input_dim = len(feature_names)
model = FireNN_DL(input_dim)
model.load_state_dict(torch.load("best_fire_model_dl.pth", map_location=torch.device('cpu')))
model.eval()

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🔥 Fire Start Prediction App 🔥")
st.write("Predict the likelihood of a fire starting today based on environmental features.")

# Input sliders
PRECIPITATION = st.slider("Rate the Rain or Snowfalls", 0.0, 1.0, 0.0)
MAX_TEMP = st.slider("Max Temp (°C)", 0.0, 120.0, 75.0)
MIN_TEMP = st.slider("Min Temp (°C)", 0.0, 120.0, 50.0)
TEMP_RANGE = MAX_TEMP - MIN_TEMP
AVG_WIND_SPEED = st.slider("Average Wind Speed (mph)", 0.0, 25.0, 5.0)
WIND_TEMP_RATIO = AVG_WIND_SPEED / MAX_TEMP if MAX_TEMP != 0 else 0
LAGGED_PRECIPITATION = st.slider("Precipitation past 7 days", 0.0, 9.0, 0.0)
LAGGED_AVG_WIND_SPEED = st.slider("Wind past 7 days", 0.0, 15.0, 5.0)

# Season selection
season = st.selectbox("Select Season", ["Fall", "Spring", "Summer", "Winter"])
SEASON_Fall = 1 if season == "Fall" else 0
SEASON_Spring = 1 if season == "Spring" else 0
SEASON_Summer = 1 if season == "Summer" else 0
SEASON_Winter = 1 if season == "Winter" else 0

# Prepare input array in same order as training
inputs = np.array([[PRECIPITATION, MAX_TEMP, MIN_TEMP, TEMP_RANGE,
                    AVG_WIND_SPEED, WIND_TEMP_RATIO,
                    LAGGED_PRECIPITATION, LAGGED_AVG_WIND_SPEED,
                    SEASON_Fall, SEASON_Spring, SEASON_Summer, SEASON_Winter]])

# Scale input
inputs_scaled = scaler.transform(inputs)

# Convert to tensor
inputs_tensor = torch.tensor(inputs_scaled, dtype=torch.float32)

# Predict
if st.button("Predict Fire Start"):
    with torch.no_grad():
        pred = model(inputs_tensor).item()
        st.success(f"🔥 Probability of Fire Starting: {pred*100:.2f}% 🔥")
        if pred >= 0.5:
            st.warning("⚠️ High risk of fire!")
        else:
            st.info("✅ Low risk of fire.")
