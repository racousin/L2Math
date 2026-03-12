import numpy as np
import joblib


# 20 cities in alphabetical order (same as competition)
CITIES = [
    "Amsterdam", "Barcelona", "Birmingham", "Brussels", "Copenhagen",
    "Dortmund", "Dublin", "Düsseldorf", "Essen", "Frankfurt am Main",
    "Köln", "London", "Manchester", "Marseille", "Milan", "Munich",
    "Paris", "Rotterdam", "Stuttgart", "Turin",
]
PARIS_IDX = CITIES.index("Paris")  # 16

# Feature indices in X_test axis 2
# 0: temperature, 1: rain, 2: wind_speed, 3: wind_direction,
# 4: humidity, 5: clouds, 6: visibility, 7: snow


class Agent:
    def __init__(self):
        self.model = joblib.load("model.pkl")

    def predict(self, X_test):
        """
        X_test: np.ndarray of shape (20, 24, 8)
            - 20 cities (alphabetical), 24 hours of history, 8 features
        Returns: np.ndarray of shape (3,)
            - [temperature, wind_speed, rain] for Paris at T+6h
        """
        # Extract features the same way as in TP1
        # Paris last-known values
        paris = X_test[PARIS_IDX]  # (24, 8)

        # Example: use last-hour values from all cities for temp, wind, rain
        # Flatten last hour across all cities
        features = []
        for city_idx in range(20):
            last_hour = X_test[city_idx, -1, :]  # (8,)
            features.extend([last_hour[0], last_hour[1], last_hour[2]])  # temp, rain, wind

        # Add Paris lag features (last 6 hours)
        for lag in range(1, 7):
            features.extend([
                paris[-lag, 0],  # temperature
                paris[-lag, 1],  # rain
                paris[-lag, 2],  # wind_speed
            ])

        # Add time-like features from pattern (hour can be inferred from position)
        # Note: you may want to add more features here

        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)
        return prediction.flatten()[:3]
