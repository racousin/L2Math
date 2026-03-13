import numpy as np
import joblib

PARIS_IDX = 16
TEMP_IDX = 0
RAIN_IDX = 1
WIND_IDX = 2
WIND_DIR_IDX = 3
HUMIDITY_IDX = 4
CLOUDS_IDX = 5
SNOW_IDX = 7

FEATURE_INDICES = [TEMP_IDX, RAIN_IDX, WIND_IDX, WIND_DIR_IDX, HUMIDITY_IDX, CLOUDS_IDX, SNOW_IDX]


class Agent:
    def __init__(self):
        self.model = joblib.load("model.pkl")

    def predict(self, X_test):
        features = []

        # All cities, last hour, 7 weather features (excl visibility)
        for feat_idx in FEATURE_INDICES:
            for city_idx in range(20):
                features.append(X_test[city_idx, -1, feat_idx])

        # Paris lag features: T-1 to T-6 for temperature, wind_speed, rain
        for lag in range(1, 7):
            features.append(X_test[PARIS_IDX, -lag, TEMP_IDX])
            features.append(X_test[PARIS_IDX, -lag, WIND_IDX])
            features.append(X_test[PARIS_IDX, -lag, RAIN_IDX])

        # Time features (not available from X_test — use 0 as placeholder)
        features.extend([0, 0, 0])  # hour, day_of_week, month

        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)
        return prediction.flatten()[:3]
