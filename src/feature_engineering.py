"""Feature engineering module for credit card transaction data."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates derived features from raw transaction data."""

    def __init__(self):
        self.feature_names = []

    def create_time_features(self, df):
        """Extract time-based features from the Time column."""
        if "Time" not in df.columns:
            return df
        df = df.copy()
        df["Hour"] = (df["Time"] / 3600).astype(int) % 24
        df["DayPeriod"] = pd.cut(
            df["Hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True,
        )
        df["IsNight"] = (df["Hour"] < 6).astype(int) | (df["Hour"] > 22).astype(int)
        self.feature_names.extend(["Hour", "IsNight"])
        return df

    def create_amount_features(self, df):
        """Generate statistical features from transaction amounts."""
        df = df.copy()
        df["Amount_Log"] = np.log1p(df["Amount"])
        df["Amount_Squared"] = df["Amount"] ** 2
        df["Amount_Bin"] = pd.qcut(df["Amount"], q=10, labels=False, duplicates="drop")

        amount_mean = df["Amount"].mean()
        amount_std = df["Amount"].std()
        df["Amount_ZScore"] = (df["Amount"] - amount_mean) / (amount_std + 1e-8)
        df["Amount_IsHigh"] = (df["Amount_ZScore"] > 2).astype(int)

        self.feature_names.extend(["Amount_Log", "Amount_Squared", "Amount_Bin", "Amount_ZScore", "Amount_IsHigh"])
        return df

    def create_v_interaction_features(self, df, top_n=5):
        """Create interaction terms between top V-features."""
        v_cols = [c for c in df.columns if c.startswith("V")][:top_n]
        df = df.copy()
        for i in range(len(v_cols)):
            for j in range(i + 1, len(v_cols)):
                name = f"{v_cols[i]}_{v_cols[j]}_interaction"
                df[name] = df[v_cols[i]] * df[v_cols[j]]
                self.feature_names.append(name)
        return df

    def create_aggregate_features(self, df):
        """Create aggregate statistical features from V-columns."""
        v_cols = [c for c in df.columns if c.startswith("V")]
        df = df.copy()
        df["V_Mean"] = df[v_cols].mean(axis=1)
        df["V_Std"] = df[v_cols].std(axis=1)
        df["V_Max"] = df[v_cols].max(axis=1)
        df["V_Min"] = df[v_cols].min(axis=1)
        df["V_Skew"] = df[v_cols].skew(axis=1)
        self.feature_names.extend(["V_Mean", "V_Std", "V_Max", "V_Min", "V_Skew"])
        return df

    def engineer_features(self, df):
        """Run the full feature engineering pipeline."""
        df = self.create_time_features(df)
        df = self.create_amount_features(df)
        df = self.create_v_interaction_features(df)
        df = self.create_aggregate_features(df)
        print(f"Created {len(self.feature_names)} new features")
        return df
