import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import joblib


def main():
    df = pd.read_csv("assam_procurement_cleaned.csv")

    df = df.rename(columns={
        'tender/value/amount': 'amount',
        'tender/tenderPeriod/durationInDays': 'duration_days',
        'tender/numberOfTenderers': 'num_bidders',
        'tender/procurementMethod': 'proc_method',
        'buyer/name': 'buyer',
        'fiscal_year': 'year',
        'tender/status': 'status'
    })

    numeric_cols = ['amount', 'duration_days', 'num_bidders']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

    categorical_cols = ['proc_method', 'buyer', 'status']
    for col in categorical_cols:
        if df[col].isnull().all():
            df[col] = "Unknown"
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
            else:
                df[col] = df[col].fillna("Unknown")

    buyer_count = df['buyer'].value_counts()
    df['buyer_frequency'] = df['buyer'].map(buyer_count)
    df['proc_method_encoded'] = df['proc_method'].astype('category').cat.codes
    df['status_encoded'] = df['status'].astype('category').cat.codes

    features = [
        'amount',
        'duration_days',
        'num_bidders',
        'buyer_frequency',
        'proc_method_encoded',
        'status_encoded'
    ]

    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X_scaled)

    df['anomaly_score'] = model.decision_function(X_scaled)

    risk_scaler = MinMaxScaler(feature_range=(0,100))
    df['risk_score'] = risk_scaler.fit_transform(-df[['anomaly_score']])

    def categorize(score):
        if score > 70:
            return "High"
        elif score > 40:
            return "Medium"
        else:
            return "Low"

    df['risk_grade'] = df['risk_score'].apply(categorize)

    reasons = []
    for _, row in df.iterrows():
        r = []
        if row['num_bidders'] <= 1:
            r.append("Low bidder competition")
        if row['duration_days'] < 3:
            r.append("Very short tender duration")
        if row['amount'] > df['amount'].quantile(0.90):
            r.append("Unusually high contract value")
        if row['buyer_frequency'] > df['buyer_frequency'].quantile(0.90):
            r.append("High buyer concentration")
        if not r:
            r.append("No strong anomaly indicator")
        reasons.append(r)

    df['risk_reasons'] = reasons

    final_output = df[['amount', 'risk_score', 'risk_grade', 'risk_reasons']]
    final_output.to_csv("corruption_risk_report.csv", index=False)

    joblib.dump(model, "risk_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(risk_scaler, "risk_scaler.pkl")

    print("Training complete and models saved")


if __name__ == '__main__':
    main()
