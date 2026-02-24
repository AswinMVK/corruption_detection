from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load and preprocess data
df = pd.read_csv("assam_procurement_cleaned.csv")

# Rename columns
df = df.rename(columns={
    'tender/value/amount': 'amount',
    'tender/tenderPeriod/durationInDays': 'duration_days',
    'tender/numberOfTenderers': 'num_bidders',
    'tender/procurementMethod': 'proc_method',
    'buyer/name': 'buyer',
    'fiscal_year': 'year',
    'tender/status': 'status'
})

# Handle missing values
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

# Feature engineering
buyer_count = df['buyer'].value_counts()
df['buyer_frequency'] = df['buyer'].map(buyer_count)
df['proc_method_encoded'] = df['proc_method'].astype('category').cat.codes
df['status_encoded'] = df['status'].astype('category').cat.codes

# Load models
model = joblib.load("risk_model.pkl")
scaler = joblib.load("scaler.pkl")
risk_scaler = joblib.load("risk_scaler.pkl")

# Features list
features = [
    'amount',
    'duration_days',
    'num_bidders',
    'buyer_frequency',
    'proc_method_encoded',
    'status_encoded'
]

# Prepare X for the full dataset
X = df[features]
X_scaled = scaler.transform(X)

# Compute anomaly scores, risk scores, grades
df['anomaly_score'] = model.decision_function(X_scaled)
df['risk_score'] = risk_scaler.transform(-df[['anomaly_score']])

def categorize(score):
    if score > 70:
        return "High"
    elif score > 40:
        return "Medium"
    else:
        return "Low"

df['risk_grade'] = df['risk_score'].apply(categorize)

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# Compute thresholds
high_amount = df['amount'].quantile(0.90)
high_buyer_freq = df['buyer_frequency'].quantile(0.90)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analysis')
def analysis():
    # Generate risk grade distribution plot
    fig, ax = plt.subplots()
    df['risk_grade'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
    ax.set_title('Risk Grade Distribution')
    ax.set_xlabel('Risk Grade')
    ax.set_ylabel('Count')
    plt.savefig('static/images/risk_grade.png')
    plt.close()

    # Generate risk score distribution plot
    fig2, ax2 = plt.subplots()
    df['risk_score'].hist(ax=ax2, bins=20, color='lightgreen')
    ax2.set_title('Risk Score Distribution')
    ax2.set_xlabel('Risk Score')
    ax2.set_ylabel('Frequency')
    plt.savefig('static/images/risk_score.png')
    plt.close()

    # Generate PCA clusters plot
    fig3, ax3 = plt.subplots()
    scatter = ax3.scatter(df['pca1'], df['pca2'], c=df['cluster'], cmap='viridis')
    ax3.set_title('Clusters in PCA Space')
    ax3.set_xlabel('PCA Component 1')
    ax3.set_ylabel('PCA Component 2')
    plt.colorbar(scatter)
    plt.savefig('static/images/clusters_pca.png')
    plt.close()

    # Generate amount vs num_bidders by cluster
    fig4, ax4 = plt.subplots()
    scatter2 = ax4.scatter(df['amount'], df['num_bidders'], c=df['cluster'], cmap='viridis')
    ax4.set_title('Amount vs Number of Bidders by Cluster')
    ax4.set_xlabel('Amount')
    ax4.set_ylabel('Number of Bidders')
    plt.colorbar(scatter2)
    plt.savefig('static/images/clusters_amount_bidders.png')
    plt.close()

    # Compute cluster descriptions
    cluster_desc = df.groupby('cluster')[features].mean().round(2)
    cluster_dict = cluster_desc.to_dict('index')

    return render_template('analysis.html', cluster_dict=cluster_dict)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        amount = float(request.form['amount'])
        duration_days = float(request.form['duration_days'])
        num_bidders = float(request.form['num_bidders'])
        buyer_frequency = float(request.form['buyer_frequency'])
        proc_method_encoded = float(request.form['proc_method_encoded'])
        status_encoded = float(request.form['status_encoded'])

        manual_input = pd.DataFrame([{
            'amount': amount,
            'duration_days': duration_days,
            'num_bidders': num_bidders,
            'buyer_frequency': buyer_frequency,
            'proc_method_encoded': proc_method_encoded,
            'status_encoded': status_encoded
        }])

        manual_input = manual_input[features]
        manual_scaled = scaler.transform(manual_input)
        anomaly_score = model.decision_function(manual_scaled)[0]

        risk_input = pd.DataFrame({'anomaly_score': [-anomaly_score]})
        risk_score = risk_scaler.transform(risk_input)[0][0]

        if risk_score > 70:
            risk_grade = "High"
        elif risk_score > 40:
            risk_grade = "Medium"
        else:
            risk_grade = "Low"

        reasons = []
        if num_bidders <= 1:
            reasons.append("Low bidder competition")
        if duration_days < 3:
            reasons.append("Very short tender duration")
        if amount > high_amount:
            reasons.append("Unusually high contract value")
        if buyer_frequency > high_buyer_freq:
            reasons.append("High buyer concentration")
        if not reasons:
            reasons.append("No strong anomaly indicator")

        return render_template('predict.html', risk_score=round(risk_score, 2), risk_grade=risk_grade, reasons=reasons, submitted=True)

    return render_template('predict.html', submitted=False)

@app.route('/records')
def records():
    # Show first 100 records to avoid large page
    table = df.head(100).to_html(classes='table table-striped', index=False)
    return render_template('records.html', table=table)

if __name__ == '__main__':
    app.run(debug=True)