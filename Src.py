#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, silhouette_score, adjusted_rand_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load the datasets
customers_df = pd.read_csv("./Datasets/Customers.csv")
transactions_df = pd.read_csv("./Datasets/Transactions_New.csv")

# Merge Customer Name into transactions_df
transactions_df = transactions_df.merge(customers_df[["CustomerID", "CustomerName"]], on="CustomerID", how="left")

# Aggregate total spending and transactions per customer
customer_spending = transactions_df.groupby("CustomerID").agg(
    Total_Spending=("TotalValue", "sum"),
    Total_Transactions=("TransactionID", "count")
).reset_index()

# Aggregate spending per product category
category_spending = transactions_df.pivot_table(
    index="CustomerID", columns="Category", values="TotalValue", aggfunc="sum", fill_value=0
).reset_index()

# Merge all features into a single dataset
customer_data = customer_spending.merge(category_spending, on="CustomerID", how="left")
customer_data = customer_data.merge(customers_df, on="CustomerID", how="left")

# Select only numerical columns for scaling
numeric_columns = customer_data.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_data[numeric_columns])

# Convert back to DataFrame
customer_scaled_df = pd.DataFrame(scaled_features, columns=numeric_columns)
customer_scaled_df.insert(0, "CustomerID", customer_data["CustomerID"])

# Define Autoencoder
input_dim = scaled_features.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(16, activation='relu')(input_layer)
encoded = Dense(8, activation='relu')(encoded)
encoded = Dense(3, activation='relu')(encoded)

decoded = Dense(8, activation='relu')(encoded)
decoded = Dense(16, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Compile the Autoencoder
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, shuffle=True, verbose=1)

# Extract the Encoder part
encoder = Model(input_layer, encoded)
encoded_features = encoder.predict(scaled_features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
customer_encoded_df = pd.DataFrame(encoded_features, columns=["Feature1", "Feature2", "Feature3"])
customer_encoded_df.insert(0, "CustomerID", customer_data["CustomerID"])
customer_encoded_df["Cluster"] = kmeans.fit_predict(encoded_features)

# Define cluster labels
cluster_labels = {
    0: "Low Spenders",
    1: "Medium Spenders",
    2: "High Spenders",
    3: "VIP Customers"
}
customer_encoded_df["Segment"] = customer_encoded_df["Cluster"].map(cluster_labels)
customer_encoded_df.drop(columns=["Cluster"], inplace=True)

# Merge with customer details
final_customer_df = customer_data.merge(customer_encoded_df, on="CustomerID", how="left")

def get_customer_info(customer_id):
    
    customer_details = final_customer_df[final_customer_df["CustomerID"] == customer_id]

    if customer_details.empty:
        return "Customer not found."
    
    customer_name = customer_details["CustomerName"].values[0]
    segment = customer_details["Segment"].values[0]
    details_text = f"Customer ID: {customer_id}\n"
    details_text += f"Customer Name: {customer_name}\n"
    details_text += f"Group: {segment}\n"
    details_text += f"Total Spending: ₹{customer_details['Total_Spending'].values[0]:.2f}\n"
    details_text += f"Total Transactions: {customer_details['Total_Transactions'].values[0]}\n\n"
    details_text += "Spending Breakdown:\n"
    for category in category_spending.columns[1:]:
        if category in customer_details.columns:
            value = float(customer_details[category].values[0])
        if value > 0:
            details_text += f"{category}: ₹{value:.2f}\n"
    return details_text  