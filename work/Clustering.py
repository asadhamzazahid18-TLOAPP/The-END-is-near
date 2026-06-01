import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# pop = pd.read_csv(
# 'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')

pop = pd.read_csv(
    'work/ai-adoption-fortune500-synthetic-dataset-2020-2025.csv', delimiter=',')

print(pop.head(50))

# Step 1 - Clean data
pop_clean = pop[pop['AI_ROI_Percent'] > 0].copy()
pop_clean = pop_clean.reset_index(drop=True)

le = LabelEncoder()

# Encode categorical variables
pop_clean['Industry_Encoded'] = le.fit_transform(pop_clean['Industry'])
pop_clean['Country_Encoded'] = le.fit_transform(pop_clean['Country'])
pop_clean['Company_Type_Encoded'] = le.fit_transform(pop_clean['Company_Type'])
pop_clean['Employee_encoded'] = le.fit_transform(pop_clean['Employee_Size'])


x = pop_clean[['AI_Maturity_Score', 'AI_ROI_Percent']]

# Feature Scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
pop_clean['Cluster'] = kmeans.fit_predict(x_scaled)


# Inverse transform cluster centers to original scale
centers = scaler.inverse_transform(kmeans.cluster_centers_)


# Visualization
plt.figure(figsize=(10, 6))
plt.grid(True)
plt.scatter(pop_clean['AI_Maturity_Score'], pop_clean['AI_ROI_Percent'],
            c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(label='Cluster')
plt.xlabel('AI_Maturity_Score')
plt.ylabel('AI_ROI_Percent')
plt.title('AI Business Clustering (K=5)')
plt.legend()
plt.show()
print(pop_clean.groupby('Cluster')[
      ['AI_Maturity_Score', 'AI_ROI_Percent']].mean())


print('-----------------')

# Analyze clustering based on Revenue and AI ROI
x = pop_clean[['AI_ROI_Percent', 'Revenue_USD']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


kmeans = KMeans(n_clusters=5, random_state=42)
pop_clean['Cluster'] = kmeans.fit_predict(x_scaled)


centers = scaler.inverse_transform(kmeans.cluster_centers_)


plt.figure(figsize=(10, 6))
plt.grid(True)
plt.scatter(pop_clean['AI_ROI_Percent'], pop_clean['Revenue_USD'],
            c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(label='Cluster')
plt.xlabel('AI_ROI_Percent')
plt.ylabel('Revenue_USD')
plt.title('AI Business Clustering (K=5)')
plt.legend()
plt.show()
print(pop_clean.groupby('Cluster')[
      ['AI_ROI_Percent', 'Revenue_USD']].mean())

print('_____-----__________')


x = pop_clean[['Year', 'Revenue_USD']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


kmeans = KMeans(n_clusters=5, random_state=42)
pop_clean['Cluster'] = kmeans.fit_predict(x_scaled)


centers = scaler.inverse_transform(kmeans.cluster_centers_)


plt.figure(figsize=(10, 6))
plt.grid(True)
plt.scatter(pop_clean['Year'], pop_clean['Revenue_USD'],
            c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(label='Cluster')
plt.xlabel('Year')
plt.ylabel('Revenue-USD')
plt.title('AI Business Clustering (K=5)')
plt.legend()
plt.show()
print(pop_clean.groupby('Cluster')[
      ['Year', 'Revenue_USD']].mean())
