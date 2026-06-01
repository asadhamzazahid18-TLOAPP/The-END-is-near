import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

pop = pd.read_csv('project-2/AI_Impact_on_Jobs_2030.csv', delimiter=',')
print(pop.head(50))

# Step 1 - Clean data
pop_clean = pop[pop['Average_Salary'] > 0].copy()
pop_clean = pop_clean.reset_index(drop=True)

# Fix: separate encoders for each column so they don't overwrite each other
le_job = LabelEncoder()
le_edu = LabelEncoder()
le_risk = LabelEncoder()

pop_clean['Job_Encoded'] = le_job.fit_transform(pop_clean['Job_Title'])
pop_clean['Education_Encoded'] = le_edu.fit_transform(
    pop_clean['Education_Level'])
pop_clean['Risk_Encoded'] = le_risk.fit_transform(pop_clean['Risk_Category'])

# Verify inverse transform works
pop_clean['Job_Title_Check'] = le_job.inverse_transform(
    pop_clean['Job_Encoded'])

x = pop_clean[['Job_Encoded', 'Automation_Probability_2030']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

sn.lineplot(pop_clean, x='AI_Exposure_Index', y='Tech_Growth_Factor',
            hue='Automation_Probability_2030', sizes=(8, 15), palette='pastel').set(title='AI impacts on JOB')
plt.title("Will AI exposure affect tech growth?")
plt.show()

# K-Means Clustering - Automation Probability (K=6)
kmeans = KMeans(n_clusters=6, random_state=42)
pop_clean['Cluster'] = kmeans.fit_predict(x_scaled)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Fix: map Job_Encoded back to Job_Title for x-axis labels
plt.figure(figsize=(12, 6))
plt.grid(True)
scatter = plt.scatter(pop_clean['Job_Encoded'], pop_clean['Automation_Probability_2030'],
                      c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers[:, 0], centers[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')

# Fix: show actual job titles on x-axis instead of encoded numbers
job_ticks = sorted(pop_clean['Job_Encoded'].unique())
job_labels = le_job.inverse_transform(job_ticks)
plt.xticks(ticks=job_ticks, labels=job_labels, rotation=90, fontsize=7)

plt.xlabel('Job Title')
plt.ylabel('Automation_Probability_2030')
plt.title('AI impact on jobs (K=6)')
plt.legend()
plt.tight_layout()
plt.show()

print(pop_clean.groupby('Cluster')[
      ['Job_Title', 'Automation_Probability_2030']].apply(lambda x: x))
print('-----------------')

# K-Means Clustering - Job vs Salary (K=5)
scaler2 = StandardScaler()
x_scaled2 = scaler2.fit_transform(x)

kmeans2 = KMeans(n_clusters=5, random_state=42)
pop_clean['Cluster'] = kmeans2.fit_predict(x_scaled2)

centers2 = scaler2.inverse_transform(kmeans2.cluster_centers_)

plt.figure(figsize=(12, 6))
plt.grid(True)
plt.scatter(pop_clean['Job_Encoded'], pop_clean['Average_Salary'],
            c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers2[:, 0], centers2[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(label='Cluster')

# Fix: actual job titles on x-axis
plt.xticks(ticks=job_ticks, labels=job_labels, rotation=90, fontsize=7)

plt.xlabel('Job Title')
plt.ylabel('Average_Salary')
plt.title('Jobs and salaries (K=5)')
plt.legend()
plt.tight_layout()
plt.show()

print(pop_clean.groupby('Cluster')[
      ['Job_Title', 'Average_Salary']].apply(lambda x: x))
print('_____-----__________')

# K-Means Clustering - Years Experience vs Salary (K=5)
x3 = pop_clean[['Years_Experience', 'Average_Salary']]

scaler3 = StandardScaler()
x_scaled3 = scaler3.fit_transform(x3)

kmeans3 = KMeans(n_clusters=5, random_state=42)
pop_clean['Cluster'] = kmeans3.fit_predict(x_scaled3)

centers3 = scaler3.inverse_transform(kmeans3.cluster_centers_)

plt.figure(figsize=(10, 6))
plt.grid(True)
plt.scatter(pop_clean['Years_Experience'], pop_clean['Average_Salary'],
            c=pop_clean['Cluster'], cmap='viridis', alpha=0.6)
plt.scatter(centers3[:, 0], centers3[:, 1],
            c='red', marker='X', s=200, zorder=5, label='Centroids')
plt.colorbar(label='Cluster')
plt.xlabel('Years_Experience')
plt.ylabel('Average_Salary')
plt.title('Years and Experience (K=5)')
plt.legend()
plt.tight_layout()
plt.show()

print(pop_clean.groupby('Cluster')[
      ['Years_Experience', 'Average_Salary']].mean())

# job title is the real deciding factor .
# it shapes your salary, automation risk, and how much your experience even matters. Experience alone won't save you if your role is built to be automated.
