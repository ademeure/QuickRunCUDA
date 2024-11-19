import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import argparse

# Define the input and output file paths
input_file = 'sm0246_latency_l2_BIG.csv'
output_file = 'sm0246_latency_l2_BIG_pivoted.csv'

# Read the CSV file
# - header=None: No header in the CSV
# - names=['Key', 'Category', 'Value']: Assign column names
# - skipinitialspace=True: Skip spaces after delimiters
df = pd.read_csv(input_file, header=None, names=['Key', 'Category', 'Value'], skipinitialspace=True)

# Display the original DataFrame (optional)
print("Original DataFrame:")
print(df)

# Pivot the DataFrame
# - index='Key': Unique identifiers for rows
# - columns='Category': Unique identifiers for columns
# - values='Value': Data to populate the cells
# - aggfunc='first': In case of duplicates, take the first occurrence
pivot_df = df.pivot_table(index='Key', columns='Category', values='Value', aggfunc='first')

# Reset the index to turn 'Key' back into a column
pivot_df.reset_index(inplace=True)

# Optional: Rename the category columns for clarity
# This will rename columns like 0 and 2 to 'Category_0' and 'Category_2'
pivot_df.columns = ['Key'] + [f'Category_{int(col)}' for col in pivot_df.columns if col != 'Key']

# Prepare data for clustering (exclude 'Key' column)
X = pivot_df.iloc[:, 1:].values
# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
# You can adjust n_clusters based on your needs
parser = argparse.ArgumentParser()
parser.add_argument('--n-clusters', type=int, default=80, help='Number of clusters for KMeans (default: 80)')
args = parser.parse_args()
n_clusters = args.n_clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=2000, tol=1e-5, n_init=20)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
pivot_df['Cluster'] = cluster_labels
# Calculate statistics for each cluster and store them
cluster_stats = []
for cluster in range(n_clusters):
    cluster_data = pivot_df[pivot_df['Cluster'] == cluster]
    means = cluster_data.iloc[:, 1:-1].mean()
    stds = cluster_data.iloc[:, 1:-1].std()
    # Find outliers for each category
    outliers = {}
    for col in cluster_data.columns[1:-1]:  # Exclude Key and Cluster columns
        data = cluster_data[col]
        mean = data.mean()
        std = data.std()
        # Get largest 5 outliers
        largest_outliers = data.nlargest(5)
        outliers[col] = largest_outliers.tolist()
        # Get smallest 5 outliers (in opposite order)
        smallest_outliers = data.nsmallest(5)[::-1]
        outliers[col] += smallest_outliers.tolist()

    cluster_stats.append({
        'cluster': cluster,
        'size': len(cluster_data),
        'means': means,
        'stds': stds,
        'max_std': stds.max(),
        'outliers': outliers
    })

# Sort clusters by maximum standard deviation
cluster_stats.sort(key=lambda x: x['max_std'], reverse=True)

# Print top 10 clusters with largest deviations
print("\nTop 10 Clusters with Largest Deviations:")
for i, stats in enumerate(cluster_stats[:10]):
    print(f"\nCluster {stats['cluster']} (Size: {stats['size']}, Max StdDev: {stats['max_std']:.3f})")
    print("Mean values:")
    print(stats['means'])
    print("Standard deviation:")
    print(stats['stds'])
    if stats['outliers']:
        print("Outliers:")
        for col, values in stats['outliers'].items():
            print(f"{col}: {values} ==> diff: {(max(values) - min(values)):.3f}")

# Print size of all clusters (ordered)
for i, stats in enumerate(cluster_stats):
    print(f"{i}: {stats['size']}")

# Save the clustered DataFrame
pivot_df.to_csv(output_file, index=False)

# Save the 1st cluster
cluster_data = pivot_df[pivot_df['Cluster'] == cluster_stats[0]['cluster']]
cluster_data.to_csv('sm0246_latency_l2_BIG_cluster0.csv', index=False)

# Save the cluster which has key 25 in it (the entire cluster, not just that row)
key25_cluster = pivot_df[pivot_df['Key'] == 25]['Cluster'].iloc[0]  # Get cluster number containing key 25
cluster_data = pivot_df[pivot_df['Cluster'] == key25_cluster]  # Get all rows in that cluster
cluster_data.to_csv('sm0246_latency_l2_BIG_key25.csv', index=False)

print(f"\nTransformed data with clusters has been saved to '{output_file}'.")