import pandas as pd

# Read the dataset from CSV
df = pd.read_csv(r'C:\\Users\\avant\\OneDrive\\Desktop\\PHASE 2\\Execution\\datasets\\mins.csv')

# Group by userId and sort by serendipity score
grouped = df.groupby('userId').apply(lambda x: x.sort_values('serendipity_score', ascending=False))

# Define k (top k serendipity scores to keep)
k = 5

# Select top k serendipity scores for each user
top_k_serendipity = grouped.groupby('userId').head(k)

# Save to CSV
top_k_serendipity.to_csv('top_k_serendipity_scores.csv', index=False)
