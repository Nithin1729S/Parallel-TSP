import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('tsp_bruteforce_execution_times.csv')

# Convert Pandas Series to NumPy arrays to avoid the indexing issue
nodes = data['Nodes'].to_numpy()
execution_time = data['Execution_Time'].to_numpy()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(nodes, execution_time, marker='o', label='Bruteforce')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('TSP Brute Force Sequential Execution Time')
plt.grid(True)
plt.legend()

# Save the plot as an image file
plt.savefig('bruteforce_sequential.png')

# Display the plot
plt.show()
