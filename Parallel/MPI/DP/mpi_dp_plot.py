import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('mpi_dp_execution_times.csv')
plt.figure(figsize=(10, 6))
plt.plot(data['Nodes'], data['Execution_Time'], marker='o', color='b', linestyle='-')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Number of Nodes for TSP')
plt.yscale('log')  # Optional: Use logarithmic scale if values vary widely

# Set x-axis ticks to be integer nodes from the data
plt.xticks(data['Nodes'])  # Use the unique node values for x-axis ticks
plt.grid(True)
plt.savefig('tsp_dp_mpi.png')