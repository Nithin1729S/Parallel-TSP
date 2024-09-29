import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file with execution times
data = pd.read_csv('tsp_execution_times.csv')

# Plotting the execution time vs number of nodes
plt.plot(data['Nodes'], data['Execution_Time'], marker='o')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('TSP Brute Force Sequential Execution Time')
plt.grid(True)
plt.savefig('bruteforce_sequential.png')
plt.show()
