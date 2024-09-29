import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('tsp_bruteforce_execution_times.csv')
plt.figure(figsize=(10, 6))
plt.plot(data['Nodes'], data['Execution_Time'], marker='o', label='Bruteforce')  # This adds the label
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('TSP Brute Force Sequential Execution Time')
plt.grid(True)
plt.legend()
plt.savefig('bruteforce_sequential.png')

