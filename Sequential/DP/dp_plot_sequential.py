import pandas as pd
import matplotlib.pyplot as plt

data_dp = pd.read_csv('tsp_dp_execution_times.csv')
plt.figure(figsize=(10, 6))
plt.plot(data_dp['Nodes'], data_dp['Execution_Time'], marker='o', label='DP (Held-Karp)')
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('TSP DP (Held-Karp) Sequential Execution Time')
plt.grid(True)
plt.legend()
plt.savefig('tsp_dp_sequential.png')
