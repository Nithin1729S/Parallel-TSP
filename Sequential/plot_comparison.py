

import pandas as pd
import matplotlib.pyplot as plt

data_bruteforce = pd.read_csv('./Bruteforce/tsp_bruteforce_execution_times.csv')
data_dp = pd.read_csv('./DP/tsp_dp_execution_times.csv')

plt.figure(figsize=(10, 6))


plt.plot(data_bruteforce['Nodes'], data_bruteforce['Execution_Time'], marker='o', label='Bruteforce')


plt.plot(data_dp['Nodes'], data_dp['Execution_Time'], marker='o', label='DP (Held-Karp)')


plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('TSP Execution Times: Brute Force vs Dynamic Programming')
plt.grid(True)
plt.legend()

plt.savefig('tsp_combined_execution_times.png')



