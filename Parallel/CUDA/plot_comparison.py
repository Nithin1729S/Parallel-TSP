import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('tsp_cuda.csv')  # Ensure your file is named appropriately

# Extract columns into separate lists
nodes = data['Nodes'].tolist()
execution_time_bruteforce = data['Execution_Time_Bruteforce'].tolist()
execution_time_cuda = data['Execution_Time_CUDA'].tolist()

# Set the figure size
plt.figure(figsize=(14, 7))

# Plot the execution times for both approaches
plt.plot(nodes, execution_time_bruteforce, label='Brute Force', marker='o')
plt.plot(nodes, execution_time_cuda, label='DP', marker='x')

# Adding labels and title
plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title('CUDA Execution Time Comparison: Brute Force vs DP ')
plt.legend()
plt.grid()


# Optionally, save the plot as an image
plt.savefig('execution_time_comparison.png')
