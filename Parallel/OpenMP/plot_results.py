import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
data = pd.read_csv('tsp_dp_execution_times.csv')

# Plot 1: Execution Time vs Number of Threads (for a fixed number of nodes = 20)
constant_nodes = 20
filtered_data_nodes = data[data['Nodes'] == constant_nodes]
schedules = ['static', 'dynamic', 'guided']
colors = ['blue', 'orange', 'green']

plt.figure(figsize=(10, 6))
for schedule, color in zip(schedules, colors):
    schedule_data = filtered_data_nodes[filtered_data_nodes['Schedule_Type'] == schedule]
    threads = schedule_data['Threads'].to_numpy()
    execution_time = schedule_data['Execution_Time'].to_numpy()
    plt.plot(threads, execution_time, label=schedule, marker='o', color=color)

plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title(f'Execution Time vs Number of Threads (Nodes = {constant_nodes})')
plt.legend(title="Schedule Type")
plt.grid(True)
plt.savefig('Execution_Time_vs_Number_of_Threads.png', dpi=300)
plt.show()

# Plot 2: Execution Time vs Number of Nodes (for a fixed number of threads = 8)
constant_threads = 8
filtered_data_threads = data[data['Threads'] == constant_threads]

plt.figure(figsize=(10, 6))
for schedule, color in zip(schedules, colors):
    schedule_data = filtered_data_threads[filtered_data_threads['Schedule_Type'] == schedule]
    plt.plot(schedule_data['Nodes'].to_numpy(), schedule_data['Execution_Time'].to_numpy(), label=schedule, marker='o', color=color)

plt.xlabel('Number of Nodes')
plt.ylabel('Execution Time (seconds)')
plt.title(f'Execution Time vs Number of Nodes (Threads = {constant_threads})')
plt.legend(title="Schedule Type")
plt.grid(True)
plt.savefig('Execution_Time_vs_Number_of_Nodes.png', dpi=300)
plt.show()
