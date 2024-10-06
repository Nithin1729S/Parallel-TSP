import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv("tsp_dp_execution_times.csv")

# Helper function to compute the average execution time for runtime
def compute_runtime_average(subset, nodes_or_threads, group_by_col):
    avg_exec_time = []
    for value in nodes_or_threads:
        # Get the rows that correspond to the current value (either Nodes or Threads)
        subset_value = subset[subset[group_by_col] == value]
        if not subset_value.empty:
            # Compute the average of execution times for runtime schedules
            avg_time = subset_value['Execution_Time'].mean()
            avg_exec_time.append(avg_time)
    return avg_exec_time

# Plot 1: Execution Time vs Number of Nodes (constant threads)
def plot_time_vs_nodes(threads):
    plt.figure(figsize=(10,6))
    for sched in ['static', 'dynamic', 'guided']:
        subset = data[(data['Threads'] == threads) & (data['Schedule_Type'] == sched)]
        if subset.empty:
            print(f"No data for {sched} with {threads} threads.")
            continue
        nodes = subset['Nodes'].values
        exec_time = subset['Execution_Time'].values
        plt.plot(nodes, exec_time, marker='o', label=sched)

    # Compute and plot the average runtime schedule
    subset_runtime = data[(data['Threads'] == threads) & (data['Schedule_Type'] == 'runtime')]
    if not subset_runtime.empty:
        nodes = sorted(subset_runtime['Nodes'].unique())
        avg_exec_time = compute_runtime_average(subset_runtime, nodes, 'Nodes')
        if avg_exec_time:
            plt.plot(nodes, avg_exec_time, marker='o', linestyle='--', color='purple', label='runtime (average)')

    plt.title(f'Execution Time vs Number of Nodes (Threads={threads})')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'exec_time_vs_nodes_threads_{threads}.png')
    plt.show()

# Plot 2: Execution Time vs Number of Threads (constant nodes)
def plot_time_vs_threads(nodes):
    plt.figure(figsize=(10,6))
    for sched in ['static', 'dynamic', 'guided']:
        subset = data[(data['Nodes'] == nodes) & (data['Schedule_Type'] == sched)]
        if subset.empty:
            print(f"No data for {sched} with {nodes} nodes.")
            continue
        threads = subset['Threads'].values
        exec_time = subset['Execution_Time'].values
        plt.plot(threads, exec_time, marker='o', label=sched)

    # Compute and plot the average runtime schedule
    subset_runtime = data[(data['Nodes'] == nodes) & (data['Schedule_Type'] == 'runtime')]
    if not subset_runtime.empty:
        threads = sorted(subset_runtime['Threads'].unique())
        avg_exec_time = compute_runtime_average(subset_runtime, threads, 'Threads')
        if avg_exec_time:
            plt.plot(threads, avg_exec_time, marker='o', linestyle='--', color='purple', label='runtime (average)')

    plt.title(f'Execution Time vs Number of Threads (Nodes={nodes})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'exec_time_vs_threads_nodes_{nodes}.png')
    plt.show()

# Generate plots
plot_time_vs_nodes(8)  # Example with 8 threads
plot_time_vs_threads(20)  # Example with 20 nodes
