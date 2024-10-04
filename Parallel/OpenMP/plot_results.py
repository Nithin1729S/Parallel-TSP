import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
data = pd.read_csv("tsp_dp_execution_times.csv")

# Plot 1: Execution Time vs Number of Nodes (constant threads)
def plot_time_vs_nodes(threads):
    plt.figure(figsize=(10,6))
    for sched in ['static', 'dynamic', 'guided']:
        subset = data[(data['Threads'] == threads) & (data['Schedule_Type'] == sched)]
        if subset.empty:
            print(f"No data for {sched} with {threads} threads.")
            continue
        # Convert to NumPy array to avoid multi-dimensional indexing issues
        nodes = subset['Nodes'].values
        exec_time = subset['Execution_Time'].values
        plt.plot(nodes, exec_time, marker='o', label=sched)

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
        # Convert to NumPy array to avoid multi-dimensional indexing issues
        threads = subset['Threads'].values
        exec_time = subset['Execution_Time'].values
        plt.plot(threads, exec_time, marker='o', label=sched)

    plt.title(f'Execution Time vs Number of Threads (Nodes={nodes})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'exec_time_vs_threads_nodes_{nodes}.png')
    plt.show()

# Generate plots
plot_time_vs_nodes(8)  # Example with 8 threads
plot_time_vs_threads(23)  # Example with 20 nodes
