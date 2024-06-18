import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Data
strategy = ['Vanilla', 'Axis', 'Rubric', 'Axis+Rubric']
tasks = ['LF', 'F', 'IF', 'R']

# Errors data (replace with actual data)
errors = {
    'Vanilla': np.array([[225, 36, 528-225-36], [220, 21, 483-220-21], [161, 36, 379-161-36], [360, 47, 494-360-47]]),  # [X, X_j, X_u]
    'Axis': np.array([[90, 28, 528-90-28], [127, 111, 483-127-111], [95, 42, 379-95-42], [308, 38, 494-308-38]]),
    'Rubric': np.array([[79, 32, 528-79-32], [131, 31, 483-131-31], [74, 30, 379-74-30], [332, 39, 494-332-39]]),
    'Axis+Rubric': np.array([[74, 42, 528-74-42], [114, 78, 483-114-78], [86, 35, 379-86-35], [310, 49, 494-310-49]])
}

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.15  # Thinner bars
index = np.arange(len(strategy))

# Define colors for each task
colors = ['#9fc5e8', '#b6d7a8', '#f9cb9c', '#d5a6bd']

# Iterate over tasks and plot bars for each strategy
for i, task in enumerate(tasks):
    for j, strat in enumerate(strategy):
        X = errors[strat][i][0]
        X_j = errors[strat][i][1]
        X_u = errors[strat][i][2]

        ax.bar(index[j] + i * bar_width, X, width=bar_width, edgecolor='black', color=colors[i])
        ax.bar(index[j] + i * bar_width, X_j, width=bar_width, bottom=X, edgecolor='black', color=colors[i], hatch='*')
        ax.bar(index[j] + i * bar_width, X_u, width=bar_width, bottom=X + X_j, edgecolor='black', color='lightgray')

# Configure the plot
ax.set_ylabel('# Errors', fontsize=30)
ax.set_xlabel('Evaluation Strategy', fontsize=30)
ax.set_xticks(index + bar_width*1.5)
ax.set_xticklabels(strategy, fontsize=25)
ax.set_ylim(0, 600)  # Increase the y-axis range

# Custom legend
legend_elements = [
    Patch(facecolor=colors[0], edgecolor='black', label='LF'),
    Patch(facecolor=colors[1], edgecolor='black', label='F'),
    Patch(facecolor=colors[2], edgecolor='black', label='IF'),
    Patch(facecolor=colors[3], edgecolor='black', label='R'),
    Patch(facecolor='lightgray', edgecolor='black', label='Undetected'),
    Patch(facecolor='white', edgecolor='black', label='Justification', hatch='*')
]
ax.legend(handles=legend_elements, fontsize=17, ncol=6)  # Increase legend font size
plt.grid(True, axis='y', linestyle='-')

# Increase the font size of the ticks
ax.tick_params(axis='both', which='major', labelsize=25)

plt.tight_layout()
plt.savefig('justification_errors.pdf', dpi=300)
# plt.show()
