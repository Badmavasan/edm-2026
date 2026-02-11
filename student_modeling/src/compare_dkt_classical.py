import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# 1. CONFIGURATION: Define your file paths here
platform_a_files = {
    'IRT': '../results/platform_a/irt/global/error_dependent/fold_metrics.csv',
    'BKT': '../results/platform_a/bkt/global/error_dependent/fold_metrics.csv',
    'PFA': '../results/platform_a/pfa/global/error_dependent/fold_metrics.csv',
    'DKT (EI)': '../results/platform_a/dkt/global/error_independent/fold_metrics.csv', # Error Independent
    'DKT (ED)': '../results/platform_a/dkt/global/error_dependent/fold_metrics.csv'    # Error Dependent
}

platform_b_files = {
    'IRT': '../results/platform_b/irt/global/error_dependent/fold_metrics.csv',
    'BKT': '../results/platform_b/bkt/global/error_dependent/fold_metrics.csv',
    'PFA': '../results/platform_b/pfa/global/error_dependent/fold_metrics.csv',
    'DKT (EI)': '../results/platform_b/dkt/global/error_independent/fold_metrics.csv', # Error Independent
    'DKT (ED)': '../results/platform_b/dkt/global/error_dependent/fold_metrics.csv'
}

# 2. CONFIGURATION: Define colors for each model
# You can use hex codes or standard color names
model_colors = {
    'IRT': '#2f4858',      # Muted Blue
    'BKT': '#33658a',      # Muted Green
    'PFA': '#f6ae2d',      # Muted Red
    'DKT (EI)': '#f26419', # Muted Purple
    'DKT (ED)': '#86c166'  # Muted Green
}

# Function to load and process files
def load_comparison_data(files_dict, platform_label):
    data_list = []
    for model_name, file_path in files_dict.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Calculate means of the metrics
            means = df[['auc', 'f1', 'precision', 'recall']].mean()
            for metric_name, val in means.items():
                data_list.append({
                    'Platform': platform_label,
                    'Model': model_name,
                    'Metric': metric_name.upper(),
                    'Value': val
                })
        else:
            print(f"Warning: File {file_path} not found.")
    return data_list

# Combine data from both platforms
all_data = load_comparison_data(platform_a_files, 'Platform A') + \
           load_comparison_data(platform_b_files, 'Platform B')

df_plot = pd.DataFrame(all_data)

# 3. PLOTTING
sns.set_theme(style="whitegrid", font_scale=1.5)

# Create a faceted bar plot
g = sns.catplot(
    data=df_plot,
    kind="bar",
    x="Platform",
    y="Value",
    hue="Model",
    col="Metric",
    col_wrap=2,
    palette=model_colors, # Using the custom colors defined above
    height=5,
    aspect=1.2,
    sharey=False,
    legend=False, # We will create a single legend manually
    edgecolor='black',
    linewidth=0.5
)

# Customizing the appearance of each subplot
for ax in g.axes.flat:
    ax.set_ylim(0, 1.05) # Metrics are between 0 and 1
    # Clean up titles (e.g., "Metric = AUC" becomes "AUC")
    ax.set_title(ax.get_title().split('=')[-1].strip(), fontweight='bold', fontsize=20)
    ax.set_xlabel('', fontsize=20, fontweight='bold')
    ax.set_ylabel('Score', fontsize=20, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# 4. SINGLE LEGEND: Create a high-quality unified legend at the bottom
handles = [mpatches.Patch(facecolor=color, edgecolor='black', linewidth=0.5, label=name)
           for name, color in model_colors.items()]
legend = g.fig.legend(
    handles=handles,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.01),
    frameon=True,
    title="Prediction Models",
    fontsize=18,
    title_fontsize=20,
)
legend.get_title().set_fontweight('bold')

# Adjust layout to make room for the legend
plt.subplots_adjust(bottom=0.15, hspace=0.3)

# Save as high-resolution PNG
plt.savefig('comparative_analysis_high_res.png', dpi=300, bbox_inches='tight')
plt.show()

print("Graph generated and saved as 'comparative_analysis_high_res.png'")