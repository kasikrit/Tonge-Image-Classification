#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:27:01 2024

@author: kasikritdamkliang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def add_labels(ax, bars, fontsize):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.2f}'.format(height),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 12),  # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='top',
        fontsize=fontsize)
        
     
#%%
# # Manually inputting the data
# data = {
#     'Model': ['VGG16', 'VGG16', 'VGG16', 'VGG19', 'VGG19', 'VGG19',
#               'EfficientNetB7', 'EfficientNetB7', 'EfficientNetB7',
#               'Xception', 'Xception', 'Xception',             
#               'InceptionResNetV2', 'InceptionResNetV2', 'InceptionResNetV2',
#               'InceptionV3', 'InceptionV3', 'InceptionV3',
#               'DenseNet121', 'DenseNet121', 'DenseNet121',              
#               'MobileNetV2', 'MobileNetV2', 'MobileNetV2',
#               'ViT_b16', 'ViT_b16', 'ViT_b16'],
#     'precision': [0.82, 0.77, 0.66, 0.82, 0.87, 0.65, 0.79, 0.86, 0.77,
#                   0.83, 0.84, 0.81, 0.83, 0.82, 0.74, 0.83, 0.81, 0.82,
#                   0.87, 0.86, 0.82, 0.84, 0.85, 0.76, 0.85, 0.80, 0.81],
#     'f1score': [0.82, 0.77, 0.65, 0.81, 0.86, 0.65, 0.78, 0.85, 0.75,
#                 0.83, 0.84, 0.79, 0.83, 0.82, 0.73, 0.83, 0.78, 0.80,
#                 0.87, 0.84, 0.77, 0.84, 0.81, 0.75, 0.84, 0.80, 0.76],
#     'accuracy': [0.89, 0.85, 0.78, 0.88, 0.91, 0.77, 0.86, 0.90, 0.84,
#                  0.89, 0.89, 0.86, 0.89, 0.88, 0.82, 0.89, 0.86, 0.86,
#                  0.92, 0.89, 0.84, 0.90, 0.88, 0.83, 0.90, 0.86, 0.84],
#     'sensitivity': [0.82, 0.77, 0.66, 0.81, 0.86, 0.65, 0.78, 0.85, 0.75,
#                     0.83, 0.84, 0.79, 0.83, 0.82, 0.73, 0.83, 0.79, 0.80,
#                     0.87, 0.84, 0.76, 0.84, 0.82, 0.75, 0.84, 0.80, 0.77],
#     'specificity': [0.92, 0.88, 0.83, 0.91, 0.93, 0.83, 0.90, 0.92, 0.88,
#                     0.92, 0.92, 0.89, 0.92, 0.91, 0.87, 0.92, 0.89, 0.90,
#                     0.94, 0.92, 0.88, 0.93, 0.91, 0.88, 0.93, 0.90, 0.88]
# }

# Convert the data dictionary into a pandas DataFrame
# df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
# df.to_csv('A1_model_performance.csv', index=False)


#%% A2 performances
# Based on the provided data, we will create a CSV file

# First, we'll define the data in a dictionary
data = {
    'Model': [
        'VGG16', 'VGG16', 'VGG16', 'VGG19', 'VGG19', 'VGG19',
        'EfficientNetB7', 'EfficientNetB7', 'EfficientNetB7', 'Xception', 'Xception', 'Xception',
        'InceptionResNetV2', 'InceptionResNetV2', 'InceptionResNetV2', 'InceptionV3', 'InceptionV3', 'InceptionV3',
        'DenseNet121', 'DenseNet121', 'DenseNet121', 'MobileNet', 'MobileNet', 'MobileNet',
        'MobileNetV2', 'MobileNetV2', 'MobileNetV2', 'ViT_b16', 'ViT_b16', 'ViT_b16'
    ],
    'precision': [
        0.79, 0.79, 0.73, 0.77, 0.68, 0.12, 0.66, 0.74, 0.47, 0.72, 0.85, 0.80,
        0.88, 0.77, 0.83, 0.82, 0.79, 0.83, 0.84, 0.77, 0.81, 0.80, 0.73, 0.83,
        0.83, 0.77, 0.83, 0.79, 0.73, 0.80
    ],
    'f1score': [
        0.79, 0.79, 0.73, 0.76, 0.68, 0.18, 0.66, 0.74, 0.43, 0.72, 0.84, 0.80,
        0.88, 0.77, 0.83, 0.82, 0.80, 0.83, 0.84, 0.77, 0.81, 0.80, 0.73, 0.83,
        0.83, 0.78, 0.83, 0.79, 0.73, 0.80
    ],
    'accuracy': [
        0.86, 0.86, 0.83, 0.84, 0.79, 0.57, 0.78, 0.83, 0.62, 0.82, 0.89, 0.87,
        0.92, 0.85, 0.89, 0.89, 0.86, 0.89, 0.90, 0.85, 0.88, 0.88, 0.82, 0.89,
        0.89, 0.85, 0.89, 0.87, 0.83, 0.87
    ],
    'sensitivity': [
        0.79, 0.79, 0.74, 0.76, 0.69, 0.33, 0.66, 0.74, 0.43, 0.72, 0.89, 0.80,
        0.88, 0.77, 0.83, 0.82, 0.80, 0.84, 0.84, 0.77, 0.81, 0.80, 0.74, 0.83,
        0.83, 0.78, 0.83, 0.79, 0.74, 0.81
    ],
    'specificity': [
        0.90, 0.90, 0.87, 0.89, 0.84, 0.67, 0.84, 0.74, 0.72, 0.86, 0.92, 0.91,
        0.95, 0.89, 0.92, 0.92, 0.90, 0.92, 0.93, 0.89, 0.91, 0.91, 0.87, 0.92,
        0.92, 0.89, 0.92, 0.91, 0.87, 0.81
    ]
}

# Convert the data dictionary into a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('A2_model_performance.csv', index=False)

#%%

csv_file_path = 'A1_model_performance.csv'
# csv_file_path = 'A2_model_performance.csv'
# Read the CSV file into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Now df holds your data and you can view the first few rows using df.head()
print(df.head())

#%% A3-D1.csv
# csv_file_path = 'A3-D1.csv'
csv_file_path = 'A3-D2.csv'
df = pd.read_csv(csv_file_path)
print(df.head())

#%%
# Plotting the bars with error bars using different shades of green
fig, ax = plt.subplots(figsize=(24, 12), dpi=300)
# fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
fontsize = 14
plt.rcParams.update({'font.size': fontsize})

# Let's continue the code to calculate mean and standard deviation and plot the bar graph with SD indicators

# First, we'll need to calculate the mean and standard deviation for each metric for each model.
# We will create a new DataFrame for this purpose.

# Prepare DataFrame for mean and standard deviation
mean_std_df = pd.DataFrame()

for metric in ['precision', 'f1score', 'accuracy', 'sensitivity', 'specificity']:
    mean_std_df[metric + '_mean'] = df.groupby('Model')[metric].mean()
    mean_std_df[metric + '_std'] = df.groupby('Model')[metric].std()

# Reset the index to get 'Model' as a column
mean_std_df.reset_index(inplace=True)

# Plotting the mean and standard deviation bars for each metric
bar_width = 0.18  # Width of the bars
n_metrics = 5  # Number of metrics to plot
index = np.arange(len(mean_std_df))  # Number of models
colors = ['#6AA84F', '#38761D', '#274E13', '#3C6',
          # '#6d9eeb',
          '#9ACD32'] 

# Create subplots for each metric
for i, metric in enumerate(['precision', 'f1score', 'accuracy', 'sensitivity', 'specificity']):
    # Calculate mean and standard deviation
    means = mean_std_df[metric + '_mean']
    stds = mean_std_df[metric + '_std']

    # Create bars for each metric
    bars = ax.bar(index + i * bar_width, means, bar_width, yerr=stds, capsize=5,
                  label=metric.capitalize(),
                   color=colors[i % len(colors)],
                  alpha=0.7)

    # Call the function to add labels on the bars
    add_labels(ax, bars, fontsize)

# Customize the plot
ax.set_xlabel('Model', fontsize=fontsize+2)
ax.set_ylabel('Metrics', fontsize=fontsize+2)
ax.set_title('Mean and SD for each model by metrics', fontsize=fontsize+2)
ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)  # Positioning x-ticks in the middle of the group of bars
ax.set_xticklabels(mean_std_df['Model'], rotation=45, ha='right', fontsize=fontsize+2)
ax.tick_params(axis='y', labelsize=fontsize+2)
ax.legend(title='Metrics',
          fontsize=fontsize+2,
          loc='lower right')

# Show plot
plt.tight_layout()
plt.show()
fig.savefig("A1-DSLR.png")
# fig.savefig("A3-D2.png")
# fig.savefig("A2-DSLR.png")

#%%
import pandas as pd
from scipy import stats

# Assuming 'df' is your DataFrame
# We will use ANOVA (Analysis of Variance) to check if there are significant differences

# Performing ANOVA for sensitivity
f_val, p_val = stats.f_oneway(df[df['Model'] == 'DenseNet121']['sensitivity'],
                              df[df['Model'] == 'VGG16']['sensitivity'],
                              df[df['Model'] == 'VGG19']['sensitivity'],
                              df[df['Model'] == 'EfficientNetB7']['sensitivity'],
                              df[df['Model'] == 'Xception']['sensitivity'],
                              df[df['Model'] == 'InceptionResNetV2G19']['sensitivity'],
                              df[df['Model'] == 'InceptionV3']['sensitivity'],
                              df[df['Model'] == 'MobileNetV2']['sensitivity'],
                              df[df['Model'] == 'ViT_b16']['sensitivity'],
                              )
print(f"Sensitivity ANOVA F-value: {f_val}, p-value: {p_val}")

# Performing ANOVA for specificity
f_val, p_val = stats.f_oneway(df[df['Model'] == 'DenseNet121']['specificity'],                           
                              df[df['Model'] == 'VGG16']['specificity'],
                              df[df['Model'] == 'VGG19']['specificity'],
                              df[df['Model'] == 'EfficientNetB7']['specificity'],
                              df[df['Model'] == 'Xception']['specificity'],
                              df[df['Model'] == 'InceptionResNetV2G19']['specificity'],
                              df[df['Model'] == 'InceptionV3']['specificity'],
                              df[df['Model'] == 'MobileNetV2']['specificity'],
                              df[df['Model'] == 'ViT_b16']['specificity'],
                              )
print(f"Specificity ANOVA F-value: {f_val}, p-value: {p_val}")



