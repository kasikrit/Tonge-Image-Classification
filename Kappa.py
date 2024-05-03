from imutils import paths
import configs
config = configs.get_config()
import numpy as np
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score

print(config['DATA']['TRAIN_PATH'])

trainPaths = list(paths.list_images(config['DATA']['TRAIN_PATH']))   
totalTrain = len(trainPaths)
print(totalTrain)
trainPaths.sort()

features = pd.read_csv(config['DATA']['KAPPA_PATH'])
print(features.shape)
print(features.head())


#%%
# Adding some NaN values to the 'Class-KK' and 'Class-PC' columns for demonstration
features_with_nan = features.copy()
features_with_nan.loc[1, 'Class-KK'] = np.nan
features_with_nan.loc[3, 'Class-PC'] = np.nan

# Remove rows with NaN values in either 'Class-KK' or 'Class-PC' columns
features_no_nan = features_with_nan.dropna(subset=['Class-KK', 'Class-PC'])

labels = features_no_nan[['Class-KK', 'Class-PC']].values

# Pairwise Cohen's Kappa for demonstration
cohen_kappa_12 = cohen_kappa_score(labels[:, 0], labels[:, 1])
# cohen_kappa_13 = cohen_kappa_score(labels[:, 0], labels[:, 2])
# cohen_kappa_23 = cohen_kappa_score(labels[:, 1], labels[:, 2])

print(cohen_kappa_12) #0.11508987634035439


#%%
features = features.dropna(subset=['Class'])

labels = features[['KK', 'PJ', 'TS']].to_numpy().astype('uint8')

labels = labels - 1

#%%
# Convert the labels into a format suitable for Fleiss' Kappa calculation:
# Count of raters for each category per item. In this simplified example, each image can belong to one of three classes.
# For Fleiss' kappa, we need to convert this to a count matrix where each row represents an image,
# and each column represents the count of raters who assigned the image to each class.

# Initialize a count matrix with zeros: rows = number of images, columns = number of classes
class_counts = np.zeros((labels.shape[0], 3))  # 3 classes: vata, pitta, kapha

# Populate the count matrix
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        class_counts[i, labels[i, j]] += 1

# Calculate Fleiss' Kappa
fleiss_kappa_score = fleiss_kappa(class_counts)

print(fleiss_kappa_score) #0.29735550041309106
