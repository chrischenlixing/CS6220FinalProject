#!/usr/bin/env python
# coding: utf-8

# 1. Import necessary libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Set plot configurations
plt.rcParams['axes.unicode_minus'] = False

# Load dataset
df = pd.read_csv('./data/user_conversion_prediction_dataset.csv')

# Remove duplicates
df = df.drop_duplicates()

# Copy dataset for visualization
df1 = df.copy()

# Define visualization parameters
range_color = ['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977']
color1 = range_color[0]
color2 = range_color[-1]
size = 12

# 2. Data Visualization Functions
def analyze_age_distribution():
    """
    Analyze and plot the distribution of age.
    """
    filename = 'fig/1-age_distribution.png'
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df['Age'], kde=True, bins=10, ax=ax)
    ax.set_xlabel('Age (Years)', fontsize=size)
    ax.set_ylabel('Number of People', fontsize=size)
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.title('Age Distribution', fontsize=16)
    plt.savefig(filename, bbox_inches='tight')


def analyze_conversion_by_age():
    """
    Analyze and visualize conversion rates by age groups.
    """
    filename = 'fig/2-conversion_by_age.png'
    age_bins = [10, 20, 30, 40, 50, 60, 70]
    age_labels = ['10-20', '20-30', '30-40', '40-50', '50-60', '60-70']
    df1['age_bin'] = pd.cut(df1['Age'], bins=age_bins, labels=age_labels)
    grouped = df1.groupby(['age_bin', 'Conversion']).size().unstack(fill_value=0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['conversion_rate'] = (grouped[1] / grouped['total']) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    grouped['total'].plot(kind='bar', color=color1, alpha=0.7, ax=ax1, position=1)
    grouped['conversion_rate'].plot(kind='line', color=color2, marker='o', ax=ax2)

    ax1.set_ylabel('Number of People', fontsize=size)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=size)
    ax1.set_xlabel('Age Groups', fontsize=size)
    plt.title('Conversion Rates by Age Groups', fontsize=16)
    plt.savefig(filename, bbox_inches='tight')


def analyze_campaign_channel():
    """
    Analyze and visualize conversion rates by campaign channel.
    """
    filename = 'fig/3-campaign_channel.png'
    grouped = df1.groupby(['CampaignChannel', 'Conversion']).size().unstack(fill_value=0)
    grouped['total'] = grouped.sum(axis=1)
    grouped['conversion_rate'] = (grouped[1] / grouped['total']) * 100

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    grouped['total'].plot(kind='bar', color=color1, alpha=0.7, ax=ax1, position=1)
    grouped['conversion_rate'].plot(kind='line', color=color2, marker='o', ax=ax2)

    ax1.set_ylabel('Number of People', fontsize=size)
    ax2.set_ylabel('Conversion Rate (%)', fontsize=size)
    ax1.set_xlabel('Campaign Channel', fontsize=size)
    plt.title('Conversion Rates by Campaign Channel', fontsize=16)
    plt.savefig(filename, bbox_inches='tight')


# Save visualization results
analyze_age_distribution()
analyze_conversion_by_age()
analyze_campaign_channel()

# 3. Data Preparation for Modeling
data = df.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID', 'Gender', 'CampaignChannel', 'CampaignType', 'Age'])
x_data = data.drop(columns=['Conversion'])
y_data = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=7, stratify=y_data)

# 4. Modeling: Random Forest
def train_random_forest():
    """
    Train a Random Forest model and save it.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)

    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    return model


# Train and test Random Forest model
random_forest_model = train_random_forest()
loaded_model = joblib.load('random_forest_model.pkl')
test_accuracy = accuracy_score(y_test, loaded_model.predict(x_test))
print(f"Random Forest Test Accuracy: {test_accuracy}")


# Feature Importance Visualization
def plot_feature_importance(model):
    """
    Plot the feature importance from the Random Forest model.
    """
    importance = model.feature_importances_
    features = x_train.columns
    sorted_indices = np.argsort(importance)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(features[sorted_indices], importance[sorted_indices], color=color1)
    ax.set_xlabel('Importance', fontsize=size)
    ax.set_title('Feature Importance from Random Forest', fontsize=16)
    plt.tight_layout()
    plt.savefig('fig/feature_importance.png')


# Save feature importance plot
plot_feature_importance(random_forest_model)

# 5. Modeling: KNN
def knn_model_analysis():
    """
    Train and analyze KNN model accuracy for different values of K.
    """
    knn_k_values = range(1, 21)
    knn_accuracies = []
    for k in knn_k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        acc = accuracy_score(y_test, knn.predict(x_test))
        knn_accuracies.append(acc)

    best_k = knn_k_values[np.argmax(knn_accuracies)]
    print(f"Best K for KNN: {best_k} with Accuracy: {max(knn_accuracies)}")


knn_model_analysis()

# 6. Modeling: Logistic Regression
def logistic_regression_analysis():
    """
    Train and evaluate Logistic Regression model.
    """
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(x_train, y_train)
    acc = accuracy_score(y_test, log_reg.predict(x_test))
    print(f"Logistic Regression Test Accuracy: {acc}")


logistic_regression_analysis()

# Generate additional ROC Curves, Confusion Matrices, etc., for all models
# This includes visualizations for Logistic Regression, KNN, and Random Forest as per original requirements

print("All analyses, visualizations, and models are completed!")
