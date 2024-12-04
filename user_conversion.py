#!/usr/bin/env python
# coding: utf-8

# 1 Import necessary modules
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False

# 2 Data Processing
# 2.1 Read data
df = pd.read_csv('./data/user_conversion_prediction_dataset.csv')
print(df.head())


# 2.2 Field Description
# | Field                | Description                                         |
# |:---------------------|:--------------------------------------------------- |
# | CustomerID           | Unique identifier for each customer                 |
# | Age                  | Age of the customer                                 |
# | Gender               | Gender of the customer (Male/Female)                |
# | Income               | Annual income of the customer in USD                |
# | CampaignChannel      | Channel of marketing (Email, Social Media, SEO, PPC, Referral) |
# | CampaignType         | Type of marketing campaign (Awareness, Consideration, Conversion, Retention) |
# | AdSpend              | Marketing spending in USD                           |
# | ClickThroughRate     | Click rate on marketing content                     |
# | ConversionRate       | Rate of clicks converted into desired actions       |
# | WebsiteVisits        | Total number of website visits                      |
# | PagesPerVisit        | Average number of pages visited per session         |
# | TimeOnSite           | Average time spent per visit in minutes             |
# | SocialShares         | Number of shares on social media                    |
# | EmailOpens           | Number of times marketing emails were opened        |
# | EmailClicks          | Number of clicks on links in marketing emails       |
# | PreviousPurchases    | Number of previous purchases                        |
# | LoyaltyPoints        | Accumulated loyalty points of the customer          |
# | AdvertisingPlatform  | Advertising platform (Confidential)                 |
# | AdvertisingTool      | Advertising tool (Confidential)                     |
# | Conversion           | Target variable: Binary variable indicating whether the customer converted (1) or not (0) |


# 2.3 Remove duplicate entries
df = df.drop_duplicates()


# 3 Feature Analysis
df1 = df.copy()
range_color = ['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977']
color1 = range_color[0]
color2 = range_color[-1]
size = 12

# Analyze age distribution using KDE curve
# Visualizes the age distribution to understand user demographics.
def age():
    filename='fig/1-age.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Ages(Year)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['Age'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Age Distribution',size=16)
    plt.savefig(filename, bbox_inches='tight')
age()

# Analyze conversion rates for different age groups
# Calculates conversion rates for each age range.
def conversion_by_age():
    filename = 'fig/2-conversion_by_age.png'
    labels=['10-20','20-30','30-40','40-50','50-60','60-70']
    df1['age_bin'] = pd.cut(df1['Age'],bins=[10,20,30,40,50,60,70],labels=labels)
    grouped = df1.groupby(['age_bin', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='age_bin', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(labels, y_data1, label='Number of People',color=color1, alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(labels, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color1)

    axis2.plot(labels, y_data2,label='Converstion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Converstion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(80, 90)
    for i, (xt, yt) in enumerate(zip(labels, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.88, 0.92))
    axis2.legend(loc=(0.88, 0.87))
    plt.gca().spines["left"].set_color(range_color[0])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Conversion Rate Distribution based on Ages",size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# Users are concentrated in the 10-70 age range.
# Conversion rates across different age groups show minor fluctuations,
# indicating minimal influence of age on conversion rates.
conversion_by_age()

# Analyze user distribution and conversion rates across marketing channels
def campaign_channel():
    filename = 'fig/3-campaign_channel.png'
    CampaignChannel_dict = {
        'Email': 'Email',
        'PPC': 'Pay-Per-Click',
        'Referral': 'Referral',
        'SEO': 'Search Engine Optimization',
        'Social Media': 'Social Media'
    }
    grouped = df1.groupby(['CampaignChannel', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='CampaignChannel', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    merged['CampaignChannel'] = merged['CampaignChannel'].replace(CampaignChannel_dict)
    x_data = merged['CampaignChannel'].values.tolist()
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[2], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[2])

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=color2, marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate (%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(80, 90)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.88, 0.92))
    axis2.legend(loc=(0.88, 0.87))
    plt.gca().spines["left"].set_color(range_color[2])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of People and Conversion Rates Across Channels",size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
# Conversion rates across five marketing channels fluctuate within 2%.
# Marketing channels have minimal impact on whether customers convert.
campaign_channel()

# Distribution of marketing types
def campaign_type():
    filename = 'fig/4-campaign_type.png'
    CampaignType_dic = {
        'Awareness': 'Awareness',
        'Consideration': 'Consideration',
        'Conversion': 'Conversion',
        'Retention': 'Retention'
    }
    grouped = df1.groupby(['CampaignType', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='CampaignType', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    merged['CampaignType'] = merged['CampaignType'].replace(CampaignType_dic)

    x_data = merged['CampaignType'].values.tolist()
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[4], alpha=0.8)
    axis1.set_ylim(0, 3000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[4])

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(80, 95)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.88, 0.92))
    axis2.legend(loc=(0.88, 0.87))
    plt.gca().spines["left"].set_color(range_color[4])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of Number of People and Conversion Rates Across Marketing Types", size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')

# Conversion rates for different marketing types vary up to 10%.
# The "Conversion" type has the highest conversion rate at 93.36%.
campaign_type()

# Distribution of marketing spending
# Helps understand whether ad spending is distributed as expected, identifying imbalances.
def ad_spend():
    filename = 'fig/5-ad_spend.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Marketing Spend (USD)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['AdSpend'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Marketing Spend', size=16)
    plt.savefig(filename, bbox_inches='tight')
# Spending is evenly distributed.
ad_spend()

# Distribution of marketing spending and conversion rates
def ad_spend_conversion():
    filename = 'fig/6-ad_spend_conversion.png'
    labels=['0-2000','2000-3000','3000-4000','4000-5000','5000-6000','6000-7000','7000-8000','8000-1000']
    df1['AdSpend_bin'] = pd.cut(df1['AdSpend'],bins=[0,2000,3000,4000,5000,6000,7000,8000,10000],labels=labels)
    grouped = df1.groupby(['AdSpend_bin', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='AdSpend_bin', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    x_data = labels
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=color1, alpha=0.8)
    axis1.set_ylim(0, 2000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color1)

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(80, 95)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.05, 0.92))
    axis2.legend(loc=(0.05, 0.87))
    plt.gca().spines["left"].set_color(range_color[0])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of Number of People and Conversion Rates Across Marketing Spend Ranges", size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
# Fluctuations in conversion rates across different marketing spending ranges are significant, exceeding 10%. 
# Thus, marketing spending has a notable impact on whether customers convert.
ad_spend_conversion()

# Website Click-Through Rate Distribution
def click_through_rate():
    filename = 'fig/7-click_through_rate.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Website Click-Through Rate (%)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['ClickThroughRate'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Website Click-Through Rate',size=16)
    plt.savefig(filename, bbox_inches='tight')
click_through_rate()

def click_through_rate_conversion():
    filename = 'fig/8-click_through_rate_conversion.png'
    labels=['0-0.05','0.05-0.1','0.1-0.15','0.15-0.2','0.2-0.25','0.25-0.3']
    df1['ClickThroughRate_bin'] = pd.cut(df1['ClickThroughRate'],bins=[0,0.05,0.1,0.15,0.2,0.25,0.3],labels=labels)
    grouped = df1.groupby(['ClickThroughRate_bin', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='ClickThroughRate_bin', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    x_data = labels
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[1], alpha=0.8)
    axis1.set_ylim(0, 2000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[1])

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(70, 95)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.05, 0.92))
    axis2.legend(loc=(0.05, 0.87))
    plt.gca().spines["left"].set_color(range_color[1])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of Number of People and Conversion Rates Across Click-Through Rate Intervals", size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
# Conversion rates fluctuate significantly across different click-through rate intervals, exceeding 10%.
# Click-through rates have a clear impact on whether customers convert.
click_through_rate_conversion()

# Distribution of total website visits
def website_visits():
    filename = 'fig/9-website_visits.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Total Number of Website Visits', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['WebsiteVisits'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Total Website Visits',size=16)
    plt.savefig(filename, bbox_inches='tight')
website_visits()

def website_visits_conversion():
    filename = 'fig/10-website_visits_conversion.png'
    labels=['0-10','10-20','20-30','30-40','40-50']
    df1['WebsiteVisits_bin'] = pd.cut(df1['WebsiteVisits'],bins=[0,10,20,30,40,50],labels=labels)
    grouped = df1.groupby(['WebsiteVisits_bin', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='WebsiteVisits_bin', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    x_data = labels
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[2], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[2])

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(70, 95)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.05, 0.92))
    axis2.legend(loc=(0.05, 0.87))
    plt.gca().spines["left"].set_color(range_color[2])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of Total Website Visits and Conversion Rates by Customers", size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    fig.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
# Conversion rates fluctuate significantly with total website visits.
website_visits_conversion()

# Average time on site each visit.
def time_on_site():
    filename = 'fig/11-time_on_site.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Average Time per Visit', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['TimeOnSite'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Average Time per Visit Distribution',size=16)
    plt.savefig(filename, bbox_inches='tight')
time_on_site()

def time_on_site_conversion():
    filename = 'fig/12-time_on_site_conversion.png'
    labels=['0-3','3-6','6-9','9-12','12-15']
    df1['TimeOnSite_bin'] = pd.cut(df1['TimeOnSite'],bins=[0,3,6,9,12,15],labels=labels)
    grouped = df1.groupby(['TimeOnSite_bin', 'Conversion']).size().reset_index(name='counts')
    converted = grouped[grouped['Conversion'] == 1]  
    not_converted = grouped[grouped['Conversion'] == 0]
    merged = converted.merge(not_converted, on='TimeOnSite_bin', suffixes=('_1', '_0'))   
    merged['all'] = merged['counts_1'] + merged['counts_0']
    merged['ratio'] = (merged['counts_1'] / merged['all'])*100
    merged = merged.round(2)
    x_data = labels
    y_data1 = merged['all'].values.tolist()
    y_data2 = merged['ratio'].values.tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[4], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[4])

    axis2.plot(x_data, y_data2,label='Conversion Rate', color=range_color[-1], marker="o", linewidth=2)
    axis2.set_ylabel('Conversion Rate(%)', color=color2,fontsize=size)
    axis2.tick_params('y', colors=color2, labelsize=size)
    axis2.set_ylim(70, 95)
    for i, (xt, yt) in enumerate(zip(x_data, y_data2)):  
        axis2.text(xt, yt + 0.3, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color2) 

    axis1.legend(loc=(0.05, 0.92))
    axis2.legend(loc=(0.05, 0.87))
    plt.gca().spines["left"].set_color(range_color[4])
    plt.gca().spines["right"].set_color(range_color[-1]) 
    plt.gca().spines["left"].set_linewidth(2)
    plt.gca().spines["right"].set_linewidth(2) 

    plt.title("Distribution of Time Spent per Website Visit and Conversion Rates by Customers",size=16)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.tight_layout()  
    plt.savefig(filename, bbox_inches='tight')
# Significant fluctuations in conversion rates based on time spent on the site highlight its importance in conversion analysis.
time_on_site_conversion()

# --- Model Analysis ---
# Data Preprocessing
data = df.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID','Gender', 'CampaignChannel', 'CampaignType','Age'])

 # Visualize correlations between features using a heatmap
def model_heatmap(): 
    corrdf = data.corr()
    plt.figure(figsize=(12, 12), dpi=80)
    sns.heatmap(corrdf, annot=True,cmap="rainbow", linewidths=0.05,square=True,annot_kws={"size":8}, cbar_kws={'shrink': 0.8})
    plt.title("Heatmap of Feature Correlations",size=16)
    plt.tight_layout()
    filename = 'fig/13-model_heatmap.png'
    plt.savefig(filename, bbox_inches='tight')
model_heatmap()

# Drop columns first
data = df.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID','Gender', 'CampaignChannel', 'CampaignType','Age'])

x_data = data.drop(columns=['Conversion'])
y = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=7, stratify=y)

# KNN Model Analysis
# Determine the optimal value of K for the KNN model
knn_k_values = range(1, 21)
knn_accuracies = []
for k in knn_k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(x_train, y_train)
    y_pred = knn_model.predict(x_test.values)
    accuracy = accuracy_score(y_test, y_pred)
    knn_accuracies.append(accuracy)
best_k = knn_k_values[np.argmax(knn_accuracies)]
best_accuracy = max(knn_accuracies)
print(f"Best value of K: {best_k}")
print(f"Corresponding accuracy: {best_accuracy}")

def get_best_k():
    plt.figure(figsize=(10, 6))
    plt.plot(knn_k_values, knn_accuracies, marker='o', linestyle='-')
    plt.title('KNN accuracy for different values of K')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')
    plt.xticks(knn_k_values)
    plt.grid(False)
    filename = 'fig/14-best_k.png'
    plt.savefig(filename, bbox_inches='tight')
get_best_k()

k = 15
knn_model_best_k = KNeighborsClassifier(n_neighbors=best_k)
knn_model_best_k.fit(x_train, y_train)
train_accuracy = accuracy_score(y_train, knn_model_best_k.predict(x_train.values))
test_accuracy = accuracy_score(y_test, knn_model_best_k.predict(x_test.values))
print(f"KNN Train Accuracy: {train_accuracy}")
print(f"KNN Test Accuracy: {test_accuracy}")

y_pred = knn_model_best_k.predict(x_test.values)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"KNN Confusion Matrix: \n{conf_matrix}")

def get_model_knn():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6), dpi=80)
    sns.heatmap(cm, annot=True, cmap='rainbow', fmt='d', square=True,annot_kws={"size":12},cbar_kws={'shrink': 0.8})    
    plt.xlabel('Predicted Values')  
    plt.ylabel('Actual Values')  
    plt.title("Confusion Matrix", size=16)
    plt.tight_layout()
    filename = 'fig/15-model_cm_knn.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_knn()


def get_model_roc_knn():
    y_probs = knn_model_best_k.predict_proba(x_test.values)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.tick_params('both', colors=color1, labelsize=size)
    axis1.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    axis1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color=color1, fontsize=size)
    plt.ylabel('Recall Rate', color=color1, fontsize=size)
    plt.title(f'ROC Curve - {type(knn_model_best_k).__name__}', size=16)
    plt.legend(loc="lower right",fontsize=size)
    plt.tight_layout()
    filename = 'fig/16-model_roc_knn.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"KNN AUC: {auc}")
get_model_roc_knn()


# Evaluate AUC values and their significance
# AUC = 1: Perfect classifier. A model that achieves this score has perfect predictive capability.
# 0.5 < AUC < 1: Better than random guessing. Indicates predictive value with proper threshold tuning.
# AUC = 0.5: Equivalent to random guessing (e.g., flipping a coin); the model lacks predictive value.
# AUC < 0.5: Worse than random guessing; the model's predictions are misleading.


# Logistic Regression Model
# Train Logistic Regression model
x_data = data.drop(columns=['Conversion'])
y = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=7, stratify=y)

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(x_train, y_train)

# Predictions
y_pred_lr = log_reg_model.predict(x_test)
y_prob_lr = log_reg_model.predict_proba(x_test)[:, 1]

# Evaluate Logistic Regression
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(f"Logistic Regression Train Accuracy: {accuracy_score(y_train, log_reg_model.predict(x_train))}")
print(f"Logistic Regression Test Accuracy: {accuracy_lr}")
print(f"Logistic Regression Confusion Matrix: \n{conf_matrix_lr}")

# Confusion Matrix Heatmap
def get_model_cm_lr():
    cm = confusion_matrix(y_test, y_pred_lr)
    plt.figure(figsize=(6, 6), dpi=80)
    sns.heatmap(cm, annot=True, cmap='rainbow', fmt='d', square=True, annot_kws={"size":12}, cbar_kws={'shrink': 0.8})
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values')
    plt.title("Confusion Matrix - Logistic Regression", size=16)
    plt.tight_layout()
    filename = 'fig/17-model_cm_lr.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_cm_lr()

# ROC Curve
def get_model_roc_lr():
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)

    plt.figure(figsize=(12, 6), dpi=80)
    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'AUC = {auc_lr:.2f}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - Logistic Regression')
    plt.legend(loc="lower right")
    plt.tight_layout()
    filename = 'fig/18-model_roc_lr.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Logistic Regression AUC: {auc_lr}")
get_model_roc_lr()

# Random Forest Model
# Train and evaluate a Random Forest model for classification
x_data = data.drop(columns=['Conversion'])
y = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=7, stratify=y)

random_forest_model = RandomForestClassifier(random_state=15)
random_forest_model.fit(x_train, y_train)

y_pred = random_forest_model.predict(x_test)
train_accuracy = accuracy_score(y_train, random_forest_model.predict(x_train))
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Random Forest Train Accuracy: {train_accuracy}")
print(f"Random Forest Test Accuracy: {test_accuracy}")
print(f"Random Forest Confusion Matrix: \n{conf_matrix}")

def get_model_cm_rf():
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6), dpi=80)
    sns.heatmap(cm, annot=True, cmap='rainbow', fmt='d', square=True,annot_kws={"size":12},cbar_kws={'shrink': 0.8})    
    plt.xlabel('Predicted Values')  
    plt.ylabel('True Values')  
    plt.title("Confusion Matrix", size=16)
    plt.tight_layout()
    filename = 'fig/19-model_cm_rf.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_cm_rf()

# ROC curve
def get_model_roc_rf():
    y_probs = random_forest_model.predict_proba(x_test.values)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)

    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.tick_params('both', colors=color1, labelsize=size)
    axis1.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc:.2f}')
    axis1.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', color=color1, fontsize=size)
    plt.ylabel('Recall Rate', color=color1, fontsize=size)
    plt.title(f'ROC Curve - {type(random_forest_model).__name__}', size=16)
    plt.legend(loc="lower right",fontsize=size)
    plt.tight_layout()
    filename = 'fig/20-model_roc_rf.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"RF AUC: {auc}")
get_model_roc_rf()

# Stability Analysis for Random Forest Model
# Evaluate model stability using bootstrap resampling
def get_model_roc_rf_stability():
    y_probs = random_forest_model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    # Calculate the FPR, TPR, and thresholds of the original ROC curve
    fpr_orig, tpr_orig, thresholds_orig = roc_curve(y_test, y_probs)  
    print(f"RF Original FPR (first 5): {fpr_orig[:5]}")
    print(f"RF Original TPR (first 5): {tpr_orig[:5]}")
    print(f"RF Thresholds (first 5): {thresholds_orig[:5]}")

    n_bootstraps = 100 
    fpr_bootstraps = np.zeros((n_bootstraps, len(fpr_orig)))  
    tpr_bootstraps = np.zeros((n_bootstraps, len(tpr_orig)))  

    # Calculate multiple ROC curves
    for i in range(n_bootstraps):  
        x_resample, y_resample = resample(x_test, y_test)  
        y_probs_resample = random_forest_model.predict_proba(x_resample)[:, 1]  
        fpr_resample, tpr_resample, _ = roc_curve(y_resample, y_probs_resample)  
        
        fpr_interp = interp1d(np.linspace(0, 1, len(fpr_resample)), fpr_resample, fill_value="extrapolate")(np.linspace(0, 1, len(fpr_orig)))  
        tpr_interp = interp1d(np.linspace(0, 1, len(tpr_resample)), tpr_resample, fill_value="extrapolate")(np.linspace(0, 1, len(tpr_orig)))  
        fpr_bootstraps[i] = fpr_interp  
        tpr_bootstraps[i] = tpr_interp  

    # Calculate the 95% confidence interval for the TPR
    tpr_ci = np.percentile(tpr_bootstraps, [2.5, 97.5], axis=0)  
    print(f"95% Confidence Interval (TPR): Lower = {tpr_ci[0][:5]}, Upper = {tpr_ci[1][:5]}")

    # Plot the ROC curve and confidence interval
    plt.figure(figsize=(12, 6), dpi=80)  
    plt.plot(fpr_orig, tpr_orig, color='blue', lw=2, label=f'AUC = {auc:.2f}')  
    plt.fill_between(fpr_orig, tpr_ci[0], tpr_ci[1], color='blue', alpha=0.2, label='95% Confidence Interval')  
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # 随机猜测线  
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])  
    plt.xlabel('False Positive Rate', color=color1, fontsize=size)
    plt.ylabel('Recall Rate', color=color1, fontsize=size)
    plt.title('ROC Curve - 95% Confidence Interval')
    plt.legend(loc="lower right",fontsize=size)  
    plt.grid(False)  
    plt.tight_layout()
    filename = 'fig/21-model_roc_rf_stability.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_roc_rf_stability()

# Feature Importance
def get_feature_importances():
    feature_importances = random_forest_model.feature_importances_
    features_rf = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})
    features_rf.sort_values(by='Importance', inplace=True)

    x_data = features_rf['Feature'].tolist()
    y_data = features_rf['Importance'].tolist()

    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xlim(0, 0.1)
    ax.tick_params('both', colors=color1, labelsize=size)
    bars = ax.barh(x_data, y_data, color=range_color[1])

    for bar in bars:  
        w = bar.get_width()
        ax.text(w+0.001, bar.get_y()+bar.get_height()/2, '%.4f'%w, ha='left', va='center')
        
    plt.xlabel('Importance', color=color1, fontsize=size)
    plt.ylabel('Feature', color=color1, fontsize=size)
    plt.title('Random Forest Feature Importance', size=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.tight_layout()
    filename = 'fig/22-feature_importances.png'
    plt.savefig(filename, bbox_inches='tight')
get_feature_importances()

# Save the trained Random Forest model  
model_filename = "random_forest_model.pkl"
def save_rf_model():
    joblib.dump(random_forest_model, model_filename)
    print(f"Model saved to {model_filename}")
save_rf_model()

# Test loading and evaluating the saved model  
def load_and_test_saved_model():
    loaded_model = joblib.load("random_forest_model.pkl")
    test_accuracy = accuracy_score(y_test, loaded_model.predict(x_test))
    print(f"Test Accuracy of loaded model: {test_accuracy}")
load_and_test_saved_model()