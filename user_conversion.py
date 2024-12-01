#!/usr/bin/env python
# coding: utf-8

# 1 导入模块
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import joblib

warnings.filterwarnings('ignore')

# plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2 数据处理
# 2.1 读取数据
df = pd.read_csv('./data/user_conversion_prediction_dataset.csv')
print(df.head())

# 2.3 字段说明
# | 字段	         | 说明    |
# |:----------       |:-------- |
# | CustomerID       | 每个客户的唯一标识符 |
# | Age            | 客户的年龄 |
# | Gender          | 客户的性别（男性/女性） |
# | Income          | 客户的年收入，以美元计 |
# | CampaignChannel    | 营销活动传递的渠道：电子邮件(Email)、社交媒体(Social Media)、搜索引擎优化(SEO)、付费点击(PPC)、推荐(Referral)） |
# | CampaignType      | 营销活动的类型：意识(Awareness)、考虑(Consideration)、转化(Conversion)、留存(Retention) |
# | AdSpend         | 在营销活动上的花费，以美元计 |
# | ClickThroughRate   | 客户点击营销内容的比率 |
# | ConversionRate     | 点击转化为期望行为（如购买）的比率 |
# | WebsiteVisits     | 访问网站的总次数 |
# | PagesPerVisit     | 每次会话平均访问的页面数 |
# | TimeOnSite       | 每次访问平均在网站上花费的时间（分钟） |
# | SocialShares      | 营销内容在社交媒体上被分享的次数 |
# | EmailOpens       | 营销电子邮件被打开的次数 |
# | EmailClicks       | 营销电子邮件中链接被点击的次数 |
# | PreviousPurchases   | 客户之前进行的购买次数 |
# | LoyaltyPoints     | 客户累积的忠诚度积分数 |
# | AdvertisingPlatform | 广告平台（保密） |
# | AdvertisingTool    | 广告工具：保密 |
# | Conversion       | 目标变量：二元变量，表示客户是否转化（1）或未转化（0） |

# 2.4 删除重复值
df = df.drop_duplicates()


# 3 数据分析-特征分析
df1 = df.copy()
range_color = ['#045275', '#089099', '#7CCBA2', '#FCDE9C', '#F0746E', '#DC3977']
color1 = range_color[0]
color2 = range_color[-1]
size = 12

# 年龄分布
# 使用 KDE 曲线（核密度估计）直观地展示年龄数据的分布趋势。帮助了解用户的年龄分布特点，例如用户是否集中在某些年龄段。
def get_Age_analyze1():
    filename='fig/1-Age_analyze1.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Ages(Year)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['Age'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Ages Distribution',size=16)
    plt.savefig(filename, bbox_inches='tight')
get_Age_analyze1()

#在每个年龄段中，分别统计完成转化（Conversion == 1）和未完成转化（Conversion == 0）的人数。计算转化率，即完成转化人数占总人数的百分比。
def get_Age_analyze2():
    filename = 'fig/2-Age_analyze2.png'
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

    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(labels, y_data1, label='Number of People',color=color1, alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(labels, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color1)

    # 绘制plot
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
#客户年龄集中分布在10-70岁之间。
#不同年龄段的客户转化率波动不大，所以年龄对客户是否转化没有太大影响。
get_Age_analyze2()

# 分析不同营销渠道的用户分布及其转化率
def get_CampaignChannel_analyze():
    filename = 'fig/3-CampaignChannel_analyze.png'
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

    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[2], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[2])

    # 绘制plot
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
# 五个不同营销渠道的用户转化率波动在2%以内，所以营销渠道对客户是否转化没有太大影响。
get_CampaignChannel_analyze()

# 营销类型分布 感觉没什么用
def get_CampaignType_analyze():
    filename = 'fig/4-CampaignType_analyze.png'
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

    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[4], alpha=0.8)
    axis1.set_ylim(0, 3000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[4])

    # 绘制plot
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

# 不同营销类型的用户转化率波动最大接近10%，所以营销类型对客户是否转化有一定的影响。
# “转化”类型的转化率最高，为 93.36%。
get_CampaignType_analyze()

# 营销花费分布
#帮助了解广告支出的分布是否符合预期，例如是否有特定区间占比过高或过低。判断广告预算是否合理分布。
def get_AdSpend_analyze1():
    filename = 'fig/5-AdSpend_analyze1.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Marketing Spend (USD)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['AdSpend'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Marketing Spend', size=16)
    plt.savefig(filename, bbox_inches='tight')
#分布均匀
get_AdSpend_analyze1()

# 
def get_AdSpend_analyze2():
    filename = 'fig/6-AdSpend_analyze2.png'
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
    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=color1, alpha=0.8)
    axis1.set_ylim(0, 2000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=color1)

    # 绘制plot
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
# 不同营销花费的用户转化率波动比较明显，最大超过了10%，所以营销花费对客户是否转化有明显的影响。
get_AdSpend_analyze2()

# 网站点击率分布
def get_ClickThroughRate_analyze1():
    filename = 'fig/7-ClickThroughRate_analyze1.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Website Click-Through Rate (%)', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['ClickThroughRate'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Website Click-Through Rate',size=16)
    plt.savefig(filename, bbox_inches='tight')
get_ClickThroughRate_analyze1()

def get_ClickThroughRate_analyze2():
    filename = 'fig/7-ClickThroughRate_analyze2.png'
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
    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[1], alpha=0.8)
    axis1.set_ylim(0, 2000)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[1])

    # 绘制plot
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
# 不同点击率的用户转化率波动同样比较明显，最大超过了10%，所以点击率对客户是否转化也有明显的影响。
get_ClickThroughRate_analyze2()

# 访问网站总次数分布
def get_WebsiteVisits_analyze1():
    filename = 'fig/8-WebsiteVisits_analyze1.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Total Number of Website Visits', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['WebsiteVisits'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Distribution of Total Website Visits',size=16)
    plt.savefig(filename, bbox_inches='tight')
get_WebsiteVisits_analyze1()

def get_WebsiteVisits_analyze2():
    filename = 'fig/9-WebsiteVisits_analyze2.png'
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
    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[2], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[2])

    # 绘制plot
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
#不同客户访问网站总次数的用户转化率波动比较明显，最大超过了10%，所以客户访问网站总次数对客户是否转化有明显的影响。
get_WebsiteVisits_analyze2()

# 每次访问平均时间分布
def get_TimeOnSite_analyze1():
    filename = 'fig/10-TimeOnSite_analyze1.png'
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    axis1.set_xlabel('Average Time per Visit', color=color1, fontsize=size)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)
    sns.histplot(df['TimeOnSite'], kde=True, bins=10)
    axis1.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.title('Average Time per Visit Distribution',size=16)
    plt.savefig(filename, bbox_inches='tight')
get_TimeOnSite_analyze1()

def get_TimeOnSite_analyze2():
    filename = 'fig/11-TimeOnSite_analyze2.png'
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
    # 创建图像
    fig = plt.figure(figsize=(12, 6), dpi=80)
    axis1 = fig.add_axes((0.1, 0.1, 0.8, 0.8))# (left, bottom, width, height) 
    axis2 = axis1.twinx()

    # 绘制bar
    axis1.bar(x_data, y_data1, label='Number of People',color=range_color[4], alpha=0.8)
    axis1.set_ylim(0, 2500)
    axis1.set_ylabel('Number of People', color=color1, fontsize=size)
    axis1.tick_params('both', colors=color1, labelsize=size)

    for i, (xt, yt) in enumerate(zip(x_data, y_data1)):  
        axis1.text(xt, yt + 50, f'{yt:.2f}',size=size,ha='center', va='bottom', color=range_color[4])

    # 绘制plot
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
#不同客户每次访问网站时间的用户转化率波动比较明显，所以客户每次访问网站时间对客户是否转化有明显的影响。
get_TimeOnSite_analyze2()

# 4 模型分析
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 数据预处理
# Drop columns first
data = df.drop(columns=['AdvertisingPlatform', 'AdvertisingTool', 'CustomerID','Gender', 'CampaignChannel', 'CampaignType','Age'])


def get_model_analyze(): 
    # 绘制热力图
    corrdf = data.corr()
    plt.figure(figsize=(12, 12), dpi=80)
    sns.heatmap(corrdf, annot=True,cmap="rainbow", linewidths=0.05,square=True,annot_kws={"size":8}, cbar_kws={'shrink': 0.8})
    plt.title("Heatmap of Feature Correlations",size=16)
    plt.tight_layout()
    filename = 'fig/12-model_heatmap.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_analyze()

# 预测列
x_data = data.drop(columns=['Conversion'])
y = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=7, stratify=y)

# 4.1 KNN近邻算法模型
# 找到最高精度的k值
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
    filename = 'fig/13-best_k.png'
    plt.savefig(filename, bbox_inches='tight')
get_best_k()

k = 15
knn_model_best_k = KNeighborsClassifier(n_neighbors=best_k)
knn_model_best_k.fit(x_train, y_train)
train_accuracy = accuracy_score(y_train, knn_model_best_k.predict(x_train.values))
test_accuracy = accuracy_score(y_test, knn_model_best_k.predict(x_test.values))
print(f"KNN Train Accuracy: {train_accuracy}")
print(f"KNN Test Accuracy: {test_accuracy}")

# 评估
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
    filename = 'fig/14-model_cm_knn.png'
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
    filename = 'fig/15-model_roc_knn.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"KNN AUC: {auc}")
get_model_roc_knn()


# AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。
# 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。
# AUC = 0.5，跟随机猜测一样（例：抛硬币），模型没有预测价值。
# AUC < 0.5，比随机猜测还差。


# 4.2 逻辑回归
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
    filename = 'fig/16-model_cm_lr.png'
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
    filename = 'fig/17-model_roc_lr.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"Logistic Regression AUC: {auc_lr}")
get_model_roc_lr()

# 4.3 随机森林
x_data = data.drop(columns=['Conversion'])
y = data['Conversion']
x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.2, random_state=7, stratify=y)

random_forest_model = RandomForestClassifier(random_state=15)
random_forest_model.fit(x_train, y_train)

# 评估

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
    filename = 'fig/18-model_cm_rf.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_cm_rf()

# ROC曲线
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
    filename = 'fig/19-model_roc_rf.png'
    plt.savefig(filename, bbox_inches='tight')
    print(f"RF AUC: {auc}")
get_model_roc_rf()

# 模型性能稳定性
def get_model_roc_rf_stability():
    # 预测
    y_probs = random_forest_model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    # 计算原始ROC曲线的FPR, TPR, 和thresholds  
    fpr_orig, tpr_orig, thresholds_orig = roc_curve(y_test, y_probs)  
    print(f"RF Original FPR (first 5): {fpr_orig[:5]}")
    print(f"RF Original TPR (first 5): {tpr_orig[:5]}")
    print(f"RF Thresholds (first 5): {thresholds_orig[:5]}")

    n_bootstraps = 100 
    fpr_bootstraps = np.zeros((n_bootstraps, len(fpr_orig)))  
    tpr_bootstraps = np.zeros((n_bootstraps, len(tpr_orig)))  

    # 计算多个ROC曲线  
    for i in range(n_bootstraps):  
        x_resample, y_resample = resample(x_test, y_test)  
        y_probs_resample = random_forest_model.predict_proba(x_resample)[:, 1]  
        fpr_resample, tpr_resample, _ = roc_curve(y_resample, y_probs_resample)  
        # 线性插值
        fpr_interp = interp1d(np.linspace(0, 1, len(fpr_resample)), fpr_resample, fill_value="extrapolate")(np.linspace(0, 1, len(fpr_orig)))  
        tpr_interp = interp1d(np.linspace(0, 1, len(tpr_resample)), tpr_resample, fill_value="extrapolate")(np.linspace(0, 1, len(tpr_orig)))  
        fpr_bootstraps[i] = fpr_interp  
        tpr_bootstraps[i] = tpr_interp  

    # 计算置信区间  
    # fpr_mean = np.mean(fpr_bootstraps, axis=0)
    # fpr_ci = np.percentile(fpr_bootstraps, [2.5, 97.5], axis=0)
    # tpr_mean = np.mean(tpr_bootstraps, axis=0)
    tpr_ci = np.percentile(tpr_bootstraps, [2.5, 97.5], axis=0)  
    print(f"95% Confidence Interval (TPR): Lower = {tpr_ci[0][:5]}, Upper = {tpr_ci[1][:5]}")

    # 绘制ROC曲线和置信区间  
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
    filename = 'fig/20-model_roc_rf_stability.png'
    plt.savefig(filename, bbox_inches='tight')
get_model_roc_rf_stability()


# 特征重要性
def get_feature_importances():
    # 示例数据  
    feature_importances = random_forest_model.feature_importances_
    features_rf = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importances})
    features_rf.sort_values(by='Importance', inplace=True)

    x_data = features_rf['Feature'].tolist()
    y_data = features_rf['Importance'].tolist()
    # 创建条形图  
    fig = plt.figure(figsize=(12, 6), dpi=80)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.set_xlim(0, 0.1)
    ax.tick_params('both', colors=color1, labelsize=size)
    bars = ax.barh(x_data, y_data, color=range_color[1])

    # 在每个条形上方显示值  
    for bar in bars:  
        w = bar.get_width()
        ax.text(w+0.001, bar.get_y()+bar.get_height()/2, '%.4f'%w, ha='left', va='center')
        
    plt.xlabel('Importance', color=color1, fontsize=size)
    plt.ylabel('Feature', color=color1, fontsize=size)
    plt.title('Random Forest Feature Importance', size=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)  
    plt.tight_layout()
    filename = 'fig/21-feature_importances.png'
    plt.savefig(filename, bbox_inches='tight')
get_feature_importances()

# 保存模型
model_filename = "random_forest_model.pkl"
joblib.dump(random_forest_model, model_filename)
print(f"Model saved to {model_filename}")

# 测试加载模型
loaded_model = joblib.load(model_filename)
test_accuracy = accuracy_score(y_test, loaded_model.predict(x_test))
print(f"Test Accuracy of loaded model: {test_accuracy}")

print("test cloud build ci/cd")