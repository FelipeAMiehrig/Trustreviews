
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats as st
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")

def identify_trends(data, consecutive_n = 3) -> Tuple[List[int], List[int]]:
    """identifying which indexes of the sequence are a consecutive up or down trend

    Parameters
    ----------
    data : 
        array like containing the historical data 
        
    consecutive_n : 
         (Default value = 3)
        the number of consecutive up or down steps in order to be considered a sequence
    Returns 
    -------
    tuple of list of interges
            contains the indices of the up and down trend elements of the sequence for colouring
    """
    upward_trend = []
    downward_trend = []
    trend_length = 0
    last_direction = None

    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            if last_direction == 'up':
                trend_length += 1
            else:
                trend_length = 1
            last_direction = 'up'
        elif data[i] < data[i - 1]:
            if last_direction == 'down':
                trend_length += 1
            else:
                trend_length = 1
            last_direction = 'down'
        else:
            trend_length = 0
            last_direction = None
        n_to_go = consecutive_n
        if trend_length >= consecutive_n:
            if last_direction == 'up':
                upward_trend.append(i)
            elif last_direction == 'down':
                downward_trend.append(i)

    return upward_trend, downward_trend


def plot_language_distribution(df: pd.DataFrame, target_month: Optional[int] = None, target_year: Optional[int] = None, 
                                start_date: Optional[str] = None, end_date: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """ bar plots showing the review count for each language 

    Parameters
    ----------
    df: pd.DataFrame :
        dataframe containing the whole review dataset 

    target_month: Optional[int] :
         (Default value = None)
        the keywords plot will only contain words from reviews given on that month

    target_year: Optional[int] :
         (Default value = None)
         the keywords plot will only contain words from reviews given on that year

    start_date: Optional[str] :
         (Default value = None)
        the plot will only contain words from reviews given on the specified time window (MM/DD/YYYY)

    end_date: Optional[str] :
         (Default value = None)
         the plot will only contain words from reviews given on the specified time window (MM/DD/YYYY)
        
    output_file: Optional[str] :
         (Default value = None)
         outputfile path in case user wants to save the figure

    Returns
    -------

    """
    if target_month is not None:
        df = df[(df['created_date'].dt.month == target_month) & (df['created_date'].dt.year == target_year)]
    elif start_date is not None:
        df = df[(df['created_date']>= start_date) & (df['created_date']< end_date)]
    sns.set_style('whitegrid')
    language_counts = df['language'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=language_counts.index, y=language_counts.values, palette='viridis')

    # Adding titles and labels
    plt.title('Counts of reviews per Languages', fontsize=15)
    plt.xlabel('Language', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.yscale('log')
    # Show the plot
    if output_file is not None:
        plt.savefig(f'{output_file}/language_distribution_plot.png')
    plt.show()


def plot_piechart_teams(df: pd.DataFrame, output_file: Optional[str] = None) -> None:
    """ generates pie chart with rate of review allocation for each team 

    Parameters
    ----------
    df: pd.DataFrame :
        dataframe containing the whole review dataset 

    output_file: Optional[str] :
         (Default value = None)
        outputfile path in case user wants to save the figure
    Returns
    -------

    """
    
    aggr = df.groupby('related_team').count().reset_index()
    #sns.set(style="whitegrid")

    plt.figure(figsize=(5, 5))
    plt.pie(aggr['created_date'], labels=aggr['related_team'], autopct='%1.1f%%', startangle=140)
    plt.title('reviews assigned to each team')
    if output_file is not None:
        plt.savefig(f'{output_file}/rating_series_plot.png')
    plt.show()



def plot_rating_series(df: pd.DataFrame, team: Optional[str] = 'all', output_file: Optional[str] = None) -> None:
    """

    Parameters
    ----------
    df: pd.DataFrame :
        dataframe containing the whole review dataset 
        
    team: Optional[str] :
         (Default value = 'all')
        str containing the name of the team the data should be filtered on. Could also be 'all' for the whole dataset
         
    output_file: Optional[str] :
         (Default value = None)
        outputfile path in case user wants to save the figure
    Returns
    -------

    """
    if team is None or team != 'all':
        selected = df[df.related_team==team]
    else: 
        selected = df
    time_df = selected[['created_date','label']].set_index('created_date')

    # Resample by week and calculate mean
    weekly_summary = time_df.resample('W').agg({'label': ['mean', 'sem']})
    weekly_summary.columns = weekly_summary.columns.droplevel()

    confidence = st.norm.ppf(0.90)  # Z-score for 90% confidence
    weekly_summary['lower'] = weekly_summary['mean'] - confidence * weekly_summary['sem']
    weekly_summary['upper'] = weekly_summary['mean'] + confidence * weekly_summary['sem']
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_summary.index, weekly_summary['mean'], marker='o', linestyle='-', label='Weekly Average')
    plt.fill_between(weekly_summary.index, weekly_summary['lower'], weekly_summary['upper'], color='b', alpha=0.2)

    plt.title('Weekly Average of Numeric Value with 90% Confidence Interval', fontsize=15)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Average stars', fontsize=12)
    plt.legend()
    plt.grid(True)
    if output_file is not None:
        plt.savefig(f'{output_file}/rating_series_plot.png')
    plt.show()



def plot_number_reviews_series(df: pd.DataFrame, team: Optional[str] = 'all', output_file: Optional[str] = None) -> None:
    """

    Parameters
    ----------
    df: pd.DataFrame :
        dataframe containing the whole review dataset 
    team: Optional[str] :
         (Default value = 'all')
        str containing the name of the team the data should be filtered on. Could also be 'all' for the whole dataset
         
    output_file: Optional[str] :
         (Default value = None)
         outputfile path in case user wants to save the figure

    Returns
    -------

    """

    if team is None or team != 'all':
        selected = df[df.related_team==team]
    else: 
        selected = df
    time_df = selected[['created_date','label']].set_index('created_date')

        # Resample by week and calculate mean
    weekly_summary = time_df.resample('W').agg({'label': ['count']})
    weekly_summary.columns = weekly_summary.columns.droplevel()
    mean_count = weekly_summary['count'].mean()
    std_count = weekly_summary['count'].std()

    # Calculate control limits
    ucl = mean_count + 3 * std_count
    cl = mean_count
    lcl = mean_count - 3 * std_count

    above_ucl = weekly_summary['count'] > ucl
    below_lcl = weekly_summary['count'] < lcl

    upward_trend, downward_trend = identify_trends(weekly_summary['count'].values, 3)
    # Plotting with trend identification
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_summary.index, weekly_summary['count'], marker='o', linestyle='-', label='Weekly Average')

    # Add control lines
    plt.axhline(y=ucl, color='r', linestyle='--', label='UCL')
    plt.axhline(y=cl, color='g', linestyle='-', label='CL')
    plt.axhline(y=lcl, color='b', linestyle='--', label='LCL')

    plt.scatter(weekly_summary.iloc[upward_trend].index, weekly_summary.iloc[upward_trend]['count'], color='red', label='Upward Trend > 3 Weeks', zorder=3  )
    plt.scatter(weekly_summary.iloc[downward_trend].index, weekly_summary.iloc[downward_trend]['count'], color='green', label='Downward Trend > 3 Weeks', zorder=3  )
    # Highlight points outside the control limits
    plt.scatter(weekly_summary[above_ucl].index, weekly_summary[above_ucl]['count'], color='red', zorder=3 )
    plt.scatter(weekly_summary[below_lcl].index, weekly_summary[below_lcl]['count'], color='red', zorder=3)


    # Finalize plot
    plt.title('Weekly count of reviews related to ' + team, fontsize=15)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.legend()
    plt.grid(True)
    if output_file is not None:
        plt.savefig(f'{output_file}/{team}_series_n_reviews.png')
    plt.show()



def plot_keywords(df: pd.DataFrame, team: str, target_language: str ='en', target_month: Optional[int] = None,
                   target_year: Optional[int] = None,  start_date: Optional[str] = None, end_date: Optional[str] = None, output_file: Optional[str] = None) -> None:
    """

    Parameters
    ----------
    df: pd.DataFrame :
        dataframe containing the whole review dataset 
    team: str :
        str containing the name of the team the data should be filtered on
        
    target_language: str :
         (Default value = 'en')
         the sataset will be filtered in order to plot keywords exclusively from the selected language

    target_month: Optional[int] :
         (Default value = None)
        the keywords plot will only contain words from reviews given on that month

    target_year: Optional[int] :
         (Default value = None)
         the keywords plot will only contain words from reviews given on that year

    start_date: Optional[str] :
         (Default value = None)
        the plot will only contain words from reviews given on the specified time window (MM/DD/YYYY)

    end_date: Optional[str] :
         (Default value = None)
         the plot will only contain words from reviews given on the specified time window (MM/DD/YYYY)

    output_file: Optional[str] :
         (Default value = None)
        outputfile path in case user wants to save the figure

    Returns
    -------

    """

    selected = df[df.related_team==team]
    if target_month is not None:
        selected = selected[(selected['created_date'].dt.month == target_month) & (selected['created_date'].dt.year == target_year)]
    else:
        selected = selected[(selected['created_date']>= start_date) & (selected['created_date']< end_date)]

    keyword_columns = [item for item in selected.columns.values if item.startswith('keyword')]
    selected = selected[selected['language'] == target_language]
    selected = pd.melt(selected, value_vars=keyword_columns, id_vars=[team,'label']).rename(columns={'value': 'keyword'})
    selected = selected[['keyword', 'label',team]].groupby('keyword').mean().reset_index()
    selected['label'] = selected['label'].apply(lambda x: x+ np.random.rand() -0.5)
    selected[team] = selected[team].apply(lambda x: x+ np.random.normal(0, 0.2) )
    plt.figure(figsize=(15, 9))
    scatter = plt.scatter(selected['label'], selected[team], c=selected['label'], cmap='viridis', s=10)

    # Adding annotations
    for i in range(len(selected)):
        plt.text(selected['label'].iloc[i], selected[team].iloc[i], selected['keyword'].iloc[i], 
                color='red', ha='right', fontsize=8)

    # Adding colorbar
    plt.colorbar(scatter)

    # Setting labels
    plt.xlabel('stars')
    plt.ylabel('pertinence (similarity with topic)')
    if target_month is not None:
        plt.title('Keywords for reviews in '+target_language + ' mostly related to '+ team + ' team in ' +str(target_month) + '-' + str(target_year))
    else:
        plt.title('Keywords for reviews in ' + target_language +' mostly related to '+ team + ' team from ' +str(start_date) + '-' + str(end_date))
    if output_file is not None:
        plt.savefig(f'{output_file}/{team}_keywords_plot.png')
    plt.show()
