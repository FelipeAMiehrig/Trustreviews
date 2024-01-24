# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import emoji
import html

def append_if_different(title: str, text: str)-> str:
    """ concats title and review text if they are different

    Parameters
    ----------
    title :
        title of the review
        
    text :
        content of the review

    Returns
    -------
        concated text

    """
    if title[:-1] == text[:len(title)-1]:
        return text
    elif title != text:
        return title + ' ' + text 
    else:
        return text
    


# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def build_dataset(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    Parameters
    ----------
    input_filepath :
        filepath to find  thee raw data in
        
    output_filepath :
        filepath to save the processed data in

    Returns
    -------

    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(f'{input_filepath}/data.csv', index_col=0)
    df['created_date'] = pd.to_datetime(df['created_date'], utc=True)
    df.dropna(subset=['Text', 'stars'], inplace=True)
    df['Text'] = df['Text'].apply(lambda x: emoji.demojize(html.unescape(x), delimiters=(" ", " ")))
    df['Title'] = df['Title'].apply(lambda x: emoji.demojize(html.unescape(x), delimiters=(" ", " ")))
    df['full_text'] = df.apply(lambda row: append_if_different(row['Title'], row['Text']), axis=1)
    df = df.sort_values(by='created_date')
    df = df.rename(columns={"full_text": "text", "stars": "label"}).reset_index(drop=True)
    logger.info('saving processed dataset')
    df.to_csv(f'{output_filepath}/processed.csv', index=False, encoding="utf-8")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    build_dataset()
