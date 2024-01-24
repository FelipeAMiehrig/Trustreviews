# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pandas as pd
import os
import warnings
from keybert import KeyBERT
from transformers import AutoTokenizer
import torch
import hydra
from hydra.core.config_store import ConfigStore 
from omegaconf import DictConfig
from transformers import AutoModelForSequenceClassification, BertModel

from src.features.utils import get_all_words_languages, calculate_keywords_for_all, \
       get_topic_representation, calculate_similarity_for_all
from src.data.make_dataset import build_dataset

warnings.filterwarnings("ignore")

@hydra.main(config_path='../conf', config_name='config.yaml', version_base=None)
def build_keywords(cfg: DictConfig):
    """Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    Parameters
    ----------
    cfg: DictConfig :
        

    Returns
    -------

    """
    logger = logging.getLogger(__name__)
    if not os.path.exists(f'{cfg.paths.out_folder_raw}/processed.csv'):
        build_dataset(cfg.paths.in_folder_raw, cfg.paths.out_folder_raw)
    df =pd.read_csv(f'{cfg.paths.in_folder_processed}/processed.csv', encoding="utf-8")

    model_name =cfg.model.name
    kw_model = KeyBERT(model_name)

    docs = df.text.values

    languages = list(set(df.language.values))
    all_stopwords = get_all_words_languages(languages)
    logger.info('calculating keywords for all instances')
    df_keywords = calculate_keywords_for_all(docs, kw_model,stopwords=all_stopwords,n_words= cfg.keywords.n_words)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    topic_definition = {key: value for key, value in zip(cfg.topics.words, cfg.topics.enriched_text)}
    topics =  cfg.topics.words

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.to(device)

    logger.info('gettings topic embeddings')
    dict_topics = get_topic_representation(topic_definition, model, tokenizer, device)
    logger.info('calculating similarites between topics and reviews')
    df_similarities = calculate_similarity_for_all(docs, dict_topics,model, tokenizer, device)
    highest_value_column = df_similarities.idxmax(axis=1)

    # Create a new DataFrame with this information
    team_df = pd.DataFrame(highest_value_column, columns=['related_team'])
    logger.info('saving final csv')
    final_df = pd.concat([df,df_keywords, df_similarities, team_df],axis=1)
    final_df.to_csv(f'{cfg.paths.out_folder_processed}/keywords_data.csv', index=False, encoding="utf-8")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    build_keywords()