

import pandas as pd
import torch
import pycountry
import nltk
from torch import Tensor
from typing import List, Dict, Tuple
from nltk.corpus import stopwords
from torch.nn.functional import cosine_similarity


def flatten_list(nested_list : List[List[str]]) -> List[str]:
    """ flattens a list of lists of strs

    Parameters
    ----------
    nested_list : List[List[str]] :
        list of lists of strs to be flattened
        

    Returns
    -------
        flattened list

    """

    flat_list = []
    for element in nested_list:
        if isinstance(element, list):
            # If the element is a list, extend flat_list with the flattened version of this element
            flat_list.extend(flatten_list(element))
        else:
            # If the element is not a list (including strings), append it directly to flat_list
            flat_list.append(element)
    return flat_list

def get_language_name(abbr: str) -> str:
    """ gets the unabbreviated word for the language in english

    Parameters
    ----------
    abbr: str :
        str containing the abbreviation of the language like: en, de, it ...
        

    Returns
    -------
        the full name of the language in english like: german, italian ...
    """
    try:
        return pycountry.languages.get(alpha_2=abbr).name.lower()
    except AttributeError:
        return 'english'
    


def get_all_words_languages(languages: List[str]) -> List[str]:
    """ gets all stopwords from a specific language

    Parameters
    ----------
    languages: List[str] :
        list containing all languages to get stop words from 
        
    Returns
    -------
        list containing all stop words from all the desired languages

    """
    nltk.download('stopwords')
    all_lang = [get_language_name(language) for language in languages]
    all_lang = [lang for lang in all_lang if lang in stopwords.fileids()]
    all_stopwords = [list(set(stopwords.words(lang_code))) for lang_code in all_lang]
    return flatten_list(all_stopwords)

def get_dict_of_keywords(keywords: List[Tuple[str, float]]) -> Dict[str, str]:
    """ outputs a dictionary containing the keywords extracted for a review

    Parameters
    ----------
    keywords: List[Tuple[str, float]] :
        list of tuples of strings (identified keywords) and the scores 

    Returns
    -------
        a dictionary containing only the keyword identifierL example: keyword_1  and the word itself in the value 

    """
    dict_keywords = {}
    for i, ele in enumerate(keywords):
        dict_keywords['keyword_'+str(i+1)] = ele[0]
    return dict_keywords

def calculate_keywords_for_all(docs: List[str], kw_model,stopwords: List[str] ,max_n_grams: int = 2, n_words: int = 3) -> pd.DataFrame:
    """ for a list of documents calculates the dictionary of keywords for all of them and combines them in a df

    Parameters
    ----------
    docs: List[str] :
      list of documents
        
    kw_model :
        KeyBERT model
        
    stopwords: List[str] :
        list of stopwords to use
        
    max_n_grams: int :
         (Default value = 2)
         number of n_grams to use on the keyword extraction
    n_words: int :
         (Default value = 3)
         mnumber of keywords to extract per review

    Returns
    -------
        A df containing a coluon for each keyword extracted

    """
    list_dicts = []
    for doc in docs:
        keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1,max_n_grams)
        , use_maxsum = True, nr_candidates=20, top_n=n_words,stop_words =stopwords, diversity=0.2
        )
        dict_keywords = get_dict_of_keywords(keywords)
        list_dicts.append(dict_keywords)
    return pd.DataFrame(list_dicts)

def get_last_hidden_state_sentence(doc: str, model, tokenizer, device) -> Tensor:
    """gets a last hidden state representation of a sentence (str) excluding sentence start and end characters

    Parameters
    ----------
    doc: List[str] :
        the str to be passed
        
    model :
        the NN model to embed the text
        
    tokenizer :
        the model specific tokenizer
        
    device :
        the pytorch device

    Returns
    -------
        A tensor containin the embedded sentence with a dimension for each token

    """
    sentence = doc

# Tokenize the sentence and prepare input to the model
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

# Pass the inputs through the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

# Extract the last layer hidden states
    return hidden_states[-1][0, 1:-1, :]

def get_topic_representation(dict_topics: Dict[str, str], model, tokenizer, device) -> Dict[str, Tensor]:
    """ gets the topics and their enriched text in dict form and outputs a dict containing the topic 
    and the embedded tensor representation

    Parameters
    ----------
    dict_topics: Dict[str, str] :
        dict of topics(keys) and enriched text (values)
        
    model :
        NN language model
        
    tokenizer :
        the model tokenizer
        
    device :
        the pytorch device
        

    Returns
    -------
        a dict with the topics and the embedded representations

    """
    dict_representation = {}
    for key, value in dict_topics.items():
        sentence_representation = get_last_hidden_state_sentence(value, model, tokenizer, device)
        dict_representation[key] = sentence_representation.mean(dim=0).unsqueeze(0)
    return dict_representation

def get_dict_similarity_sentence_topic(doc: str, dict_topics, model, tokenizer, device) -> Dict[str, float]:
    """gets a sentence and the dict of topics outputing the distance between the sentence and each topic in a dict

    Parameters
    ----------
    doc: str :
        sentence to be calculated distance 
        
    dict_topics :
        dict of topics and their enriched text
        
    model :
        NN language model
        
    tokenizer :
        model tokenizer
        
    device :
        pytorch device
        

    Returns
    -------
        returns the dict with the distance for each topic to the passed text

    """
    dict_sim_topics = {}
    try:
        sentence_representation = get_last_hidden_state_sentence(doc, model, tokenizer, device) 
    except:
        for key, value in dict_topics.items():
            dict_sim_topics[key]= 0
        return dict_sim_topics
    for key, value in dict_topics.items():
        sim = cosine_similarity(sentence_representation.mean(dim=0).unsqueeze(0),dict_topics[key])
        dict_sim_topics[key]= sim.item()
    return dict_sim_topics

def calculate_similarity_for_all(docs: List[str], dict_topics,model, tokenizer, device) -> pd.DataFrame:
    """gets all texts in the list and the dict of topics outputing the distance between the sentence and each topic in a dict

    Parameters
    ----------
    docs: List[str] :
        list of texts to be calculated distance 
        
    dict_topics :
        dict of topics and their enriched text
        
    model :
        NN language model
        
    tokenizer :
        model tokenizer
        
    device :
        pytorch device

    Returns
    -------
        dataframe containing the distances

    """
    list_dicts= []
    for doc in docs:
        list_dicts.append(get_dict_similarity_sentence_topic(doc,dict_topics,model, tokenizer, device))
    return pd.DataFrame(list_dicts)