trustpilot_reviews
==============================

keyword extraction of user reviews using KeyBERT (https://maartengr.github.io/KeyBERT/)  
Review content to topic mapping using custom cosine similarity canculations.  
Both methods use the subclass of pretrained BERT LLM for multilingual text (https://huggingface.co/bert-base-multilingual-cased)   
For a showcase of the results please refer to notebooks/showcase.ipynb

## get started

The project uses a slightly altered version of the cookie cutter data science template as shown bellow.  
We also use hydra for parameter configuration (https://hydra.cc/docs/intro/)

to start:
```bash
pip install -e .
```
then we instal the required packages 

```bash
pip install -r requirements.txt
```

After that we can start by running the script the script that generates 
the keywords for each review and calculates the similarities between the reviews and each team (ex: legal, product, commercial)


```bash
python src/features/build_keywords.py
```

We can change the number of keywords per review by passing an argument (default is 3)

```bash
python src/features/build_keywords.py keywords.n_words=2
```

now the keywords and the similarity scores were computed and the output dataset was saved in data/final
Time to generate plots 

```bash
python src/visualization/visualize.py 
```

Now we generated several plots that are saved under report/figures
plots for reviews assigned to specific teams are found under report/figures/{team}


We can change parameters for the plots like the language of keywords shown by passing arguments (defaults to english "en")

```bash
python src/visualization/visualize.py plots.language=it
```

for a specific month and year we want to see keyword plots from (defaults are December 2021)

```bash
python src/visualization/visualize.py plots.date.month=11 plots.date.year=2021
```
we can also change the time window of the keyword plot by running (MM/DD/YYYY)

```bash
python src/visualization/visualize.py plots.date.start_date=11/01/2021 plots.date.end_date=11/07/2021 
```

to wrap up, lets plot everything and keyword plots for reviews written in German for the month of October in 2021


```bash
python src/visualization/visualize.py plots.language=de plots.date.month=10 plots.date.year=2021 
```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── final          <- final data with keywords and similarity to teams(topics).
    │   ├── processed      <- The processed version of the dataset, emojis are tranlated to words and text is cleaned.
    │   └── raw            <- The original data provided.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Eventual models that would be fine tuned on the review dataset would go here
    │
    ├── notebooks          <- Jupyter notebooks for preeliminary EDA and experiments
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Project plan and eventual automatic generated reports would go here
    │   └── figures        <- Plots of the analysis go here
    |       └── commercial <- Plots reletad to each teams go here
    |       └── legal
    |       └── product
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py <- contains the build_dataset function that gets the original data, processes and saves in data/processed
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_keywords.py <- build_keywords function that extracts keywords for the whole dataset and computes similarities between reviews and trustpilot teams
    |   |   └── utils.py <- contains several functions to be called on build_features.py 
    |   |                            
    │   │
    │   ├── models         <- Eventual scripts to train (fine tune) models and then use trained models to make
    │   │   │                 predictions would go here
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├─── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   |    └── visualize.py <- Generates plots for general EDA and for each team that are saved under reports/figures/
    |   |    └── plots.py     <- Contains plotting functions to be called on visualize.py
    |   └── conf 
    |       └── confg.yaml  <- file containing the configuration for hydra
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
