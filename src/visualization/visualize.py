import pandas as pd
import os
import hydra
from hydra.core.config_store import ConfigStore 
from omegaconf import DictConfig
import warnings
from src.visualization.plots import plot_language_distribution, plot_rating_series, plot_piechart_teams, plot_number_reviews_series, plot_keywords

warnings.filterwarnings("ignore")

@hydra.main(config_path='../conf', config_name='config.yaml', version_base=None)
def generate_report(cfg:DictConfig):
    """ plots all the required charts and saves them under report/figures

    Parameters
    ----------
    cfg:DictConfig :
        hydra config object

    Returns
    -------

    """
    
    df = pd.read_csv(f'{cfg.paths.in_folder_reports}/keywords_data.csv', encoding="utf-8")
    df['created_date'] = pd.to_datetime(df['created_date'], utc=True)
    teams = cfg.topics.words
    
    plot_language_distribution(df, output_file=cfg.paths.out_folder_figures)

    plot_piechart_teams(df, output_file=cfg.paths.out_folder_figures)

    plot_rating_series(df, output_file=cfg.paths.out_folder_figures)

    plot_number_reviews_series(df, output_file=cfg.paths.out_folder_figures)

    for team in teams:

        if not os.path.exists(f'{cfg.paths.out_folder_figures}/{team}'):
            os.mkdir(f'{cfg.paths.out_folder_figures}/{team}')
        plot_number_reviews_series(df, team=team, output_file=f'{cfg.paths.out_folder_figures}/{team}')

        plot_keywords(df, team=team, target_language=cfg.plots.language, start_date=cfg.plots.date.start_date,
                                    end_date=cfg.plots.date.end_date, output_file=f'{cfg.paths.out_folder_figures}/{team}')

if __name__ == '__main__':

    generate_report()



