import torch
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.callbacks import Callback
from scipy.stats import nbinom
from darts import TimeSeries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
import rich
import plotnine
from plotnine import *
import matplotlib.pyplot as plt
from matplotlib.cm import gnuplot2
from matplotlib.colors import to_hex
from typing import Optional, List, Tuple
from likelihood_utils import PositiveQuantileRegression
from model_utils import ModelFactory

import sys
sys.path.append('../externals/respinow_ml')
from externals.respinow_ml.src.load_data import *
from externals.respinow_ml.src.plot_functions import *
from externals.respinow_ml.src.scoring_functions import evaluate_models


def get_covariates_dict(use_covariates, covariates):
    # Train and validate model
    dict_train = {}
    dict_val = {}
    if use_covariates:
        dict_train = {'past_covariates': covariates,
                           'val_past_covariates': covariates}
        dict_val = {'past_covariates': covariates}
    return dict_train, dict_val


def reshape_backtest_likelihood_params(backtest, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]):
    dfs = []
    for b in backtest:
        dfs.append(reshape_forecast_likelihood_params(ts_forecast=b, quantiles=quantiles))

    return pd.concat(dfs)


def reshape_forecast_likelihood_params(ts_forecast, quantiles):
    ts_df = ts_forecast.pd_dataframe()
    df_temp = pd.DataFrame()  # Create a new dataframe to store the quantile values
    ############################
    # Very Experimental Bugfix #
    ############################
    cols = pd.DataFrame(ts_forecast.pd_dataframe().columns.str.split('_'))["component"].apply(lambda x: x[0]).unique()
    cols_r = cols + "_r"
    cols_p = cols + "_p"
    cols = np.concatenate([cols_r, cols_p])

    ts_df.columns = cols
    ###############
    #  Bugfix End #
    ###############

    # Find unique base column names
    base_names = set(col.split("_")[0] for col in ts_forecast.columns)

    # Calculate and store the quantiles in the new dataframe
    for base in base_names:
        p_col = f"{base}_p"
        r_col = f"{base}_r"

        for i, row in ts_df.iterrows():
            # ! network actually outputs mu and alpha, not p and r
            p = row[p_col]
            r = row[r_col]
            quantile_values = nbinom.ppf(quantiles, n=r, p=1 - p)  # here p is the probability of success not failure
            assert not np.isnan(quantile_values).any(), f"Nan value found for p:{p}, r:{r}"

            for q, value in zip(quantiles, quantile_values):
                df_temp.at[i, f"{base}_{q}"] = value
    df_temp.index.name = 'date'
    df_temp.index.freq = '7D'
    df_temp.columns.name = 'component'

    df_temp = df_temp.reset_index().melt(id_vars='date')
    df_temp['quantile'] = df_temp.component.apply(lambda x: x.split('_')[-1])

    df_temp['strata'] = df_temp.component.apply(lambda x: x.split('-', 2)[-1].split('_')[0])  # x[19:].split('_')[0])
    df_temp[['location', 'age_group']] = df_temp.apply(extract_info, axis=1)
    df_temp['forecast_date'] = ts_forecast.start_time() - pd.Timedelta(days=7)  # 7 days before the first forecast
    df_temp['horizon'] = (df_temp.date - df_temp.forecast_date).dt.days // 7
    df_temp['forecast_date'] = df_temp['forecast_date'] + pd.Timedelta(days=4)  # TODO why is this here?

    df_temp['type'] = 'quantile'
    df_temp = df_temp.rename(columns={'date': 'target_end_date'})

    return df_temp[
        ['location', 'age_group', 'forecast_date', 'target_end_date', 'horizon', 'type', 'quantile', 'value']]


def plot_forecasts_fancy(plot_data, stratum='states', start=0, stride=5, horizon=None, confidence_interval=95):
    if stratum == 'national':
        plotnine.options.figure_size = (6, 2.5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group == '00+')]
        facet = 'location'
        ncol = 1
    elif stratum == 'states':
        plotnine.options.figure_size = (12, 10)
        df_temp = plot_data[(plot_data.location != 'DE') & (plot_data.age_group == '00+')]
        facet = 'location'
        ncol = 3
    elif stratum == 'age':
        plotnine.options.figure_size = (12, 5)
        df_temp = plot_data[(plot_data.location == 'DE') & (plot_data.age_group != '00+')]
        facet = 'age_group'
        ncol = 3

    y_temp = df_temp[df_temp.type == 'truth']

    if confidence_interval == 95:
        lower_quantile = 'quantile_0.025'
        upper_quantile = 'quantile_0.975'
    elif confidence_interval == 50:
        lower_quantile = 'quantile_0.25'
        upper_quantile = 'quantile_0.75'
    else:
        raise ValueError("Invalid confidence interval. Choose either 95 or 50.")

    if horizon:
        df_temp = df_temp[df_temp.horizon == horizon]
        g = {}
        color = {'color':'blue'}
        aes_color = {}
        fill = {'fill':'blue'}
        aes_fill = {}
    else:
        df_temp = df_temp[df_temp.forecast_date.isin(df_temp.forecast_date.unique()[start::stride])]
        g = {'group': 'forecast_date'}  # required if we plot multiple horizons at once
        df_temp['forecast_date'] = df_temp['forecast_date'].astype(str)
        forecast_dates = df_temp['forecast_date'].unique()
        colors = {date: to_hex(gnuplot2(i / len(forecast_dates))) for i, date in enumerate(forecast_dates)}
        aes_color = {'color':'forecast_date'}
        color = {}
        fill = {}
        aes_fill = {'fill': 'forecast_date'}

    return (
        ggplot(df_temp, aes(x='target_end_date')) +
        facet_wrap(facet, ncol=ncol, scales='free_y') +
        geom_ribbon(aes(ymin=lower_quantile, ymax=upper_quantile, **aes_fill, **g), **fill, alpha=0.3) +
        geom_line(aes(y='quantile_0.5', **aes_color, **g), **color) +
        geom_line(y_temp, aes(x='target_end_date', y='quantile_0.5')) +
        geom_point(y_temp, aes(x='target_end_date', y='quantile_0.5'), size=0.75) +
        (scale_fill_manual(values=colors) if not horizon else None) +
        (scale_color_manual(values=colors) if not horizon else None) +
        labs(x='Date', y='', title=f'{stratum.title()}{(" - Horizon: " + str(horizon)) if horizon else ""}') +
        theme_bw() +
        theme(legend_position='none')
    )


def evaluate_model(model,
                   targets,
                   start,
                   end,
                   covariates,
                   deterministic=False,
                   scaler=None,
                   verbose=True,
                   log=False,
                   model_name=None):
    hfc = model.historical_forecasts(series=targets[: end],
                                     **covariates,
                                     start=start,
                                     forecast_horizon=4,
                                     stride=1,
                                     last_points_only=False,
                                     retrain=False,
                                     verbose=True,
                                     num_samples=1 if deterministic else 100,
                                     predict_likelihood_parameters=True if deterministic else False
                                     )

    # Inverse transfrom of the scaler
    if scaler:
        hfc = [scaler.inverse_transform(ts) for ts in hfc]
        targets = scaler.inverse_transform(targets)

    if not deterministic:
        df = reshape_backtest(hfc)
    else:
        df = reshape_backtest_likelihood_params(hfc)
    df_ret = df.copy()
    model_name = model.__class__.__name__ if not model_name else model_name
    df['model'] = model_name
    # Convert dates to strings so that add_truth can be used
    df['forecast_date'] = df['forecast_date'].astype(str)
    df['target_end_date'] = df['target_end_date'].astype(str)
    df["quantile"] = df["quantile"].astype(np.float64)
    df = add_median(df)
    df = add_truth(df, 'influenza')

    if verbose:
        # Create a console object
        console = rich.console.Console()

        # Create a table
        table = rich.table.Table(title=f"{model_name} Evaluation Metrics")

        # Add columns to the table
        table.add_column("Metric", justify="left", style="cyan", no_wrap=True)
        table.add_column("National", justify="right", style="white")
        table.add_column("State", justify="right", style="white")
        table.add_column("Age", justify="right", style="white")

    # Calc Metrics
    metrics = ['wis', 'spread', 'overprediction', 'underprediction', 'c50', 'c95']
    levels = ['national', 'state', 'age']
    results = {metric: {} for metric in metrics}

    scores_dict = {}
    for level in levels:
        scores = evaluate_models(df, level=level)
        scores_dict[level] = scores
        for metric in metrics:
            value = scores.loc[scores['model'] == model_name, metric].values[0]
            results[metric][level] = value
            if log:
                wandb.log({f'{metric}_{level}': value})

    if verbose:
        # Add rows to the table
        for metric in metrics:
            table.add_row(metric.capitalize(),
                          str(results[metric]['national']),
                          str(results[metric]['state']),
                          str(results[metric]['age']))

        # Print the table
        console.print(table)

    return hfc, df_ret, scores_dict


def get_cross_validation_data(ts: TimeSeries, start: pd.Timestamp = None, end: pd.Timestamp = None)\
        -> List[Tuple[TimeSeries, pd.Timestamp]]:
    """
    Generate cross-validation data from a time series.

    Parameters:
    ts (TimeSeries): The input time series data.
    start (pd.Timestamp, optional): The start time for slicing the time series. Defaults to the start time of the series.
    end (pd.Timestamp, optional): The end time for slicing the time series. Defaults to the end time of the series.

    Returns:
    list: A list of tuples, where each tuple contains the complete series and the training cutoff.
    """
    start = ts.start_time() if start is None else start
    end = ts.end_time() if end is None else end
    ts_cut = ts[start:end]  # Cut of the end
    years = ts_cut.end_time().year - ts_cut.start_time().year
    slice_start = start
    chunks = []
    for _ in range(years):
        ts_chunk = ts_cut.slice_n_points_after(slice_start, 52)
        chunks.append(ts_chunk)
        slice_start = ts_chunk.end_time() + ts.freq * 1
    res = []
    for i, chunk in enumerate(chunks):
        test_chunk = chunk
        train_chunks = chunks[:i] + chunks[i+1:]
        res.append((train_chunks, test_chunk))
    final = []
    for train, test in res:
        series = train[0]
        for t in train[1:]:
            series = series.concatenate(t, axis=0, ignore_time_axis=True)
        test_series_end = series.end_time()
        series = series.concatenate(test, axis=0, ignore_time_axis=True)  # Concatenate the test data to fix the index
        final.append((series, test_series_end))
    return final


def evaluate_model_cv(model_factory: ModelFactory, ts: TimeSeries, targets_names: List[str], start=None, end=None):
    # Get cross validation data
    cv_series = get_cross_validation_data(ts, start, end)
    logging_dict = {"dfs": [], "scores": []}

    input_chunk_length = model_factory.model_kwargs["input_chunk_length"]
    use_covariates = model_factory.model_kwargs.pop("use_covariates")
    try:
        scale = model_factory.scale
    except KeyError:
        scale = False

    for (series, train_end) in cv_series:
        # Split into targets and covariates
        targets, covariates = target_covariate_split(series, targets_names)

        if scale:
            scaler1 = Scaler(MinMaxScaler(feature_range=(0.0001, 1)),global_fit=False)
            scaler2 = Scaler(MinMaxScaler(feature_range=(0.0001, 1)),global_fit=False)
            scaler1.fit(targets[:train_end])
            scaler2.fit(covariates[:train_end])
            targets = scaler1.transform(targets)
            covariates = scaler2.transform(covariates)

        # Dates
        trn_end = train_end - targets.freq * 52
        val_start = trn_end - (input_chunk_length - 1) * targets.freq
        val_end = train_end
        test_start = val_end + targets.freq
        test_end = series.end_time()

        # Covariates
        covariates_dict_train, covariates_dict_val = get_covariates_dict(use_covariates=use_covariates,
                                                                         covariates=covariates)

        # Train model
        targets[:trn_end]["survstat-influenza-DE"].plot(label="Train")
        targets[trn_end+targets.freq:val_end]["survstat-influenza-DE"].plot(label="Validation")
        targets[test_start:test_end]["survstat-influenza-DE"].plot(label="Test")
        plt.title("Train, Validation and Test Split")
        plt.show()

        model = model_factory.get_model()
        model.fit(series=targets[:trn_end],  # Use last year of training data for validation
                  val_series=targets[val_start:val_end],
                  # Validation data starts 1 step after train end, but includes the last INPUT_CHUNK_LENGTH steps
                  **covariates_dict_train)
        model = model_factory.model_class.load_from_checkpoint(model_name=model.model_name, best=True)


        det = not isinstance(model_factory.model_kwargs["likelihood"], PositiveQuantileRegression)
        hcf, df, scores = evaluate_model(model=model,
                                         targets=targets,
                                         start=test_start,
                                         end=test_end,
                                         covariates=covariates_dict_val,
                                         scaler=scaler1 if scale else None,
                                         deterministic=det,
                                         verbose=True,
                                         log=False,
                                         model_name=model_factory.name)
        logging_dict["dfs"].append(df)
        logging_dict["scores"].append(scores)
        logging_dict["cv_series"] = cv_series

    return logging_dict