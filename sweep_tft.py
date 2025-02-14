# if these are imported first, the code will work
import numpy as np
import darts
import pandas as pd
import torch

import sys
sys.path.append('externals/respinow_ml')# Fixes importing issues in external modules

from externals.respinow_ml.src.load_data import *
from darts.dataprocessing.transformers import StaticCovariatesTransformer
from sklearn.preprocessing import OneHotEncoder
from darts.utils.likelihood_models import NegativeBinomialLikelihood
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.callbacks import TFMProgressBar
from sklearn.preprocessing import MinMaxScaler

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import wandb

from utils import get_covariates_dict, evaluate_model
from likelihood_utils import PositiveQuantileRegression

with wandb.init(tags=['TFT-Quantile',]):
    config = wandb.config

    # Load data
    ts_names = ['survstat', 'icosari', 'agi', 'cvn']
    ts = load_data(ts_names)

    # Create static covariates
    ts.static_covariates.drop(columns=['source', 'target'], inplace=True)

    covariates_transformer = StaticCovariatesTransformer(transformer_cat=OneHotEncoder())
    ts = covariates_transformer.fit_transform(ts)

    # Get Targets
    train_end = pd.Timestamp('2018-09-30')
    validation_start = pd.Timestamp('2018-10-07')
    validation_end = pd.Timestamp('2019-09-29')
    test_start = pd.Timestamp('2019-10-06')
    test_end = pd.Timestamp('2020-09-27')

    TARGETS = [t for t in ts.columns if 'survstat-influenza' in t]  # The 19 time series we want to predict
    targets, covariates = target_covariate_split(ts, TARGETS)

    # Static covariates
    encoders = {
        'cyclic': {'future': ['month', 'weekofyear']},
        #'transformer': Scaler()
    }


    # Normalize targets
    scale = True
    if scale:
        encoders["transformer"] = Scaler()
        scaler = Scaler(MinMaxScaler(feature_range=(0.0001, 1)),
                        global_fit=False)  # Scale the data using a Scaler from sklearn
        scaler.fit(targets[: train_end])  # Fit the scaler to the training data
        targets = scaler.transform(targets)  # Transform the data

        scaler2 = Scaler(MinMaxScaler(feature_range=(0.0001, 1)),
                         global_fit=False)  # Scale the data using a Scaler from sklearn
        scaler2.fit(covariates[: train_end])  # Fit the scaler to the training data
        covariates = scaler2.transform(covariates)  # Transform the data


    use_covariates = config.use_covariates is True or config.use_covariates == "true" or config.use_covariates == "True"
    if use_covariates is True:
        print("Using covariates")
    covariates_dict_train, covariates_dict_val = get_covariates_dict(use_covariates,
                                                                     covariates,
                                                                     train_end,
                                                                     validation_start,
                                                                     validation_end,
                                                                     config.input_chunk_length)

    wandb.log({'likelihood': 'PosiveQuantileRegression'})
    wandb.log({'learning_rate_scheduler': 'ReduceLROnPlateau'})
    #likelihood = NegativeBinomialLikelihood()
    likelihood = PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975])

    logger = WandbLogger(project='tsmixer')
    model = TFTModel(input_chunk_length=config.input_chunk_length,  # Lookback window [15, 30, 40, 45]
                     output_chunk_length=4,
                     likelihood=likelihood,
                     use_static_covariates=True,
                     hidden_size=config.hidden_size,  # Main Hyperparameter [8, 16, 32, 64, 128, 256]
                     num_attention_heads=config.num_attention_heads,  # Hyperparameter [1, 2, 4, 8]
                     dropout=config.dropout,  # Hyperparameter [0.05, 0.1, 0.2, 0.3]
                     batch_size=16,
                     n_epochs=500,
                     add_encoders=encoders,
                     save_checkpoints=True,
                     optimizer_cls=torch.optim.AdamW,
                     optimizer_kwargs={'lr': config.lr},  # Hyperparameter [1e-3, 5e-4, 1e-4]
                     lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     lr_scheduler_kwargs={'factor': 0.5, 'patience': 10, 'threshold': 1e-4},
                     pl_trainer_kwargs={"logger": logger,
                                        "accelerator": "gpu",
                                        "devices": 1,
                                        "enable_progress_bar": True,
                                        "enable_model_summary": True,
                                        "enable_checkpointing": True,
                                        "log_every_n_steps": 10,
                                        "callbacks": [TFMProgressBar(enable_sanity_check_bar=False,
                                                                     enable_validation_bar=False),
                                                      EarlyStopping(monitor="val_loss",
                                                                    patience=20,
                                                                    min_delta=1e-5,
                                                                    mode="min")
                                                      ]
                                        }
                     )

    # Train model
    model.fit(series=targets[:train_end],
              val_series=targets[validation_start - config.input_chunk_length * targets.freq:validation_end],
              **covariates_dict_train)
    tft = TFTModel.load_from_checkpoint(model_name=model.model_name, best=True)

    # Evaluate model
    _ = evaluate_model(model=model,
                       targets=targets,
                       start=validation_start,
                       end=validation_end,
                       covariates=covariates_dict_val,
                       deterministic=False,
                       scaler=scaler if scale else None,
                       verbose=False,
                       log=True)
