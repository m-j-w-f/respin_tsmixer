import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
from darts.models import DLinearModel, NLinearModel, TSMixerModel, TFTModel
from darts.utils.likelihood_models import NegativeBinomialLikelihood
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from darts.utils.callbacks import TFMProgressBar
import sys
from likelihood_utils import PositiveQuantileRegression


class MetricsLogger(Callback):
    def __init__(self):
        self.learning_rates = []
        self.train_losses = []
        self.train_losses_buffer = []
        self.val_losses = []
        self.epochs = []
        self.current_epoch = 0
        self.steps_per_epoch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.learning_rates.append(trainer.optimizers[0].param_groups[0]['lr'])
        self.train_losses_buffer.append(outputs['loss'].item())

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_losses.append(np.mean(self.train_losses_buffer))
        self.train_losses_buffer = []

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch += 1
        if self.steps_per_epoch is None:
            self.steps_per_epoch = len(trainer.train_dataloader)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.val_losses.append(trainer.callback_metrics['val_loss'].item())
        self.epochs.append(self.current_epoch)

    def on_fit_end(self, trainer, pl_module):
        self._generate_plots()

    def _generate_plots(self):
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        # Plot Learning Rate
        axs[0].plot(self.learning_rates)
        axs[0].set_title('Learning Rate over Training')
        axs[0].set_xlabel('Training Step')
        axs[0].set_ylabel('Learning Rate')
        axs[0].set_yscale('log')
        axs[0].grid(True)

        # Calculate x-axis values for train and validation losses
        train_steps = np.arange(len(self.train_losses))
        val_steps = np.arange(len(self.val_losses))

        # Determine cutoff for y-axis
        max_loss = max(max(self.train_losses), max(self.val_losses))
        y_cutoff = np.percentile(self.train_losses, 99)  # Cut off at 95th percentile

        # Plot Loss
        axs[1].plot(train_steps, self.train_losses, label='Train Loss')
        axs[1].plot(val_steps[1:], self.val_losses[1:], label='Validation Loss')
        axs[1].set_title('Loss over Training')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].set_ylim(min(min(self.val_losses[1:]), min(self.train_losses)),
                        min(y_cutoff, max_loss))  # Set y-axis limit
        axs[1].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()


class ModelFactory:
    def __init__(self, model_kwargs, fixed_kwargs):
        self.model_class = model_kwargs.pop("model_class")
        self.name = model_kwargs.pop("name")
        model_kwargs.pop("scale", None)  # Remove scale from model_kwargs
        self.model_kwargs = model_kwargs
        self.callback_classes = fixed_kwargs["pl_trainer_kwargs"].pop("callbacks_classes")
        self.callback_kwargs = fixed_kwargs["pl_trainer_kwargs"].pop("callbacks_kwargs")
        self.fixed_kwargs = fixed_kwargs

    def get_model(self):
        fixed_kwargs = self.fixed_kwargs
        fixed_kwargs["pl_trainer_kwargs"]["callbacks"] = []
        # Instantiate Callbacks
        for callback, kwargs in zip(self.callback_classes, self.callback_kwargs):
            fixed_kwargs["pl_trainer_kwargs"]["callbacks"].append(callback(**kwargs))
        # Instantiate Model
        return self.model_class(**self.model_kwargs, **self.fixed_kwargs)


class BestConfig:
    @property
    def nlinear_config(self):
        cfg = {"model_class": NLinearModel,
               "name": "NLinear + QR",
               "input_chunk_length": 50,
               "optimizer_kwargs": {'lr': 0.0001},
               "const_init": True,
               "normalize": False,
               "scale": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "use_covariates": False,
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               }
        return cfg

    @property
    def dlinear_config(self):
        cfg = {"model_class":DLinearModel,
               "name": "DLinear + QR",
               "input_chunk_length": 50,
               "use_covariates": False,
               "scale": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "kernel_size": 25,
               "optimizer_kwargs": {'lr': 0.0001},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "const_init": True,
               }
        return cfg

    @property
    def tsmixer_config(self):
        cfg = {"model_class": TSMixerModel,
               "name": "TSMixer + NB",
               "input_chunk_length": 15,
               "use_covariates": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "likelihood": NegativeBinomialLikelihood(),
               "hidden_size": 256,
               "num_blocks": 3,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0001}
               }
        return cfg

    @property
    def tsmixer_qr_config(self):
        cfg = {"model_class": TSMixerModel,
                "name": "TSMixer + QR",
               "input_chunk_length": 40,
               "use_covariates": True,
               "scale": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']},
                                'transformer': Scaler()},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "hidden_size": 16,
               "num_blocks": 2,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    @property
    def tft_config(self):
        cfg = {"model_class": TFTModel,
                "name": "TFT + NB",
               "input_chunk_length": 45,
               "use_covariates": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "likelihood": NegativeBinomialLikelihood(),
               "hidden_size": 64,
               "num_attention_heads": 8,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    @property
    def tft_qr_config(self):
        cfg = {"model_class": TFTModel,
                "name": "TFT + QR",
               "input_chunk_length": 30,
               "use_covariates": False,
               "scale": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']},
                                'transformer': Scaler()},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "hidden_size": 8,
               "num_attention_heads": 4,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    #------------------------------------
    # Configs with use of covariates True

    @property
    def nlinear_config_c(self):
        cfg = {"model_class": NLinearModel,
               "name": "NLinear + QR + C",
               "input_chunk_length": 50,
               "optimizer_kwargs": {'lr': 0.005},
               "const_init": True,
               "normalize": False,
               "scale": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "use_covariates": True,
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               }
        return cfg

    @property
    def dlinear_config_c(self):
        cfg = {"model_class":DLinearModel,
               "name": "DLinear + QR + C",
               "input_chunk_length": 50,
               "use_covariates": True,
               "scale": False,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "kernel_size": 25,
               "optimizer_kwargs": {'lr': 0.0001},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "const_init": True,
               }
        return cfg

    @property
    def tsmixer_config_c(self):
        cfg = {"model_class": TSMixerModel,
               "name": "TSMixer + NB + C",
               "input_chunk_length": 40,
               "use_covariates": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "likelihood": NegativeBinomialLikelihood(),
               "hidden_size": 16,
               "num_blocks": 3,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.001}
               }
        return cfg

    @property
    def tsmixer_qr_config_c(self):
        "best model (all other best models are without covariates)"
        cfg = {"model_class": TSMixerModel,
                "name": "TSMixer + QR + C",
               "input_chunk_length": 40,
               "use_covariates": True,
               "scale": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']},
                                'transformer': Scaler()},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "hidden_size": 16,
               "num_blocks": 2,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    @property
    def tsmixer_qr_config_nc(self):
        "NO COVARIATES"
        cfg = {"model_class": TSMixerModel,
                "name": "TSMixer + QR + NC",
               "input_chunk_length": 15,
               "use_covariates": False,
               "scale": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']},
                                'transformer': Scaler()},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "hidden_size": 256,
               "num_blocks": 2,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0001}
               }
        return cfg

    @property
    def tft_config_c(self):
        cfg = {"model_class": TFTModel,
                "name": "TFT + NB + C",
               "input_chunk_length": 30,
               "use_covariates": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']}},
               "likelihood": NegativeBinomialLikelihood(),
               "hidden_size": 16,
               "num_attention_heads": 4,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    @property
    def tft_qr_config_c(self):
        cfg = {"model_class": TFTModel,
                "name": "TFT + QR + C",
               "input_chunk_length": 30,
               "use_covariates": True,
               "scale": True,
               "add_encoders": {'cyclic': {'future': ['month', 'weekofyear']},
                                'transformer': Scaler()},
               "likelihood": PositiveQuantileRegression(quantiles=[0.025, 0.25, 0.5, 0.75, 0.975]),
               "hidden_size": 64,
               "num_attention_heads": 8,
               "dropout": 0.1,
               "optimizer_kwargs": {'lr': 0.0005}
               }
        return cfg

    @property
    def fixed_config(self):
        cfg = {"output_chunk_length": 4,
               "use_static_covariates": True,
               "batch_size": 16,
               "n_epochs": 250,
               "save_checkpoints": True,
               "optimizer_cls": torch.optim.AdamW,
               "lr_scheduler_cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
               "lr_scheduler_kwargs": {'factor': 0.5, 'patience': 10, 'threshold': 1e-4},
               "pl_trainer_kwargs": {"accelerator": "gpu",
                                     "devices": 1,
                                     "enable_progress_bar": True,
                                     "enable_model_summary": False,
                                     "enable_checkpointing": True,
                                     "log_every_n_steps": 10,
                                     "callbacks_classes": [TFMProgressBar, EarlyStopping, MetricsLogger],
                                     "callbacks_kwargs": [{"enable_sanity_check_bar": False,
                                                           "enable_validation_bar": False},
                                                          {"monitor": "val_loss",
                                                           "patience": 20,
                                                           "min_delta": 1e-5,
                                                           "mode": "min"},
                                                          {}]
                                  }
               }
        return cfg
