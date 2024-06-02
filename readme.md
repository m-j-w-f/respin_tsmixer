# Deep Learning for Infectious Disease Forecasting: TSMixer

This repo contains the code for the seminar paper "Deep Learning for Infectious Disease Forecasting: TSMixer".

## The repository is structured as follows:
### 📁 Folders:
- ```plots```: Contains the plots generated in the notebooks.
- ```results```: Contains the results of the experiments.
### 📔 Notebooks:
- ```crossvalidate.ipynb```: Notebook for cross-validating the models.
- ```explore.ipynb```: Notebook for initial experimentation.
- ```test.ipynb```: Notebook for testing and evaluating the models.

### 🐍 Python files:
- ```likelihood_utils.py```: Contains the modified Quantile Regression class.
- ```model_utils.py```: Contains the model classes with the best hyperparameters.
- ```utils.py```: Contains utility functions for data preprocessing, evaluation and plotting.
- ```sweep_*.py```: Contains the sweep functions for the hyperparameter tuning used by WandB.
