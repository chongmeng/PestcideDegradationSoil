# Modelling Pesticide Degradation in Soil: Influence of Chemical and Soil Properties on DT50 Prediction

This repository contains the analysis code associated with the manuscript:

**Modelling Pesticide Degradation in Soil: Influence of Chemical and Soil Properties on DT50 Prediction**

The study uses XGBoost regression to model pesticide degradation half-lives (DT50) in soil based on molecular descriptors and soil properties.

## Repository structure

```text
PestcideDegradationSoil/
│
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── 01_descriptor_generation.R
│   ├── 02_hyperparameter_optimization.py
│   ├── 03_model_training.py
│   ├── 04_model_interpretation.py
│   └── 05_shap_feature_selection.py
│
├── data/
│   └── README.md
│
└── results/
    └── README.md
```

## Data

The datasets are not included in this repository.

The modelling analyses use datasets containing pre-calculated molecular descriptors together with chemical and soil properties. The original SMILES information and other source data used for molecular descriptor generation are confidential and are therefore not provided.

The expected input datasets are:

```text
data.xlsx
```

The underlying data are available from the authors upon reasonable request, subject to applicable data-use restrictions.

## Analysis workflow

The repository documents the computational workflow used for the reported analyses.

### 1. Molecular descriptor generation

`01_descriptor_generation.R`

This script documents the molecular descriptor generation procedure used in the study.

The procedure uses molecular structures represented as SMILES and the Chemistry Development Kit (CDK). The original SMILES data and source datasets are confidential and are not included in this repository.

The modelling datasets used in the subsequent analyses already contain the required molecular descriptors. Therefore, descriptor generation is **not required** to run the modelling workflow using the analysis datasets.

### 2. Hyperparameter optimization

`02_hyperparameter_optimization.py`

XGBoost hyperparameters are optimized using grid search with five-fold cross-validation on the training dataset.

The hyperparameter optimization was performed separately for the different datasets/analyses. The resulting best parameters were subsequently used in the final model analyses.

### 3. Final model training and evaluation

`03_model_training.py`

The selected hyperparameters are specified in the script and used to train the final XGBoost model.

The model is evaluated using:

* Training-set MSE and R²
* Test-set MSE and R²
* Five-fold cross-validation R²
* Five-fold cross-validation MSE, including mean and standard deviation

### 4. Model interpretation

`04_model_interpretation.py`

Partial dependence plots (PDPs) and SHAP analyses are used to investigate feature effects and model interpretation.

### 5. SHAP-based feature selection

`05_shap_feature_selection.py`

SHAP values are used to assess feature importance and support feature-selection analyses.

## Modelling approach

The modelling workflow includes:

* XGBoost regression
* Natural log transformation of DT50
* Exclusion of DT50 values greater than 1000 days
* Min-Max scaling of predictor variables
* 80/20 training-test split
* Random seed: 42
* Five-fold cross-validation
* Grid-search hyperparameter optimization
* Mean squared error (MSE)
* R²

Missing values are retained where applicable, and XGBoost is used to handle missing predictor values.

## Software

The analysis was performed using:

* Python 3.9.19
* R 4.3.2
* pandas 2.2.2
* NumPy 2.0.1
* scikit-learn 1.5.1
* XGBoost 2.1.1
* Matplotlib 3.9.2
* Seaborn 0.13.2
* SHAP 0.46.0

Python dependencies are listed in `requirements.txt`.

## Running the modelling workflow

Because the underlying datasets and confidential molecular structure information are not distributed with this repository, complete reproduction of the reported results requires access to the corresponding datasets.

For users with access to the datasets, the modelling scripts are intended to be run in the following order:

```text
02_hyperparameter_optimization.py
            ↓
03_model_training.py
            ↓
04_model_interpretation.py
            ↓
05_shap_feature_selection.py
```

The descriptor-generation script (`01_descriptor_generation.R`) is not part of the required modelling workflow because the molecular descriptors are already present in the analysis datasets.

## Results

The `results/` directory is intended for generated model files, figures, and other analysis outputs.

Generated results are not included by default.


## Citation

If you use this repository or the associated analysis, please cite the associated manuscript:

> Xu, C. M. et al. *Modelling Pesticide Degradation in Soil: Influence of Chemical and Soil Properties on DT50 Prediction*. Environmental Toxicology and Chemistry.


## Contact

For questions regarding the analysis or access to the underlying data, please contact the corresponding author.
