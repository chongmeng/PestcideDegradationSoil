# PestcideDegradationSoil
Modelling Pesticide Degradation in Soil: Influence of Chemical and Soil Properties on DT50 Prediction.  
Data available on request.

## Repository Structure
- `Descriptor_cdk.r` — Generate molecular descriptors from SMILES with rcdk.
- `XGBoost_opt.py` — Hyperparameter search for XGBoost via GridSearchCV.
- `XGBoost.py` — Train/evaluate the final XGBoost model, visualize results, and save artifacts.
- `xgboost_analysis.py` — Partial Dependence Plots (PDP) and SHAP analysis using the saved model and data.
- `selectbyshap_xgboost.py` — SHAP-based feature selection and cross-validation performance.

## Data
- Not included; available on request.
- Expected file: `data.xlsx` at repo root (should include `DT50` and `SMILE` plus other features).

## Usage

1. Generate molecular descriptors based on SMILE codes
```bash
Rscript Descriptor_cdk.r
```
2. Hyperparameter search
```bash
python XGBoost_opt.py
```
3. Train, evaluate, and save model/data (edit save paths in script)
```bash
python XGBoost.py
```
4.  Model interpretation (PDP + SHAP)
```bash
python xgboost_analysis.py
```
5. SHAP-based feature selection
```bash
python selectbyshap_xgboost.py
```
## Contact/Citation
Data available on request. Please cite the associated manuscript when using this code.
