"""
Author: Wu Ziqi
Date: 2025
Description:
    Parallelized fine-grained uncertainty modeling using Random Forest-based 
    Multiple Imputation by Chained Equations (RF-MICE).
    Each sample obtains an individualized uncertainty estimate via OOB error propagation.
"""

import numpy as np
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

# ==== Step 1. Load data ====
file_path = 'data/pre/processed_data.xlsx' 
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Drop unnecessary columns
drop_cols = [
    'City', 'Mine_Name', 'Max_Absolute_Gas_Emission_Workface',
    'Max_Absolute_Gas_Emission_Roadway', 'Absolute_CO2_Emission_Mine',
    'Relative_CO2_Emission_Mine', 'Remarks', 'UID', 'Index'
]

df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Missing value mask
mask_missing = df.isnull()
n_rows, n_cols = df.shape
n_imputations = 3

def estimate_oob_error_for_column(i, j, df, n_cols, n_rows):
    """
    Estimate Out-of-Bag (OOB) squared errors for column j using Random Forest.
    For missing values, estimate uncertainty by sampling similar records 
    with the same year, province, and outburst classification.
    """
    notna_mask = ~df.iloc[:, j].isna()
    if notna_mask.sum() < 5:
        return np.full(n_rows, np.nan)  # insufficient samples

    X = df.drop(df.columns[j], axis=1).values
    y = df.iloc[:, j].values

    X_train = X[notna_mask]
    y_train = y[notna_mask]

    rf = RandomForestRegressor(n_estimators=1, bootstrap=True, max_depth=10, oob_score=False)
    rf.fit(X_train, y_train)

    # Manual OOB estimation for non-missing values
    n_samples = X_train.shape[0]
    oob_preds = np.zeros(n_samples)
    oob_counts = np.zeros(n_samples)
    rng = np.random.default_rng()

    for tree in rf.estimators_:
        bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[bootstrap_indices] = False

        if oob_mask.sum() == 0:
            continue

        X_oob = X_train[oob_mask]
        pred_oob = tree.predict(X_oob)

        oob_preds[oob_mask] += pred_oob
        oob_counts[oob_mask] += 1

    valid = oob_counts > 0
    oob_preds[valid] /= oob_counts[valid]
    oob_errors = np.full_like(y_train, np.nan)
    oob_errors[valid] = y_train[valid] - oob_preds[valid]

    # Construct full-sample error vector
    full_error = np.full(n_rows, np.nan)
    full_error[notna_mask] = oob_errors   # OOB for observed samples

    # ===  OOB estimation for missing values ===
    missing_idx = np.where(df.iloc[:, j].isna())[0]

    for idx in missing_idx:
        year = df.loc[idx, 'Year']
        province_cols = [col for col in df.columns if col.startswith('Province_')]
        province = df.loc[idx, province_cols].idxmax().replace('Province_', '')

        outburst_cols = [col for col in df.columns if col.startswith('Outburst_Identification_')]
        outburst = df.loc[idx, outburst_cols].idxmax().replace('Outburst_Identification_', '')

        # Find candidate samples with the same conditions
        candidates = []
        for sample_idx in np.where(notna_mask)[0]:
            sample_year = df.loc[sample_idx, 'Year']
            sample_province = df.loc[sample_idx, province_cols].idxmax().replace('Province_', '')
            sample_outburst = df.loc[sample_idx, outburst_cols].idxmax().replace('Outburst_Identification_', '')

            if (sample_year == year) and (sample_province == province) and (sample_outburst == outburst):
                candidates.append(sample_idx)

        if candidates:
            chosen = np.random.choice(candidates)
            full_error[idx] = oob_errors[candidates].mean()
        else:
            # Keep NaN if no similar samples
            pass

    return full_error ** 2  # return squared errors


def impute_and_estimate_u(i, df):
    """
    Perform one round of random forest imputation and 
    estimate pointwise uncertainty via OOB-based variance.
    """
    print(f"--- Imputation round {i} ---")
    rf = RandomForestRegressor(n_estimators=1, max_depth=10, bootstrap=True)
    imputer = IterativeImputer(estimator=rf, max_iter=1, tol=0.01, random_state=None, sample_posterior=False)
    imputed_array = imputer.fit_transform(df)
    imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
    print(f"--- Round {i} imputation completed ---")

     # Parallel OOB variance estimation
    U_i = Parallel(n_jobs=-1)(
        delayed(estimate_oob_error_for_column)(i, j, df, n_cols, n_rows) for j in range(n_cols)
    )

    return imputed_df, np.array(U_i)


# ==== Step 2. Parallelized imputation and uncertainty estimation ====
results = Parallel(n_jobs=-1)(
    delayed(impute_and_estimate_u)(i, df) for i in range(n_imputations)
)

# Extract results
imputed_dfs = [result[0] for result in results]
U_all = np.array([result[1] for result in results])

# ==== Step 3. Rubin’s combination rules ====
imputed_values = np.stack([df_.values for df_ in imputed_dfs], axis=0)  # [D, n, p]
B = np.var(imputed_values, axis=0, ddof=1)  # between-imputation variance

U_bar = np.nanmean(U_all, axis=0)
print("U_bar.shape:", U_bar.shape)
print("B_bar.shape:", B.shape)
T = U_bar.T + (1 + 1 / n_imputations) * B


# Keep only variance corresponding to missing positions
T_df = pd.DataFrame(T, columns=df.columns)
mean_df = pd.DataFrame(np.mean(imputed_values, axis=0), columns=df.columns)
B_df = pd.DataFrame(B, columns=df.columns)
U_df = pd.DataFrame(U_bar.T, columns=df.columns)

# ==== Step 4. Save results ====
mean_df.to_excel("data/result/uncertainty/imputed_mean.xlsx", index=False)
B_df.to_excel("data/result/uncertainty/rubin_B_between_var.xlsx", index=False)
U_df.to_excel("data/result/uncertainty/rubin_U_pointwise_var.xlsx", index=False)
T_df.to_excel("data/result/uncertainty/rubin_T_total_var_only_missing.xlsx", index=False)

print("All imputations and uncertainty estimation completed successfully.")
