# China Underground Coal Mine Methane Database and RF-MICE Imputation

**Author:** Wu Ziqi  
**Date:** 2025  

## Overview

This repository contains a database of China's underground coal mines from 2011 to 2023, along with code for performing machine learning-based multiple imputation and uncertainty estimation (RF-MICE). The work supports research on methane emissions from coal mines and related uncertainty quantification.

---

## Repository Structure

├── code/
│   ├── MICE_RF.py # Main code for RF-MICE imputation and uncertainty estimation
│   └── ... # Additional data processing and plotting scripts (available upon request)
├── data/
│   ├── Database.xlsx # Mine-level database (2011–2023) with mine characteristics and emission factors
│   └── Figure.xlsx # Auxiliary data used for figures in the paper
└── README.md

---

## Code Description

- The **`code/MICE_RF.py`** script implements a Random Forest-based Multiple Imputation by Chained Equations (RF-MICE) procedure.
- It provides:
  - Parallelized imputation of missing values in the coal mine database.
  - Out-of-Bag (OOB) variance estimation for each sample, including missing-value uncertainty propagation.
  - Rubin’s combination rules to compute final mean values and uncertainty estimates.
- The main script requires Python ≥3.8 and the following packages:
  - `numpy`, `pandas`, `scikit-learn`, `joblib`, `openpyxl`

> **Note:** The plotting and additional data processing scripts are not included in the repository; they can be requested from the author.

---

## Data Description

- **Database.xlsx**: Mine-level dataset including:
  - Year, Province, Production, Mine Type, Identification, Depth, Methane and CO₂ emissions, Coal seam info, Latitude/Longitude (rounded to 4 decimals for safety), and other mine attributes.
- **Figure.xlsx**: Auxiliary datasets used to generate figures in the manuscript.

> **Privacy Note:** Longitude, latitude, and other sensitive columns have been rounded to 4 decimal places to protect mine location confidentiality.

---

## Contact

> **Note:** Wu Ziqi — wu-zq24@mails.tsinghua.edu.cn