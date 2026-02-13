# CSCE 676 — Data Mining Project

## Predictive Maintenance via Anomaly Detection on Turbofan Engine Sensor Data

This project applies data mining techniques to the NASA C-MAPSS Turbofan Engine Degradation Simulation dataset (FD001). The goal is to detect engine degradation using both traditional and learned anomaly detection methods.

## Dataset

**NASA C-MAPSS (FD001):** 20,631 rows across 100 simulated turbofan engines run to failure. Each row contains 21 sensor measurements and 3 operational settings recorded per operational cycle.

- **Source:** [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **License:** U.S. Government Public Domain

### Data Setup

The raw data is not included in this repo. To reproduce:

1. Download the dataset from the [NASA source](https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip)
2. Extract into `data/6. Turbofan Engine Degradation Simulation Data Set/`

## Techniques

**Course topics:**
- Anomaly detection (degradation onset detection)
- Streaming/time-series analysis

**Beyond-course technique:**
- Autoencoder-based anomaly detection — train on healthy engine cycles and use reconstruction error to flag degradation

## Project Structure

```
project/
├── README.md
├── checkpoint1.ipynb    # Checkpoint 1: Dataset selection and EDA
└── data/                # Raw data (not committed)
```

## Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn, scikit-learn, scipy

## Reference

A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," in Proc. 1st Int. Conf. on Prognostics and Health Management (PHM), 2008.
