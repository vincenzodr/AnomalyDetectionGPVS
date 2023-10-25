# Anomaly Detection in GPVS

## Table of Contents
- [Introduction](#introduction)
- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [Models](#models)
- [Evaluation](#evaluation)

## Introduction

Anomaly Detection in Grid-connected Photovoltaic Systems (GPVS) is crucial for optimizing performance, efficiency, and system longevity. This repository presents a time series-based approach to PV anomaly detection.

## Overview

This project focuses on the detection of anomalies in PV systems using innovative machine learning models. The primary models include TimesNet and LSTM-FCN, designed specifically for time series data. The project addresses challenges like data imbalances, dynamic system behavior, and real-time monitoring.

## Features

- Utilizes time series data to capture temporal patterns in PV system behavior.
- Offers advanced models like TimesNet and LSTM-FCN for anomaly detection.
- Provides solutions for imbalanced data in the anomaly detection process.

## Data

The project uses the GPVS-Faults dataset, a valuable resource for PV anomaly detection.
You can access the GPVS-Faults dataset [here]([https://your-dataset-link.com](https://data.mendeley.com/datasets/n76t439f65/1)). For more information about the dataset, please refer to the documentation provided with the dataset.

## Models

The project offers two primary models for PV anomaly detection:

- **TimesNet**: A deep learning model designed to capture time series patterns in PV data. This model is particularly effective in identifying temporal anomalies. You can access the TimesNet model repository [here](https://github.com/vincenzodr/Time-Series-Library) for more information, code, and documentation.

- **LSTM-FCN**: A hybrid model that combines Long Short-Term Memory (LSTM) and Convolutional layers, providing a versatile approach to anomaly detection in PV systems.

## Evaluation

We employ evaluation metrics to assess the performance of the anomaly detection models. These metrics include:

- **Precision**: The proportion of true positive predictions out of all positive predictions.
- **Recall**: The proportion of true positive predictions out of all actual positive instances.
- **F1-Score**: The harmonic mean of precision and recall, providing a balanced assessment of model performance.
- **Accuracy**: The proportion of correct predictions out of all predictions made.
