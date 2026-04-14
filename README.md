# AI-SOC-Threat-Detection-System
A beginner-friendly, industry-oriented machine learning project for detecting cyber threats from network traffic using NSL-KDD, Random Forest, alert scoring, and visualization.


# AI-Powered Cybersecurity Threat Detection System

## Project Overview
This project is a beginner-friendly yet industry-oriented cybersecurity machine learning system that detects suspicious network activity using the NSL-KDD dataset. It simulates a real-world threat detection workflow used in SOC environments by combining data preprocessing, machine learning-based attack classification, alert generation, and visualization.

## Problem Statement
Traditional rule-based security systems struggle to detect evolving cyber threats and unusual behavior patterns. This project uses machine learning to classify network traffic as normal or attack and generate threat alerts based on prediction confidence.

## Industry Relevance
AI-driven threat detection is widely used in:
- Security Operations Centers (SOC)
- Intrusion Detection Systems (IDS)
- Fraud monitoring platforms
- Network traffic analytics
- Endpoint and cloud security

This project demonstrates how machine learning can support cybersecurity teams in detecting malicious behavior from structured traffic data.

## Features
- Load and preprocess NSL-KDD dataset
- Binary attack detection: Normal vs Attack
- Random Forest classifier for threat detection
- Alert generation with severity levels
- Evaluation metrics: Accuracy, Precision, Recall, F1 Score
- Confusion matrix and feature importance visualization
- GitHub-ready modular project structure

## Tech Stack
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Dataset
This project uses the NSL-KDD intrusion detection dataset.

Place these files in:
```text
data/raw/KDDTrain+.txt
data/raw/KDDTest+.txt
