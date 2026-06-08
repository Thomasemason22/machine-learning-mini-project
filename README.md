# Machine Learning Mini Project

This project demonstrates a complete introductory data science workflow using Python. It explores multiple real-world datasets, creates visualizations, preprocesses mixed numeric and categorical data, and compares machine learning models for binary classification.

## Project Overview

The analysis covers five datasets:

- Education and income samples
- Online gaming engagement data
- Student performance and final exam scores
- Teen mental health and depression indicators
- Maharashtra city weather trends

The machine learning portion focuses on the teen mental health dataset. It predicts `depression_label` using behavioral, academic, social media, sleep, stress, anxiety, and activity features.

## Methods Used

- Exploratory data analysis
- Data cleaning and preprocessing
- Missing value imputation
- One-hot encoding
- Feature scaling
- Correlation analysis
- Classification modeling
- Neural network training
- Model evaluation with accuracy, classification reports, and confusion matrices

## Models Compared

- Decision Tree Classifier
- K-Nearest Neighbors
- Artificial Neural Network
- Deeper Artificial Neural Network with Dropout

## Key Takeaway

The classification task shows strong overall accuracy, but the teen mental health dataset is highly imbalanced. Most records belong to the non-depression class, which means accuracy alone can be misleading. For that reason, the project reports precision, recall, F1-score, and confusion matrices in addition to accuracy.

## Repository Structure

```text
.
├── data/
│   ├── Teen_Mental_Health_Dataset.csv
│   ├── cities_weather_maharashtra.csv
│   ├── education_vs_income_econometrics_samples.csv
│   ├── online-gaming-10-04-26.csv
│   └── student_performance_finalscore.csv
├── figures/
├── src/
│   └── mini_project.py
├── requirements.txt
└── README.md
```

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python src/mini_project.py
```

For a faster test run, reduce neural network training epochs:

```bash
python src/mini_project.py --epochs 3
```

Generated charts are saved in the `figures/` folder.

## Skills Demonstrated

- Python programming
- Data analysis
- Machine learning
- Data visualization
- Pandas and NumPy
- Scikit-learn pipelines
- TensorFlow/Keras neural networks
- Model evaluation
- CSV data processing

## LinkedIn Project Description

Built a Python-based machine learning mini project that explores multiple real-world datasets and applies classification models to teen mental health data. The project includes exploratory data analysis, data preprocessing, feature scaling, one-hot encoding, visualizations, and model comparison using Decision Tree, K-Nearest Neighbors, and neural network classifiers. This project strengthened my skills in Python, data analysis, machine learning workflows, and model evaluation.
