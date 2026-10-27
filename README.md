UFC Matchup & Outcome Predictor

A data-driven machine learning tool designed to analyze historical fight statistics, engineer advanced fighter performance metrics, and predict UFC fight outcomes using Python and SQL.

Features

Data Engineering & Storage: Processes raw fight datasets (ufc-master.csv) and structures data into a local SQLite database (db/ufc.db).

Advanced Feature Engineering: Calculates dynamic metrics including historical win/loss streaks, striking/grappling differentials, and custom ELO rating systems.

Machine Learning & Analysis: Trains predictive models to forecast bout winners and leverages SHAP (SHapley Additive exPlanations) for feature importance analysis.

Modular Architecture: Clean separation of concerns across extraction (etl.py), feature generation (features.py), model training (train.py), evaluation (evaluate.py), and predictions (predict.py).

Repository Structure

├── data/
│   ├── raw/
│   └── processed/
├── db/              # SQLite database storage
├── docs/            # Project documentation and analysis reports
├── models/          # Model artifacts
├── notebooks/       # EDA and experimental notebooks
├── reports/         # Visualizations (e.g., SHAP summary plots)
├── sql/             # Database schemas and feature queries
├── src/             # Core Python scripts (ETL, training, inference)
├── .gitignore
├── README.md
└── requirements.txt


Getting Started

Prerequisites

Clone the repository and install the required dependencies:

git clone https://github.com/your-username/ufc-matchup-predictor.git
cd ufc-matchup-predictor
pip install -r requirements.txt


Usage

Run Pipeline & Predictions:

python src/predict.py


Train Models:

python src/train.py


Explore Notebooks:
Launch Jupyter to view experimental data analysis and model evaluation:

jupyter notebook notebooks/01_explore_data.ipynb


License

This project is open-source and available under the MIT License.