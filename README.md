# Real-Time Fault Diagnosis for a Chemical Process

This repository contains an end-to-end data science project that builds and deploys a machine learning model to diagnose operational faults in a simulated Sour Water Treatment Unit (SWTU). The project demonstrates a complete workflow from data exploration and model training to interpretation and deployment as an interactive web application.

**Live Demo:** [Link to your deployed Streamlit App will go here]

## Key Features

- **End-to-End Workflow:** Covers data cleaning, exploratory data analysis, model training, hyperparameter tuning, and evaluation.
- **Advanced Modeling:** Utilizes a `RandomForestClassifier` optimized with `RandomizedSearchCV` to handle complex, non-linear relationships.
- **Interpretability:** Goes beyond accuracy to provide deep insights through Feature Importance analysis and a detailed Confusion Matrix evaluation.
- **Interactive Application:** Deploys the final model as a user-friendly Streamlit web app with both file upload and manual input capabilities.
- **Professional Practices:** Emphasizes professional data science practices, including handling imbalanced data, stratified sampling, and clear documentation.

## Project Overview

The goal of this project was to bridge the gap between chemical engineering domain knowledge and data science by building a robust diagnostic tool for a common industrial process. The model analyzes 26 real-time and historical sensor readings (pressures, temperatures, flows, and levels) to classify the plant's current state into one of seven categories: "Normal Operation" or one of six specific fault conditions. This provides operators with immediate, specific insights that go far beyond simple high/low alarms, enabling faster and more accurate responses to process upsets.

## The Dataset

This project utilizes a public dataset generated using **Aspen Plus Dynamics®** as part of a Master's thesis in Chemical Engineering. It simulates a two-column Sour Water Treatment Unit (SWTU) designed to remove H₂S and NH₃ from industrial wastewater.

- **Key Characteristic:** The dataset is **highly imbalanced**, with "Normal Operation" representing the vast majority of samples. This is a realistic reflection of industrial data and required specific techniques to handle effectively.
- **Source:** The original dataset can be found at this [GitHub repository](https://github.com/nogueira-ju/SWTU_FDD). We extend our gratitude to the author for making this high-quality dataset publicly available.

## Methodology

The project followed a structured, iterative workflow:

1.  **Exploratory Data Analysis (EDA):** The initial analysis confirmed the data was clean but highly imbalanced. This critical insight guided our choice of models and evaluation metrics. A correlation heatmap also revealed multicollinearity between sensor variables, a common feature of industrial processes.
2.  **Baseline Modeling:** A `RandomForestClassifier` was trained with `class_weight='balanced'` to establish an initial performance benchmark. It achieved an impressive **95.87% accuracy** on the validation set.
3.  **Hyperparameter Tuning:** `RandomizedSearchCV` was used to optimize the model's hyperparameters, focusing on the `f1_macro` score to ensure strong performance across all classes, especially the rare fault conditions.
4.  **Final Evaluation:** The tuned model was evaluated on a completely unseen test set, achieving **92.58% accuracy**. A detailed analysis of the **Confusion Matrix** was performed to identify the model's specific strengths and weaknesses.

## Key Insights

The final model is highly accurate, but the true value comes from its interpretation:

1.  **Feature Importance:** The model identified that a small subset of flow (`FC`), pressure (`PC`), and temperature (`TC`) controllers were the most important predictors of process health. This aligns perfectly with chemical engineering first principles and gives engineers confidence in the model's logic.
2.  **Error Analysis:** The confusion matrix revealed that the model's primary weakness is occasionally misclassifying **Fault 2** (Significant Feed Composition Variance) as "Normal Operation." This specific, actionable insight is crucial for real-world deployment, as it informs operators about the model's limitations and where to focus their own expert monitoring.

## How to Run This Project

To run the Streamlit application on your local machine, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/mmsaleem3737/Chemical-Process-Fault-Diagnosis/tree/main
    cd Chemical-Process-Fault-Diagnosis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    Your web browser will open with the application running.

## Technologies Used

- **Programming Language:** Python
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Web App Framework:** Streamlit
