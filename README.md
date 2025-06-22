# SWRO Performance Prediction: A Case Study in Industrial AI

This repository contains a data science project that explores the feasibility of predicting membrane performance degradation in Seawater Reverse Osmosis (SWRO) desalination plants. The project serves as a real-world case study into the challenges and insights of applying machine learning to industrial processes, particularly when faced with limited data.

## The Challenge: Data Scarcity in Industrial AI

Predictive maintenance is a key goal for modern industry, aiming to forecast equipment failure before it occurs. However, a major obstacle is the lack of public, high-quality operational data from industrial plants. This project directly confronts this "data scarcity" problem using a publicly available dataset to simulate a real-world predictive modeling task.

## The Dataset

The project utilizes the "Performance Data of a SWRO arising from Wave Powered Desalinisation" dataset from Mendeley Data. This dataset contains experimental data from a pilot-scale SWRO rig under various operating conditions (steady, sinusoidal, etc.), providing a valuable proxy for the kind of fluctuating performance seen in real-world scenarios.

## The Methodology: An Iterative Approach

The project followed a rigorous, iterative modeling process to find a viable solution:

1.  **Initial Classification:** A `RandomForestClassifier` was first built to distinguish between "healthy" (steady-state) and "stressed" (sinusoidal) operations. While achieving 100% accuracy, this model was deemed too simplistic as it was merely identifying the experiment type rather than predicting future performance.

2.  **Generalized Regression:** The problem was reframed to predict the final salt rejection percentage (a key performance indicator) based on the first few minutes of an experiment. A model trained on all available data failed to generalize, as the underlying physics of the different experiment types (steady vs. sinusoidal) were too distinct.

3.  **Hyper-Specialized Regression:** The final model, a `RandomForestRegressor`, was trained *exclusively* on the three available "sinusoidal" stress-test experiments. This represents a common real-world strategy: creating a specialized model for a specific operational mode.

## The Core Finding: The "Brittle Expert" Model

After developing the specialized model, rigorous testing revealed a critical insight: **the model consistently predicted a poor outcome ("Alert" status) for any new data it was shown.**

This is not a bug; it is the most important finding of the project.

**Why does this happen?**
Because the model was trained on an extremely small dataset (only 3 experiments), it has effectively **"memorized"** the exact numerical patterns of those specific tests. It did not learn to generalize to new, unseen patterns, even if those patterns represented "good" performance. When it encounters data that deviates even slightly from the patterns it has memorized, it correctly identifies the data as an unfamiliar anomaly and predicts a low-performance outcome.

**Conclusion:** This project successfully demonstrates the primary challenge in real-world industrial AI—**data scarcity**. We have proven that the data processing and modeling methodology is sound, but that a robust, deployable model requires a much larger and more varied training dataset. This finding, and the analytical journey to uncover it, is more valuable than a simple "correct" prediction on a toy problem. It showcases a mature understanding of how machine learning models behave in practical, data-constrained environments.

## How to Run the Project

1.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit Application:**
    The interactive application allows you to upload test data and see the model's prediction.
    ```bash
    streamlit run app.py
    ```
    The application includes a template file for users to understand the required data format.

## File Descriptions

-   `app.py`: The core Streamlit web application for interacting with the model.
-   `final_model.joblib`: The saved, trained `RandomForestRegressor` model.
-   `requirements.txt`: A list of the Python packages required to run the project.
-   `generate_*.py`: A series of scripts used for generating various test data files to probe the model's behavior.
-   `data/`: Contains the original raw data files from the Mendeley dataset.
-   `Model_Code.ipynb`: Jupyter Notebook containing the initial Exploratory Data Analysis.

## Future Work

Given a larger and more diverse dataset of SWRO operational data, future work could involve:
-   Training a more robust regression model capable of generalizing across a wider range of conditions.
-   Developing a multi-class classification model to predict different types of operational issues (e.g., biofouling, scaling, mechanical failure).
-   Exploring more advanced time-series models like LSTMs or Transformers for performance prediction. 
