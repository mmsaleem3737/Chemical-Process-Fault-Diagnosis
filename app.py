import streamlit as st
import pandas as pd
import joblib
import os

# ====================================================================
# 1. PAGE CONFIGURATION AND MODEL LOADING
# ====================================================================

# Set page configuration
st.set_page_config(
    page_title="SWTU Fault Detection",
    page_icon="🔬",
    layout="wide"
)

# Function to load the model and columns (cached for performance)
@st.cache_resource
def load_model():
    """Load the trained model and model columns."""
    try:
        model = joblib.load('model/final_fault_detection_model.joblib')
        model_columns = joblib.load('model/model_columns.joblib')
        return model, model_columns
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'final_fault_detection_model.joblib' and 'model_columns.joblib' are in the 'model' directory.")
        return None, None

# Load the model
model, model_columns = load_model()

# Define human-readable labels for the classes
class_labels = {
    0: "Normal Operation",
    1: "Fault 1: Significant Feed Flow Variance",
    2: "Fault 2: Significant Feed Composition Variance",
    3: "Fault 3: Column 1 Sensor Problem (Pressure)",
    4: "Fault 4: Column 2 Sensor Problem (Pressure)",
    5: "Fault 5: Feed Overheating",
    6: "Fault 6: Fouling in Heat Exchanger"
}

# ====================================================================
# 2. APP UI AND LAYOUT
# ====================================================================

st.title("🔬 Real-Time Fault Diagnosis for a Sour Water Treatment Unit")
st.markdown("This application uses a trained `RandomForestClassifier` to diagnose the operational state of a simulated chemical process. Upload a data file or use the manual sliders in the sidebar to get started.")

st.sidebar.header("Data Input Method")

input_method = st.sidebar.radio(
    "Choose how to provide data:",
    ("Upload a CSV File", "Manual Slider Input")
)

# Initialize the input_df to avoid scope errors
input_df = None

if input_method == "Upload a CSV File":
    st.header("Upload Your Sensor Data")
    
    # Provide a template for the user to download
    if model_columns is not None:
        template_df = pd.DataFrame(columns=model_columns)
        template_csv = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV Template",
            data=template_csv,
            file_name="diagnosis_template.csv",
            mime="text/csv",
        )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Load the uploaded data
            input_df = pd.read_csv(uploaded_file)
            # Validate columns
            if list(input_df.columns) != model_columns:
                st.error("The uploaded file has incorrect columns. Please use the template.")
                input_df = None # Reset df to prevent prediction
            else:
                st.success("File uploaded successfully. Displaying diagnosis results below.")
        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")
            input_df = None

elif input_method == "Manual Slider Input":
    if model_columns is not None:
        st.sidebar.header("Manual Sensor Controls")
        st.sidebar.markdown("The data is standardized, so values typically range from -5 to 5.")

        with st.sidebar.form(key='input_form'):
            st.subheader("Sensor Readings")
            col1, col2 = st.columns(2)
            input_data = {}
            base_columns = [c.replace('_t', '') for c in model_columns if c.endswith('_t')]

            for i, col_name in enumerate(base_columns):
                target_col = col1 if i % 2 == 0 else col2
                input_data[f"{col_name}_t"] = target_col.slider(f"{col_name} (time t)", -5.0, 5.0, 0.0, 0.1)
                input_data[f"{col_name}_t-1"] = target_col.slider(f"{col_name} (time t-1)", -5.0, 5.0, 0.0, 0.1)
            
            submit_button = st.form_submit_button(label='Diagnose Process State')
            
            if submit_button:
                input_df = pd.DataFrame([input_data], columns=model_columns)

# ====================================================================
# 3. PREDICTION LOGIC AND RESULT DISPLAY
# ====================================================================

if input_df is not None and model is not None:
    # Make predictions
    predictions = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Create a results dataframe
    results_df = input_df.copy()
    results_df['Predicted Class'] = predictions
    results_df['Diagnosis'] = results_df['Predicted Class'].map(class_labels)
    results_df['Confidence'] = [prediction_proba[i, pred] * 100 for i, pred in enumerate(predictions)]
    
    st.header("Diagnosis Results")
    
    # Display the dataframe with diagnoses
    st.dataframe(results_df[['Diagnosis', 'Confidence'] + model_columns])

    # Add a summary
    st.write("---")
    st.subheader("Results Summary")
    fault_counts = results_df['Diagnosis'].value_counts()
    st.bar_chart(fault_counts)

else:
    st.info("Choose an input method and provide data to see the model's diagnosis.")

# ====================================================================
# 4. PROJECT DOCUMENTATION
# ====================================================================

st.write("---")
st.header("Project Documentation")

with st.expander("ℹ️ About this Project", expanded=True):
    st.write("""
    This project demonstrates an end-to-end machine learning workflow for building a fault detection and diagnosis system for a chemical process. 
    
    - **Objective:** To classify the current operational state of a Sour Water Treatment Unit (SWTU) into 'Normal' or one of six specific fault conditions.
    - **Model:** A `RandomForestClassifier` was chosen for its high performance and interpretability. The model was optimized using `RandomizedSearchCV`.
    - **Key Insight:** The model achieved over 92% accuracy on unseen test data. Feature importance analysis revealed that a handful of pressure, flow, and temperature controllers were the most critical indicators of process health.
    """)

with st.expander("📈 Model Performance and Insights"):
    st.write("""
    The final model was rigorously evaluated to ensure its robustness and to understand its specific behaviors.
    """)
    col1, col2 = st.columns(2)
    
    if os.path.exists('images/confusion_matrix.png'):
        with col1:
            st.image('images/confusion_matrix.png', caption='Confusion Matrix on Unseen Test Data')
            st.markdown("**Finding:** The model performs very well across all classes but shows a slight tendency to misclassify Fault 2 as 'Normal'. This highlights the importance of detailed error analysis.")
    
    if os.path.exists('images/feature_importance.png'):
        with col2:
            st.image('images/feature_importance.png', caption='Model Feature Importance')
            st.markdown("**Finding:** The model relies most heavily on a combination of flow, pressure, and temperature controllers, which aligns perfectly with chemical engineering principles.")

with st.expander("❓ How to Use This App"):
    st.write("""
    You can diagnose process conditions in two ways:
    
    **1. Upload a CSV File (Recommended):**
    - Click 'Download CSV Template' to get the required file format.
    - Fill the template with your own sensor data.
    - Upload the file using the 'Choose a CSV file' button.
    - The app will diagnose every row in your file.
    
    **2. Manual Slider Input:**
    - Select this option in the sidebar.
    - Adjust the 26 sliders to simulate a specific process condition.
    - Click 'Diagnose Process State' to see the result for that single data point.
    """)

with st.expander("📚 Data Source"):
    st.write("""
    This project utilizes a public dataset generated as part of a Master's thesis in Chemical Engineering. The data was created using Aspen Plus Dynamics® to simulate a Sour Water Treatment Unit (SWTU).
    
    - **Original Author:** nogueira-ju on GitHub
    - **Source Repository:** [https://github.com/nogueira-ju/SWTU_FDD](https://github.com/nogueira-ju/SWTU_FDD)
    
    We extend our gratitude to the author for making this high-quality, realistic dataset available to the public.
    """)