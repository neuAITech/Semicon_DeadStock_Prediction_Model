# Semiconductor Dead Stock Risk Predictor

## 📌 Project Overview
This project implements a Machine Learning pipeline to identify "Dead Stock" (inventory at high risk of obsolescence) within a semiconductor supply chain.

By integrating **Sales Orders, Customer Forecasts, and Purchase Orders** into a unified "Spine" dataset, the model predicts inventory write-off risks using an **XGBoost Classifier**. It features domain-specific feature engineering (based on JEDEC standards) and utilizes **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable reasons for every risk flag.

## 🚀 Key Features
* **Unified Data Spine:** Merges disparate transaction logs (Sales, POs, Forecasts) into a single timeline.
* **Automated Risk Tagging:** Rule-based logic to identify current Zombie Stock, Financial Write-offs, and Shelf-Life Expiries.
* **Semiconductor Feature Engineering:** Calculates specialized metrics like `Life_Consumed_Ratio`, `Cash_Conversion_Gap`, and `Stock_to_Target_Ratio`.
* **Leakage-Free Modeling:** Rigorous feature selection to remove target leakage before training.
* **Explainable AI:** Generates global and local SHAP plots to explain *why* specific SKUs are risky.

## 🛠️ Prerequisites
* **Python 3.8+**
* **Input Data:** An Excel file (`.xlsx`) containing the following sheets:
    * `Product_Master`
    * `Inventory_Lot_Master`
    * `Customer_Master`
    * `Customer_Forecast`
    * `Sales_Order`
    * `Purchase_Order`

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/semiconductor-risk-prediction.git](https://github.com/your-username/semiconductor-risk-prediction.git)
    cd semiconductor-risk-prediction
    ```

2.  **Create a virtual environment (Optional but Recommended):**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file (see below) or install packages manually:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap openpyxl joblib
    ```

## ⚙️ Configuration
**Crucial Step:** Before running the script, you must update the file path to point to your local dataset.

1.  Open `train_model.py` (or whatever you named your script).
2.  Locate the variable `path` near the top of the file:
    ```python
    # UPDATE THIS LINE
    path = '/Users/kanishkbhagat/DS/Semiconductors/India_Semiconductor_Distributor_v3_10000.xlsx'
    ```
3.  Change it to the actual location of your Excel file.

## 🏃‍♂️ How to Run

1.  **Execute the script:**
    Run the Python script from your terminal:
    ```bash
    python train_model.py
    ```

2.  **During Execution:**
    * The script will print data processing stats (Row counts, Data types).
    * It will display the **Feature Importance Plot** (Close the plot window to continue the script).
    * It will display **SHAP Summary & Force Plots** (Close the plot windows to finish execution).

3.  **Output:**
    * **Console:** Prints Classification Report (Precision/Recall), AUC Score, and top risk factors.
    * **File:** Saves the trained model as `semiconductor_dead_stock_model_v1.pkl` in the current directory.

## 📊 Model Outputs
The script generates the following artifacts:

| Artifact | Description |
| :--- | :--- |
| **Console Report** | Accuracy metrics, Confusion Matrix, and Data Health stats. |
| **Feature Importance Plot** | Visual ranking of which features (e.g., `Age`, `Cost`) drive the model. |
| **SHAP Force Plot** | Interactive visualization explaining the risk score for specific items. |
| **.pkl File** | The serialized XGBoost model ready for production deployment. |

## 📂 Project Structure
