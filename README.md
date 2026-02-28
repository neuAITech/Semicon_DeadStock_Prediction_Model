
# 🛡️ Semiconductor Dead Stock Predictor

An AI-powered inventory risk management tool designed to predict the probability of semiconductor components becoming **Dead Stock**. By analyzing aging, sourcing patterns, and manufacturer reliability, this application provides actionable insights for procurement and warehouse teams.

## 🚀 Overview

In the semiconductor industry, inventory stagnation can lead to massive financial write-offs due to rapid technological obsolescence and strict expiry dates. This tool uses a trained **XGBoost machine learning model** to evaluate individual SKUs and categorize them by risk level.

### Key Features

* **Real-time Risk Analysis:** Instant probability scoring for stock items.
* **Feature Engineering:** Automatically calculates critical metrics like *Life Consumed Ratio* and *Aging Criticality*.
* **Financial Exposure Tracking:** Calculates total value at risk based on unit cost and quantity.
* **Actionable Recommendations:** Suggests liquidation or monitoring strategies based on the risk score.

---

## 🛠️ Technical Stack

* **Frontend:** [Streamlit](https://streamlit.io/) (Interactive Web UI)
* **Machine Learning:** XGBoost (Gradient Boosted Decision Trees)
* **Data Handling:** Pandas & NumPy
* **Model Serialization:** Joblib

---

## 📦 Installation & Setup

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/semiconductor-dead-stock.git
cd semiconductor-dead-stock

```


2. **Install Dependencies:**
```bash
pip install -r requirements.txt

```


3. **Model Placement:**
Ensure your trained model file `semiconductor_dead_stock_model_v1.pkl` is in the root directory.
4. **Run the App:**
```bash
streamlit run app.py

```



---

## 🧠 How It Works

The application transforms simple user inputs into complex features that the AI model understands:

| Input Field | Engineered Feature | Why it Matters |
| --- | --- | --- |
| **Dates** | `Life_Consumed_Ratio` | Measures how much of the product's shelf life has already passed. |
| **Expiry** | `Is_Aging_Critical` | Boolean flag triggered if the item expires in less than 180 days. |
| **Source** | `Source_Type_Sales Order` | Identifies if the stock is a return, which historically has higher dead-stock risk. |
| **Cost/Qty** | `Total_Inventory_Value` | Weighs the financial impact of the prediction. |

---

## 📊 Risk Categories

* 🔴 **High Risk (>65%):** Items requiring immediate liquidation or bundling.
* 🟡 **Moderate Risk (35% - 65%):** Items showing signs of stagnation; re-ordering should be paused.
* 🟢 **Safe Stock (<35%):** Healthy inventory levels; normal procurement applies.

---

## 📝 Author

**Kanishk Bhagat** *AI Engineer*

---
