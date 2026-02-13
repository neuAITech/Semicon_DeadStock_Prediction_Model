import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SemiConductor AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL ---
# This uses the specific model file we saved in the training step
@st.cache_resource
def load_model():
    try:
        return joblib.load('semiconductor_dead_stock_model_v1.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please run the training script first to generate 'semiconductor_dead_stock_model_v1.pkl'.")
        return None

model = load_model()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("📦 Inventory Details")
st.sidebar.markdown("Enter stock information below:")

# 1. Basic Item Info
manufacturer = st.sidebar.selectbox(
    "Manufacturer", 
    ["Infineon Technologies", "Nxp Semiconductors", "Stmicroelectronics", "Texas Instruments", "Other"]
)

category = st.sidebar.selectbox(
    "Product Category",
    ["Mcu", "Power Ic", "Other"]
)

warehouse = st.sidebar.selectbox(
    "Warehouse Location", 
    ["Noida", "Chennai", "Pune", "Hyderabad", "Out of Stock", "Other"]
)

source_type = st.sidebar.radio(
    "Source Type", 
    ["Purchase Order (Standard)", "Sales Order (Return/Cancel)"]
)

# 2. Financials
st.sidebar.subheader("💰 Financials")
avg_unit_cost = st.sidebar.number_input("Avg Unit Cost ($)", min_value=0.01, value=12.50)
stock_qty = st.sidebar.number_input("Quantity on Hand", min_value=1, value=500)
credit_terms = st.sidebar.slider("Credit Terms (Days)", 0, 120, 60)

# 3. Dates (The Engine of the Model)
st.sidebar.subheader("📅 Lifecycle")
today = date.today()
mfg_date = st.sidebar.date_input("Manufacturing Date", value=today - timedelta(days=365))
expiry_date = st.sidebar.date_input("Expiry Date", value=today + timedelta(days=365))

# --- MAIN PANEL ---
st.title("🛡️ Semiconductor Dead Stock Predictor")
st.markdown("""
This AI tool analyzes inventory risk by detecting patterns in **aging, sourcing, and vendor reliability**.
""")

if st.sidebar.button("Run Risk Analysis", type="primary"):
    
    if model is None:
        st.stop()

    # --- STEP 1: FEATURE ENGINEERING (The "Hidden Layer") ---
    # We turn simple inputs into the complex features the model needs
    
    # Date Calculations
    days_until_expiry = (expiry_date - today).days
    total_shelf_life_days = (expiry_date - mfg_date).days
    shelf_life_months = total_shelf_life_days / 30.0
    
    # Avoid division by zero
    if total_shelf_life_days <= 0:
        life_consumed_ratio = 1.0
    else:
        life_consumed_ratio = (today - mfg_date).days / total_shelf_life_days
    
    day_index = today.timetuple().tm_yday
    is_aging_critical = 1 if days_until_expiry < 180 else 0
    total_value = avg_unit_cost * stock_qty

    # --- STEP 2: PREPARE THE DATAFRAME ---
    # This dictionary MUST match the columns X_train had during training.
    # We initialize everything to 0.
    data = {
        # High Importance Features
        'Credit_Terms_Days': credit_terms,
        'Total_Inventory_Value': total_value,
        'Avg_Unit_Cost': avg_unit_cost,
        'Source_Type_Sales Order': 1 if source_type == "Sales Order (Return/Cancel)" else 0,
        
        # Warehouse (One-Hot)
        'Primary_Warehouse_Chennai': 1 if warehouse == "Chennai" else 0,
        'Primary_Warehouse_Hyderabad': 1 if warehouse == "Hyderabad" else 0,
        'Primary_Warehouse_Noida': 1 if warehouse == "Noida" else 0,
        'Primary_Warehouse_Out of Stock': 1 if warehouse == "Out of Stock" else 0,
        'Primary_Warehouse_Pune': 1 if warehouse == "Pune" else 0,
        
        # Manufacturer (One-Hot)
        'Manufacturer_Infineon Technologies': 1 if manufacturer == "Infineon Technologies" else 0,
        'Manufacturer_Nxp Semiconductors': 1 if manufacturer == "Nxp Semiconductors" else 0,
        'Manufacturer_Stmicroelectronics': 1 if manufacturer == "Stmicroelectronics" else 0,
        'Manufacturer_Texas Instruments': 1 if manufacturer == "Texas Instruments" else 0,
        'Is_Tier1_Mfg': 1 if manufacturer in ["Infineon Technologies", "Nxp Semiconductors", "Texas Instruments", "Stmicroelectronics"] else 0,
        
        # Category (One-Hot)
        'Product_Category_Mcu': 1 if category == "Mcu" else 0,
        'Product_Category_Power Ic': 1 if category == "Power Ic" else 0,
        
        # Calculated/Date Features
        'Days_Until_Expiry': days_until_expiry,
        'Life_Consumed_Ratio': life_consumed_ratio,
        'Shelf_Life_Months': shelf_life_months,
        'Day_Index': day_index,
        'Is_Aging_Critical': is_aging_critical,
        
        # Low Importance / Placeholders (Required by model structure)
        'Customer_Type_Oem': 1, # Default assumption
        'Cash_Conversion_Gap': credit_terms - 30, # Simple proxy
    }
    
    # Convert to DataFrame
    df_input = pd.DataFrame([data])
    
    # Align columns to match model's expected order (Crucial for XGBoost)
    # This prevents "Feature Mismatch" errors
    try:
        df_input = df_input[model.get_booster().feature_names]
    except Exception as e:
        st.warning("Note: Feature alignment skipped. Ensure columns match exactly.")

    # --- STEP 3: PREDICTION ---
    risk_prob = model.predict_proba(df_input)[0][1]
    prediction = model.predict(df_input)[0]

    # --- STEP 4: VISUALIZATION ---
    st.divider()
    
    # Create 3 columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Exposure", f"${total_value:,.2f}")
    
    with col2:
        st.metric("Days Until Expiry", f"{days_until_expiry} days")
        
    with col3:
        # Dynamic color for Risk Score
        risk_color = "normal"
        if risk_prob > 0.7: risk_color = "inverse"
        st.metric("Risk Probability", f"{risk_prob:.1%}")

    st.subheader("Analysis Result")
    
    # Progress Bar for Risk
    st.progress(float(risk_prob))
    
    if risk_prob > 0.65:
        st.error(f"🚨 **HIGH RISK ALERT**")
        st.write(f"This item has a **{risk_prob:.1%}** probability of becoming Dead Stock.")
        st.markdown("**Drivers:**")
        if days_until_expiry < 180: st.write("- ⏳ Critical Aging (< 6 months)")
        if source_type == "Sales Order (Return/Cancel)": st.write("- 🔙 Source is a Return/Cancellation")
        if credit_terms > 90: st.write("- 💳 Extended Credit Terms")
        if total_value > 50000: st.write("- 💰 High Value Exposure")
        
        st.info("💡 **Recommendation:** Prioritize liquidation or bundle with fast-moving SKUs immediately.")
        
    elif risk_prob > 0.35:
        st.warning(f"⚠️ **MODERATE RISK**")
        st.write(f"This item shows signs of stagnation ({risk_prob:.1%}). Monitor closely.")
        st.write("💡 **Recommendation:** Do not re-order until stock covers < 2 months.")
        
    else:
        st.success(f"✅ **SAFE STOCK**")
        st.write(f"This item is healthy ({risk_prob:.1%}). Normal procurement operations applies.")

else:
    st.info("👈 Please adjust parameters in the sidebar and click **Run Risk Analysis**.")