import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===== 1. Page Config =====
st.set_page_config(page_title="AI Price Predictor", page_icon="⚡", layout="wide")

# ===== 2. Epic Futuristic CSS =====
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    
    /* Background & Font */
    .stApp {
        background-color: #050505;
        background-image: radial-gradient(circle at 50% 50%, #1a1a2e 0%, #050505 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }

    /* Header Styling */
    .header-box {
        text-align: center;
        padding: 40px 0;
        background: linear-gradient(90deg, transparent, rgba(0, 242, 254, 0.1), transparent);
        border-bottom: 1px solid rgba(0, 242, 254, 0.3);
        margin-bottom: 40px;
    }
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        font-weight: 700;
        color: #00f2fe;
        text-shadow: 0 0 20px rgba(0, 242, 254, 0.6);
        letter-spacing: 5px;
    }

    /* Card Styling */
    .tech-card {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(0, 242, 254, 0.2);
        border-radius: 0px 20px 0px 20px; /* ทรงเหลี่ยมตัดโค้งแบบ Sci-Fi */
        padding: 30px;
        box-shadow: 10px 10px 20px rgba(0,0,0,0.5);
        position: relative;
    }
    .tech-card::before {
        content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 2px;
        background: linear-gradient(90deg, #00f2fe, #4facfe);
    }

    /* Input Styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input {
        background-color: rgba(0, 0, 0, 0.5) !important;
        color: #00f2fe !important;
        border: 1px solid rgba(0, 242, 254, 0.3) !important;
        border-radius: 5px !important;
    }

    /* Button Styling */
    div.stButton > button {
        background: transparent;
        color: #00f2fe;
        border: 2px solid #00f2fe;
        border-radius: 0px;
        padding: 15px 30px;
        font-family: 'Orbitron', sans-serif;
        font-size: 20px;
        width: 100%;
        transition: 0.4s;
        text-transform: uppercase;
        letter-spacing: 3px;
        clip-path: polygon(10% 0, 100% 0, 90% 100%, 0 100%); /* ทรงคางหมูเท่ๆ */
    }
    div.stButton > button:hover {
        background: #00f2fe;
        color: black;
        box-shadow: 0 0 30px rgba(0, 242, 254, 0.8);
    }

    /* Result Box */
    .result-container {
        border-left: 5px solid #00f2fe;
        background: rgba(0, 242, 254, 0.05);
        padding: 30px;
        margin-top: 30px;
        text-align: center;
    }
    .predicted-price {
        font-family: 'Orbitron', sans-serif;
        font-size: 100px;
        color: #00f2fe;
        text-shadow: 0 0 30px rgba(0, 242, 254, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# ===== 3. Model Loader =====
@st.cache_resource
def load_model():
    model_path = 'mercari_price_model.pkl'
    if not os.path.exists(model_path): return None
    return joblib.load(model_path)

pipeline = load_model()

# ===== 4. UI Layout =====
st.markdown("""
    <div class="header-box">
        <div class="main-title">CORE_PREDICTOR_v1.0</div>
        <p style="color: #4facfe; letter-spacing: 2px;">NEURAL NETWORK PRICE ESTIMATION SYSTEM</p>
    </div>
""", unsafe_allow_html=True)

if pipeline is None:
    st.error("SYSTEM ERROR: MODEL_FILE_NOT_FOUND")
    st.stop()

# แบ่ง Zone ด้วย Card
with st.container():
    st.markdown('<div class="tech-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='color: #00f2fe;'>[ DATA_INPUT ]</h3>", unsafe_allow_html=True)
        name = st.text_input("IDENTIFIER (Product Name)", placeholder="Enter object name...")
        brand = st.text_input("MANUFACTURER (Brand)", value="Missing")
        condition = st.select_slider("INTEGRITY_LEVEL (Condition)", options=[1, 2, 3, 4, 5], value=3)
        
    with col2:
        st.markdown("<h3 style='color: #00f2fe;'>[ CLASSIFICATION ]</h3>", unsafe_allow_html=True)
        cat = st.text_input("SECTOR (Main Category)", placeholder="Electronics, Fashion, etc.")
        shipping = st.radio("LOGISTICS", ["Buyer Pays", "Free Shipping"], horizontal=True)
        user_val = st.number_input("BENCHMARK_PRICE (USD)", min_value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("EXECUTE ANALYSIS"):
        # Prediction Logic
        ship_val = 1 if shipping == "Free Shipping" else 0
        input_df = pd.DataFrame([{
            'name': name, 'item_condition_id': condition, 'shipping': ship_val,
            'brand_name': brand, 'cat_main': cat if cat else "Missing",
            'cat_sub1': "Missing", 'cat_sub2': "Missing"
        }])
        
        with st.spinner("SYNCHRONIZING WITH DATABASE..."):
            pred_price = np.expm1(pipeline.predict(input_df)[0])
            
            st.markdown(f"""
                <div class="result-container">
                    <p style="color: #4facfe; font-size: 20px; letter-spacing: 5px;">ESTIMATED_VALUE</p>
                    <div class="predicted-price">${pred_price:,.2f}</div>
                    <p style="color: #e0e0e0;">CURRENCY_CONVERSION: ~{pred_price*36:,.0f} THB</p>
                </div>
            """, unsafe_allow_html=True)
            
            if user_val > 0:
                diff = ((user_val - pred_price) / pred_price) * 100
                st.markdown(f"<p style='text-align: center; font-size: 20px;'>VARIANCE: {diff:+.2f}%</p>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)