import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ===== 1. ตั้งค่าหน้าเว็บ (Modern Configuration) =====
st.set_page_config(page_title="AI Price Prediction", page_icon="🤖", layout="wide")

# ===== 2. CSS สไตล์ Premium Dark & Clean =====
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0e1117; }
    
    /* การ์ดหลัก */
    .main-card {
        background: #1a1c24;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #343a40;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 25px;
    }
    
    /* ปุ่ม Execute */
    div.stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white; border: none; border-radius: 12px;
        height: 50px; font-size: 18px; font-weight: 600; width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4); }

    /* แสดงราคาประเมิน */
    .price-display {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 25px;
        padding: 40px;
        text-align: center;
        border: 1px solid #334155;
        margin-top: 20px;
    }
    .price-val { color: #38bdf8; font-size: 80px; font-weight: 800; letter-spacing: -2px; }
    .price-unit { color: #94a3b8; font-size: 30px; margin-left: 10px; }
    
    /* สถานะการวิเคราะห์ */
    .status-badge {
        padding: 8px 16px; border-radius: 50px; font-weight: 600; font-size: 14px;
        display: inline-block; margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== 3. โหลดโมเดล (มีระบบเช็ค Error) =====
@st.cache_resource
def load_model():
    model_path = 'mercari_price_model.pkl'
    if not os.path.exists(model_path):
        return None, "File not found"
    try:
        model = joblib.load(model_path)
        return model, "Success"
    except Exception as e:
        return None, str(e)

pipeline, status_msg = load_model()

# ===== 4. ส่วนหัวเว็บ =====
st.markdown("<h1 style='text-align: center; color: white;'>MERCARI AI PREDICTION</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8;'>ระบบพยากรณ์ราคาสินค้าอัจฉริยะด้วย Machine Learning</p>", unsafe_allow_html=True)

if pipeline is None:
    st.error(f"⚠️ ระบบไม่สามารถโหลดโมเดลได้: {status_msg}")
    st.info("กรุณาตรวจสอบว่ามีไฟล์ 'mercari_price_model.pkl' อยู่ใน GitHub หรือยัง?")
    st.stop()

# ===== 5. หน้าจอรับข้อมูล (Input) =====
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Product Name *", placeholder="ระบุชื่อสินค้า")
        brand_name = st.text_input("Brand", value="Missing")
        condition = st.select_slider("Condition Score", options=[1, 2, 3, 4, 5], value=3)
        
    with col2:
        cat_main = st.text_input("Main Category", placeholder="เช่น Electronics")
        shipping = st.selectbox("Shipping Fee", ["Buyer Pays", "Free Shipping"])
        target_price = st.number_input("Your Target Price (USD)", min_value=0.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("RUN PREDICTION ENGINE")
    st.markdown('</div>', unsafe_allow_html=True)

# ===== 6. แสดงผลลัพธ์ (Output) =====
if submit:
    if not name:
        st.warning("⚠️ โปรดใส่ชื่อสินค้าก่อนทำนายราคา")
    else:
        # Prepare Data
        ship_val = 1 if shipping == "Free Shipping" else 0
        input_data = pd.DataFrame([{
            'name': name, 'item_condition_id': condition, 'shipping': ship_val,
            'brand_name': brand_name, 'cat_main': cat_main if cat_main else "Missing",
            'cat_sub1': "Missing", 'cat_sub2': "Missing"
        }])

        with st.spinner("AI is calculating..."):
            try:
                pred_log = pipeline.predict(input_data)[0]
                pred_price = np.expm1(pred_log)
                
                # แสดงผลลัพธ์
                st.markdown(f"""
                    <div class="price-display">
                        <span style="color: #94a3b8; text-transform: uppercase; letter-spacing: 2px;">Predicted Market Price</span>
                        <div class="price-val"><span style="font-size: 40px;">$</span>{pred_price:,.2f}</div>
                        <p style="color: #64748b;">Approximately {pred_price * 36:,.0f} THB</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # ระบบเปรียบเทียบ (Analysis)
                if target_price > 0:
                    diff = ((target_price - pred_price) / pred_price) * 100
                    if diff > 15:
                        st.warning(f"🚩 **Overpriced by {diff:.1f}%** | ราคาสูงกว่าตลาด อาจขายออกช้า")
                    elif diff < -15:
                        st.success(f"🚀 **Fast Sell Opportunity!** | ราคาถูกกว่าตลาด {abs(diff):.1f}% โอกาสขายได้ไวสูง")
                    else:
                        st.info(f"✅ **Market Competitive** | ราคาของคุณใกล้เคียงกับราคากลาง")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")