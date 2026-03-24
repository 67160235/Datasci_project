import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. ตั้งค่าหน้าเว็บ (Page Config) =====
st.set_page_config(
    page_title="Mercari Price Predictor",
    page_icon="🛍️",
    layout="wide"
)

# ===== 2. ตกแต่ง UI ด้วย Custom CSS =====
st.markdown("""
    <style>
    div.stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border-radius: 8px;
        border: none;
        height: 55px;
        font-size: 22px;
        font-weight: bold;
        transition: all 0.3s ease;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    .result-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 6px solid #FF6B6B;
        margin-top: 20px;
    }
    .price-text {
        color: #2D3436;
        font-size: 60px;
        font-weight: 800;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== 3. โหลดโมเดล =====
@st.cache_resource
def load_model():
    try:
        return joblib.load('mercari_price_model.pkl')
    except Exception as e:
        return None

pipeline = load_model()

# ===== 4. ส่วนหัวของแอป =====
st.title("🛍️ AI Price Predictor")
st.markdown("##### ระบบผู้ช่วย AI ทำนายราคาตั้งขายสินค้าที่เหมาะสม จากข้อมูลนับล้านรายการ")
st.markdown("---")

if pipeline is None:
    st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ 'mercari_price_model.pkl' อยู่ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# ===== 5. ส่วนกรอกข้อมูล =====
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📦 ข้อมูลหลักของสินค้า")
    name = st.text_input("ชื่อสินค้า (Product Name)", placeholder="เช่น Apple iPad Pro 12.9")
    brand_name = st.text_input("ชื่อแบรนด์ (Brand)", placeholder="เช่น Apple, Nike (หากไม่มีให้พิมพ์ Missing)", value="Missing")
    
    condition_mapping = {
        "1 - ดีมาก (ของใหม่/ยังไม่แกะซีล)": 1,
        "2 - ดี (สภาพเหมือนใหม่)": 2,
        "3 - ปานกลาง (มีรอยใช้งานทั่วไป)": 3,
        "4 - พอใช้ (มีตำหนิชัดเจน)": 4,
        "5 - แย่ (ต้องซ่อมแซม)": 5
    }
    condition_selection = st.selectbox("สภาพสินค้า", options=list(condition_mapping.keys()))
    item_condition_id = condition_mapping[condition_selection]

with col2:
    st.markdown("### 🏷️ หมวดหมู่และการจัดส่ง")
    cat_main = st.text_input("หมวดหมู่หลัก (Main Category)", placeholder="เช่น Electronics, Women")
    cat_sub1 = st.text_input("หมวดหมู่ย่อย 1 (Sub Category 1)", placeholder="เช่น Computers & Tablets")
    cat_sub2 = st.text_input("หมวดหมู่ย่อย 2 (Sub Category 2)", placeholder="เช่น iPad, Laptops")
    
    shipping_option = st.radio("รูปแบบการจัดส่ง (Shipping)", ["ผู้ซื้อเป็นคนจ่ายค่าส่ง", "เราจ่ายค่าส่งเอง (ส่งฟรี)"])
    shipping = 1 if "เราจ่ายค่าส่งเอง" in shipping_option else 0

st.markdown("<br>", unsafe_allow_html=True)

# ===== 6. ปุ่มประมวลผลและแสดงผลลัพธ์ =====
_, btn_col, _ = st.columns([1, 2, 1])

with btn_col:
    predict_btn = st.button("✨ วิเคราะห์ราคาแนะนำ")

if predict_btn:
    if not name.strip():
        st.warning("กรุณากรอก 'ชื่อสินค้า' เพื่อให้ AI วิเคราะห์ข้อมูลครับ")
    else:
        with st.spinner("🤖 AI กำลังประมวลผลเทียบกับฐานข้อมูล..."):
            
            input_df = pd.DataFrame([{
                'name': name,
                'item_condition_id': item_condition_id,
                'shipping': shipping,
                'brand_name': brand_name,
                'cat_main': cat_main if cat_main else "Missing",
                'cat_sub1': cat_sub1 if cat_sub1 else "Missing",
                'cat_sub2': cat_sub2 if cat_sub2 else "Missing"
            }])
            
            predicted_log_price = pipeline.predict(input_df)[0]
            predicted_price = np.expm1(predicted_log_price)
            if predicted_price < 0: predicted_price = 0
            
            st.balloons()
            
            st.markdown(f"""
                <div class="result-card">
                    <h3 style="color: #636e72; margin-bottom: 0;">💰 ราคาที่เหมาะสมสำหรับตั้งขายคือ</h3>
                    <div class="price-text">${predicted_price:,.2f}</div>
                    <p style="color: #b2bec3; font-size: 18px;">(คิดเป็นเงินไทยประมาณ {predicted_price * 36:,.0f} บาท)</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.success(f"💡 **Tip:** หากต้องการระบายของไว ลองตั้งราคาที่ **${predicted_price*0.9:,.2f}** หรือถ้าเผื่อให้ลูกค้าต่อราคา ลองตั้งที่ **${predicted_price*1.15:,.2f}** ครับ")