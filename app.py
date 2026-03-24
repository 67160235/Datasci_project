import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. ตั้งค่าหน้าเว็บ (Page Config) =====
st.set_page_config(page_title="Mercari AI Price", page_icon="⚡", layout="wide")

# ===== 2. CSS ธีมเท่ๆ (Dark/Neon Tech) =====
st.markdown("""
    <style>
    /* นำเข้าฟอนต์ Chakra Petch แนว Sci-Fi / Tech */
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Chakra Petch', sans-serif;
    }
    
    /* ตกแต่งปุ่มกดให้เป็นสไตล์ Neon */
    div.stButton > button {
        background-color: #0d1117;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        border-radius: 8px;
        height: 55px;
        font-size: 20px;
        font-weight: 600;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 0 10px rgba(0, 255, 204, 0.2);
    }
    div.stButton > button:hover {
        background-color: #00ffcc;
        color: #0d1117;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.6);
        transform: translateY(-2px);
    }
    
    /* ตกแต่งกล่องแสดงผลลัพธ์ราคา (Dark Card) */
    .result-card {
        background-color: #161b22;
        padding: 40px 30px; 
        border-radius: 15px; 
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
        text-align: center; 
        border: 1px solid #30363d;
        border-top: 5px solid #ff0055; 
        margin-top: 20px;
    }
    .price-text {
        color: #ff0055; 
        font-size: 70px; 
        font-weight: 700; 
        margin: 10px 0; 
        text-shadow: 0 0 15px rgba(255,0,85,0.4);
    }
    .currency { font-size: 35px; color: #8b949e; font-weight: 500; }
    
    /* ตกแต่งกล่องวิเคราะห์ */
    .analysis-box {
        padding: 20px 25px; 
        border-radius: 10px; 
        margin-top: 20px; 
        text-align: left; 
        font-weight: 400; 
        background-color: #161b22;
        border: 1px solid #30363d;
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

# ===== 4. แถบเมนูด้านข้าง (Sidebar) =====
with st.sidebar:
    st.title("⚡ AI Core System")
    st.markdown("ระบบวิเคราะห์ราคาสินค้าอัจฉริยะ ประมวลผลจาก Big Data")
    st.markdown("---")
    st.markdown("### ⚙️ SYSTEM BOOT")
    st.markdown("1. ป้อนข้อมูลพารามิเตอร์สินค้า\n2. ระบุราคา Target Price\n3. เริ่มรันระบบ\n4. ตรวจสอบ Status")
    st.markdown("---")
    st.info("💡 **TIP:** หากไม่ได้ตั้ง Target Price ระบบจะข้ามโหมดเปรียบเทียบไป")

# ===== 5. ส่วนหัวของแอป =====
st.markdown("<h1 style='text-align: center;'>💻 MERCARI AI : PRICE PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8b949e; font-size: 18px;'>[ SYSTEM ONLINE ] ประเมินราคาด้วย Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

if pipeline is None:
    st.error("⚠️ SYSTEM ERROR: ไม่พบไฟล์ 'mercari_price_model.pkl' ในโฟลเดอร์")
    st.stop()

# ===== 6. ส่วนฟอร์มกรอกข้อมูล =====
with st.container():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 DATA INPUT")
            name = st.text_input("ชื่อสินค้า (Product Name) *", placeholder="e.g. Apple iPad Pro 12.9")
            brand_name = st.text_input("ชื่อแบรนด์ (Brand)", placeholder="e.g. Apple, Nike (พิมพ์ Missing หากไม่ทราบ)", value="Missing")
            
            condition_mapping = {
                "[1] สภาพใหม่ 100% / ซีลเดิม": 1,
                "[2] สภาพดีมาก / เหมือนใหม่": 2,
                "[3] สภาพใช้งานทั่วไป / มีรอยเล็กน้อย": 3,
                "[4] สภาพพอใช้ / มีตำหนิ": 4,
                "[5] สภาพต้องซ่อมแซม": 5
            }
            condition_selection = st.selectbox("สภาพสินค้า (Condition)", options=list(condition_mapping.keys()))
            item_condition_id = condition_mapping[condition_selection]

        with col2:
            st.markdown("#### 🏷️ CATEGORY & LOGISTICS")
            cat_main = st.text_input("หมวดหมู่หลัก (Main Cat)", placeholder="e.g. Electronics")
            cat_sub1 = st.text_input("หมวดหมู่ย่อย 1 (Sub Cat 1)", placeholder="e.g. Computers & Tablets")
            cat_sub2 = st.text_input("หมวดหมู่ย่อย 2 (Sub Cat 2)", placeholder="e.g. iPad")
            
            shipping_option = st.radio("รูปแบบการจัดส่ง (Shipping)", ["ผู้ซื้อจ่ายค่าส่ง (Buyer Pays)", "จัดส่งฟรี (Free Shipping)"], horizontal=True)
            shipping = 1 if "จัดส่งฟรี" in shipping_option else 0
            
        st.markdown("---")
        st.markdown("#### 🎯 TARGET PRICE")
        user_price = st.number_input("ราคาที่คุณต้องการตั้ง (USD) - ใส่ 0 หากต้องการแค่ราคากลาง", min_value=0.0, step=1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            submit_button = st.form_submit_button("⚡ EXECUTE CALCULATION")

# ===== 7. ประมวลผลและแสดงผลลัพธ์ =====
if submit_button:
    if not name.strip():
        st.error("⚠️ SYSTEM WARNING: กรุณาระบุชื่อสินค้าก่อนรันระบบ")
    else:
        with st.spinner("🔄 INITIALIZING NEURAL NETWORK..."):
            input_df = pd.DataFrame([{
                'name': name,
                'item_condition_id': item_condition_id,
                'shipping': shipping,
                'brand_name': brand_name,
                'cat_main': cat_main if cat_main else "Missing",
                'cat_sub1': cat_sub1 if cat_sub1 else "Missing",
                'cat_sub2': cat_sub2 if cat_sub2 else "Missing"
            }])
            
            try:
                predicted_log_price = pipeline.predict(input_df)[0]
                predicted_price = np.expm1(predicted_log_price)
                if predicted_price < 0: predicted_price = 0
                
                # แสดงการ์ดผลลัพธ์
                st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color: #8b949e; margin-bottom: 0; font-weight: 500;">PREDICTED MARKET VALUE</h3>
                        <div class="price-text"><span class="currency">$</span>{predicted_price:,.2f}</div>
                        <p style="color: #8b949e; font-size: 18px; font-weight: 300;">( ≈ <b>{predicted_price * 36:,.0f} THB</b> )</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # ระบบวิเคราะห์เปรียบเทียบราคา
                if user_price > 0:
                    diff = user_price - predicted_price
                    diff_pct = (diff / predicted_price) * 100 if predicted_price > 0 else 0
                    
                    if diff > (predicted_price * 0.15): 
                        st.markdown(f"""
                            <div class="analysis-box" style="border-left: 5px solid #ffcc00;">
                                <h4 style="margin:0; color:#ffcc00;">STATUS: OVERPRICED ⚠️</h4>
                                <p style="margin-top: 10px; color: #e6edf3;">ตั้งราคา <b>สูงกว่าระบบประเมิน {diff_pct:.1f}%</b></p>
                                <span style="color: #8b949e;">[LOG] สินค้ากลุ่มนี้อาจระบายออกช้า แนะนำให้ใช้กลยุทธ์อัปเกรดรูปภาพหรือโปรโมชั่นประกอบการขาย</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    elif diff < -(predicted_price * 0.15): 
                        st.markdown(f"""
                            <div class="analysis-box" style="border-left: 5px solid #00ffcc;">
                                <h4 style="margin:0; color:#00ffcc;">STATUS: FAST SELL ⚡</h4>
                                <p style="margin-top: 10px; color: #e6edf3;">ตั้งราคา <b>ถูกกว่าระบบประเมิน {abs(diff_pct):.1f}%</b></p>
                                <span style="color: #8b949e;">[LOG] ราคาอยู่ในโซนได้เปรียบ คาดว่าอัลกอริทึมจะช่วยดันยอดขายให้ระบายออกได้อย่างรวดเร็ว</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    else: 
                        st.markdown(f"""
                            <div class="analysis-box" style="border-left: 5px solid #39ff14;">
                                <h4 style="margin:0; color:#39ff14;">STATUS: OPTIMAL ✅</h4>
                                <p style="margin-top: 10px; color: #e6edf3;">ราคา ({user_price:.2f} USD) <b>สอดคล้องกับค่าเฉลี่ยตลาด</b></p>
                                <span style="color: #8b949e;">[LOG] ค่าพารามิเตอร์สมดุล สามารถแข่งขันในตลาดได้ดี และรักษากำไรไว้ได้</span>
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ SYSTEM ERROR: {e}")