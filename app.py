import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. ตั้งค่าหน้าเว็บ (Page Config) =====
st.set_page_config(page_title="Mercari Price Predictor", page_icon="🛍️", layout="wide")

# ===== 2. ตกแต่ง UI ด้วย Custom CSS =====
st.markdown("""
    <style>
    div.stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white; border-radius: 8px; border: none; height: 55px; font-size: 22px; font-weight: bold; width: 100%; transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
    }
    .result-card {
        background-color: #ffffff; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; border-top: 6px solid #FF6B6B; margin-top: 20px;
    }
    .price-text {
        color: #2D3436; font-size: 60px; font-weight: 800; margin: 10px 0;
    }
    .analysis-box {
        padding: 20px; border-radius: 10px; margin-top: 15px; font-size: 18px; text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===== 3. โหลดโมเดล =====
@st.cache_resource
def load_model():
    try:
        # โหลดไฟล์ pkl ที่เราเทรนมา
        return joblib.load('mercari_price_model.pkl')
    except Exception as e:
        return None

pipeline = load_model()

# ===== 4. ส่วนหัวของแอป =====
st.title("🛍️ AI Price Predictor")
st.markdown("##### ระบบผู้ช่วย AI ทำนายราคาตั้งขายสินค้าที่เหมาะสม พร้อมวิเคราะห์กลยุทธ์การตั้งราคา")
st.markdown("---")

if pipeline is None:
    st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ 'mercari_price_model.pkl' อยู่ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# ===== 5. ส่วนฟอร์มกรอกข้อมูล =====
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📦 ข้อมูลหลักของสินค้า")
        name = st.text_input("ชื่อสินค้า (Product Name) *", placeholder="เช่น Apple iPad Pro 12.9")
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
        
    st.markdown("---")
    st.markdown("### 🎯 แผนการตั้งราคาของคุณ")
    user_price = st.number_input("💵 ราคาที่คุณต้องการขาย (USD) - ปล่อยเป็น 0 หากต้องการให้ AI แนะนำอย่างเดียว", min_value=0.0, step=1.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button("✨ วิเคราะห์และประเมินราคา")

# ===== 6. ประมวลผลและแสดงผลลัพธ์ =====
if submit_button:
    if not name.strip():
        st.warning("กรุณากรอก 'ชื่อสินค้า' เพื่อให้ AI วิเคราะห์ข้อมูลครับ")
    else:
        with st.spinner("🤖 AI กำลังประมวลผลเทียบกับฐานข้อมูล..."):
            
            # 1. เตรียมข้อมูลให้ตรงกับที่โมเดลเทรนมา (ไม่ต้องมี item_description และเปลี่ยนชื่อ cat)
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
                # 2. ทำนายผลและแปลงค่า Log กลับเป็นราคาจริง
                predicted_log_price = pipeline.predict(input_df)[0]
                predicted_price = np.expm1(predicted_log_price)
                if predicted_price < 0: predicted_price = 0
                
                st.balloons()
                
                # 3. แสดงราคากลางจาก AI
                st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color: #636e72; margin-bottom: 0;">💰 ราคากลางที่ AI แนะนำคือ</h3>
                        <div class="price-text">${predicted_price:,.2f}</div>
                        <p style="color: #b2bec3; font-size: 18px;">(คิดเป็นเงินไทยประมาณ {predicted_price * 36:,.0f} บาท)</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # 4. ระบบวิเคราะห์เปรียบเทียบราคา
                if user_price > 0:
                    diff = user_price - predicted_price
                    diff_pct = (diff / predicted_price) * 100 if predicted_price > 0 else 0
                    
                    if diff > (predicted_price * 0.15): # ผู้ใช้ตั้งแพงกว่า AI 15% ขึ้นไป
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba;">
                                ⚠️ <b>วิเคราะห์กลยุทธ์:</b> คุณตั้งราคา <b>แพงกว่าราคาตลาดประมาณ {diff_pct:.1f}%</b><br>
                                สินค้าอาจจะขายออกได้ช้า แนะนำให้เน้นถ่ายรูปให้สวยและเขียนบรรยายให้เห็นถึงความคุ้มค่าครับ
                            </div>
                        """, unsafe_allow_html=True)
                        
                    elif diff < -(predicted_price * 0.15): # ผู้ใช้ตั้งถูกกว่า AI 15% ขึ้นไป
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb;">
                                ⚡ <b>วิเคราะห์กลยุทธ์:</b> คุณตั้งราคา <b>ถูกกว่าราคาตลาดประมาณ {abs(diff_pct):.1f}%</b><br>
                                เป็นราคาที่ดึงดูดใจมาก! น่าจะขายออกไวแน่นอนครับ (Fast Sell)
                            </div>
                        """, unsafe_allow_html=True)
                        
                    else: # ราคาใกล้เคียงกัน (บวกลบไม่เกิน 15%)
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb;">
                                ✅ <b>วิเคราะห์กลยุทธ์:</b> ราคาที่คุณตั้ง ({user_price:.2f} USD) <b>เหมาะสมและแข่งขันในตลาดได้ดีมากครับ!</b>
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ: {e}")