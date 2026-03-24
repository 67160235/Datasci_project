import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===== 1. ตั้งค่าหน้าเว็บ (Page Config) =====
st.set_page_config(page_title="Mercari AI Price", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

# ===== 2. CSS ตกแต่งให้สวยหรู =====
st.markdown("""
    <style>
    /* นำเข้าฟอนต์ Kanit จาก Google Fonts ให้ข้อความสวยงาม */
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }
    
    /* ตกแต่งปุ่มกดให้เป็น Gradient พร้อม Effect ตอนชี้เมาส์ */
    div.stButton > button {
        background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
        color: white; border-radius: 12px; border: none; height: 55px; font-size: 20px; font-weight: 600; width: 100%; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(255, 75, 43, 0.4);
    }
    div.stButton > button:hover {
        transform: translateY(-3px); box-shadow: 0 8px 25px rgba(255, 75, 43, 0.6);
    }
    
    /* ตกแต่งกล่องแสดงผลลัพธ์ราคา */
    .result-card {
        background: linear-gradient(to right bottom, #ffffff, #fcfcfc);
        padding: 40px 30px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; border-top: 8px solid #FF416C; margin-top: 20px; position: relative; overflow: hidden;
    }
    .price-text {
        color: #2D3436; font-size: 75px; font-weight: 800; margin: 15px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    }
    .currency { font-size: 35px; color: #636e72; font-weight: 500; }
    
    /* ตกแต่งกล่องวิเคราะห์เปรียบเทียบราคา */
    .analysis-box {
        padding: 20px 25px; border-radius: 12px; margin-top: 20px; text-align: left; font-weight: 400; box-shadow: 0 4px 10px rgba(0,0,0,0.03);
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
    st.title("🛍️ Mercari AI")
    st.markdown("ระบบผู้ช่วยอัจฉริยะที่จะช่วยคุณตั้งราคาสินค้าให้ขายออกไว ได้กำไรดีที่สุด!")
    st.markdown("---")
    st.markdown("### 📌 วิธีใช้งาน")
    st.markdown("1. กรอกข้อมูลสินค้าให้ครบถ้วน\n2. ระบุราคาเป้าหมายที่คุณอยากขาย\n3. กดปุ่มวิเคราะห์\n4. อ่านคำแนะนำจาก AI")
    st.markdown("---")
    st.info("💡 **รู้หรือไม่?** การระบุแบรนด์และหมวดหมู่ให้ชัดเจน จะช่วยให้ AI ทำนายราคาได้แม่นยำขึ้นอย่างมาก")

# ===== 5. ส่วนหัวของแอป =====
st.markdown("<h1 style='text-align: center; color: #2D3436;'>✨ AI Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #636e72; font-size: 18px;'>ประเมินราคาที่เหมาะสมที่สุดสำหรับสินค้าของคุณ ด้วย AI ที่เรียนรู้จากข้อมูลกว่า 1 ล้านรายการ</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ ไม่พบไฟล์โมเดล กรุณาตรวจสอบว่ามีไฟล์ 'mercari_price_model.pkl' อยู่ในโฟลเดอร์เดียวกับ app.py")
    st.stop()

# ===== 6. ส่วนฟอร์มกรอกข้อมูล =====
with st.container():
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📦 ข้อมูลหลักของสินค้า")
            name = st.text_input("ชื่อสินค้า (Product Name) *", placeholder="เช่น Apple iPad Pro 12.9")
            brand_name = st.text_input("ชื่อแบรนด์ (Brand)", placeholder="เช่น Apple, Nike (หากไม่มีให้พิมพ์ Missing)", value="Missing")
            
            condition_mapping = {
                "🌟 1 - ดีมาก (ของใหม่/ยังไม่แกะซีล)": 1,
                "✨ 2 - ดี (สภาพเหมือนใหม่)": 2,
                "👍 3 - ปานกลาง (มีรอยใช้งานทั่วไป)": 3,
                "👌 4 - พอใช้ (มีตำหนิชัดเจน)": 4,
                "🔧 5 - แย่ (ต้องซ่อมแซม)": 5
            }
            condition_selection = st.selectbox("สภาพสินค้า", options=list(condition_mapping.keys()))
            item_condition_id = condition_mapping[condition_selection]

        with col2:
            st.markdown("#### 🏷️ หมวดหมู่และการจัดส่ง")
            cat_main = st.text_input("หมวดหมู่หลัก (Main Category)", placeholder="เช่น Electronics, Women")
            cat_sub1 = st.text_input("หมวดหมู่ย่อย 1 (Sub Category 1)", placeholder="เช่น Computers & Tablets")
            cat_sub2 = st.text_input("หมวดหมู่ย่อย 2 (Sub Category 2)", placeholder="เช่น iPad, Laptops")
            
            shipping_option = st.radio("รูปแบบการจัดส่ง (Shipping)", ["📦 ผู้ซื้อจ่ายค่าส่ง", "🚚 เราจ่ายค่าส่งเอง (ส่งฟรี)"], horizontal=True)
            shipping = 1 if "เราจ่ายค่าส่งเอง" in shipping_option else 0
            
        st.markdown("---")
        st.markdown("#### 🎯 แผนการตั้งราคาของคุณ")
        user_price = st.number_input("💵 ราคาที่คุณต้องการขาย (USD) - ปล่อยเป็น 0 หากต้องการให้ AI แนะนำอย่างเดียว", min_value=0.0, step=1.0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        # จัดปุ่มให้อยู่ตรงกลางแบบสวยๆ
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            submit_button = st.form_submit_button("🚀 วิเคราะห์และประเมินราคา")

# ===== 7. ประมวลผลและแสดงผลลัพธ์ =====
if submit_button:
    if not name.strip():
        st.error("⚠️ กรุณากรอก 'ชื่อสินค้า' เพื่อให้ AI เริ่มการวิเคราะห์ครับ")
    else:
        with st.spinner("🤖 AI กำลังคำนวณและสร้างแผนกลยุทธ์..."):
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
                
                st.balloons()
                
                # แสดงการ์ดผลลัพธ์
                st.markdown(f"""
                    <div class="result-card">
                        <h3 style="color: #FF416C; margin-bottom: 0; font-weight: 600;">💎 ราคากลางที่ AI แนะนำคือ</h3>
                        <div class="price-text"><span class="currency">$</span>{predicted_price:,.2f}</div>
                        <p style="color: #b2bec3; font-size: 20px; font-weight: 400;">(คิดเป็นเงินไทยประมาณ <b>{predicted_price * 36:,.0f} บาท</b>)</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # ระบบวิเคราะห์เปรียบเทียบราคา
                if user_price > 0:
                    diff = user_price - predicted_price
                    diff_pct = (diff / predicted_price) * 100 if predicted_price > 0 else 0
                    
                    if diff > (predicted_price * 0.15): 
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #FFF4E5; border-left: 6px solid #FF8C00;">
                                <h4 style="margin:0; color:#FF8C00; font-size: 22px;">⚠️ กลยุทธ์การขาย: ราคาสูงกว่าตลาด (Premium)</h4>
                                <p style="margin-top: 10px; font-size: 18px;">คุณตั้งราคา <b>แพงกว่าราคาประเมินตลาด {diff_pct:.1f}%</b></p>
                                <span style="font-size:16px; color: #555;">💡 <b>คำแนะนำ:</b> เหมาะสำหรับสินค้าหายาก หรือสภาพสวยเป็นพิเศษ ควรเน้นถ่ายรูปให้ดูแพงและเขียนคำบรรยายให้เห็นถึงความคุ้มค่าครับ หากขายไม่ออกใน 1 สัปดาห์ ค่อยพิจารณาลดราคาลง</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    elif diff < -(predicted_price * 0.15): 
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #E3F2FD; border-left: 6px solid #1976D2;">
                                <h4 style="margin:0; color:#1976D2; font-size: 22px;">⚡ กลยุทธ์การขาย: ขายออกไว (Fast Sell)</h4>
                                <p style="margin-top: 10px; font-size: 18px;">คุณตั้งราคา <b>ถูกกว่าราคาประเมินตลาด {abs(diff_pct):.1f}%</b></p>
                                <span style="font-size:16px; color: #555;">💡 <b>คำแนะนำ:</b> เป็นราคาที่ดึงดูดใจผู้ซื้อมากๆ! โอกาสขายออกไวสูงมาก เหมาะสำหรับคนที่ต้องการระบายของอย่างรวดเร็ว (แต่ถ้าอยากได้กำไรเพิ่ม สามารถขยับราคาขึ้นได้อีกนิดนะครับ)</span>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    else: 
                        st.markdown(f"""
                            <div class="analysis-box" style="background-color: #E8F5E9; border-left: 6px solid #2E7D32;">
                                <h4 style="margin:0; color:#2E7D32; font-size: 22px;">✅ กลยุทธ์การขาย: ราคามาตรฐานตลาด (Market Price)</h4>
                                <p style="margin-top: 10px; font-size: 18px;">ราคาที่คุณตั้ง ({user_price:.2f} USD) <b>เหมาะสมและแข่งขันในตลาดได้ดีเยี่ยม!</b></p>
                                <span style="font-size:16px; color: #555;">💡 <b>คำแนะนำ:</b> เป็นราคาที่สมเหตุสมผล ลูกค้าตัดสินใจซื้อง่าย และคุณยังได้กำไรที่คุ้มค่า เป็นจุด Sweet Spot ที่ดีที่สุดในการตั้งราคาครับ</span>
                            </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการคำนวณ: {e}")