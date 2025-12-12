import streamlit as st
import requests
import time

# --- CONFIGURATION ---
# Use 'http://127.0.0.1:8000' for laptop testing
# Use 'http://YOUR_COMPUTER_IP:8000' for phone testing
#API_URL = "http://127.0.0.1:8000" 
API_URL = "https://patient-doctor-in-the-loop.onrender.com"

st.set_page_config(page_title="Health App", page_icon="💬", layout="centered")

# --- INITIALIZE STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I'm your Medical Assistant. How can I help?"}]
if "user_id" not in st.session_state:
    st.session_state.user_id = 1
if "chat_locked" not in st.session_state:
    st.session_state.chat_locked = False
if "last_status" not in st.session_state:
    st.session_state.last_status = "PENDING"

# --- STYLING ---
st.markdown("""
<style>
    /* Hide top header */
    header {visibility: hidden;}
    /* Chat bubbles */
    .stChatMessage { border-radius: 15px; padding: 10px; }
    .stChatMessage[data-testid="stChatMessageUser"] { background-color: #dcf8c6; color: black; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
c1, c2 = st.columns([3, 1])
with c1:
    st.subheader("💬 Health Chat")
with c2:
    # Status Check Button with Notification Logic
    if st.button("🔄 Status"):
        try:
            res = requests.get(f"{API_URL}/check_status/{st.session_state.user_id}")
            if res.status_code == 200:
                data = res.json()
                status = data.get('status')
                
                # POPUP LOGIC 🔔
                if status == "COMPLETED" and st.session_state.last_status != "COMPLETED":
                    st.balloons() # 🎉 Party effect
                    st.toast("✅ New Prescription Received!", icon="💊") # Popup
                    st.session_state.messages.append({"role": "assistant", "content": f"💊 **Doctor's Note:**\n\n{data['doctor_response']}"})
                    st.session_state.last_status = "COMPLETED"
                    
                elif status == "NEEDS_INFO" and st.session_state.last_status != "NEEDS_INFO":
                    st.toast("👨‍⚕️ Doctor sent a question!", icon="❓") # Popup
                    st.session_state.messages.append({"role": "assistant", "content": f"👨‍⚕️ **Doctor asks:** {data['doctor_response']}"})
                    st.session_state.last_status = "NEEDS_INFO"
                    # Unlock chat so patient can reply
                    st.session_state.chat_locked = False 
                    
                elif status == "WAITING_FOR_DOCTOR":
                    st.toast("🕒 Still waiting for doctor...", icon="⏳")
                
                elif status == "COMPLETED":
                     st.info("Treatment already completed.")
            else:
                st.error("Check Failed")
        except:
            st.toast("Offline", icon="❌")

st.divider()

# --- CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- INPUT AREA ---
if not st.session_state.chat_locked:
    if prompt := st.chat_input("Message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            payload = {"user_id": st.session_state.user_id, "text": prompt}
            res = requests.post(f"{API_URL}/predict", json=payload)
            if res.status_code == 200:
                data = res.json()
                bot_msg = data.get("message")
                st.session_state.messages.append({"role": "assistant", "content": bot_msg})
                with st.chat_message("assistant"):
                    st.markdown(bot_msg)
                
                if data.get("locked"):
                    st.session_state.chat_locked = True
                    st.rerun()
        except:
            st.error("Connection Error")

else:
    st.info("🔒 Consultation Closed. Check status for results.")
    if st.button("Start New Consultation"):
        st.session_state.chat_locked = False
        st.session_state.last_status = "PENDING"
        st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! Describe your symptoms."}]
        st.rerun()