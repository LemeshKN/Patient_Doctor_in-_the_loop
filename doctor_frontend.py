import streamlit as st
import requests
import pandas as pd

# --- CONFIGURATION ---
# Use 'http://127.0.0.1:8000' for laptop testing
# Use 'http://YOUR_IP:8000' for phone testing
API_URL = "http://127.0.0.1:8000" 

st.set_page_config(page_title="Doctor App", page_icon="👨‍⚕️", layout="centered")

# --- INITIALIZE STATE ---
if "cart" not in st.session_state: st.session_state.cart = []
if "selected_case" not in st.session_state: st.session_state.selected_case = None

# --- STYLING (Mobile Friendly) ---
st.markdown("""
<style>
    .stButton button { width: 100%; border-radius: 8px; height: 3em; }
    .status-badge { padding: 5px 10px; border-radius: 10px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ==================================================
# VIEW 1: THE WAITING ROOM (Main Screen)
# ==================================================
if not st.session_state.selected_case:
    st.title("👨‍⚕️ Doctor Dashboard")
    
    # 1. Fetch Patients
    try:
        res = requests.get(f"{API_URL}/pending_cases")
        cases = res.json()
    except:
        st.error("⚠️ Server Connection Error. Is api.py running?")
        cases = []

    st.subheader(f"📋 Patient Queue")
    
    if st.button("🔄 Refresh List"):
        st.rerun()

    st.write("---")

    if not cases:
        st.info("✅ No patients waiting.")
    else:
        for case in cases:
            # Layout: ID | Status | Button
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 2])
                
                # Col 1: ID & Name
                with c1:
                    st.markdown(f"**👤 Patient #{case['user_id']}**")
                    st.caption(f"Case ID: {case['case_id']}")

                # Col 2: Status
                with c2:
                    status = case.get('status', 'PENDING')
                    if status == "PENDING":
                        st.markdown("🔴 **Pending**")
                    elif status == "COMPLETED":
                        st.markdown("🟢 **Done**")
                    elif status == "NEEDS_INFO":
                        st.markdown("🟡 **Replied**")

                # Col 3: Action
                with c3:
                    if status == "COMPLETED":
                        st.button("View", key=f"btn_{case['case_id']}", disabled=True)
                    else:
                        if st.button("Open", key=f"btn_{case['case_id']}", type="primary"):
                            st.session_state.selected_case = case
                            st.session_state.cart = [] 
                            st.rerun()
                st.divider()

# ==================================================
# VIEW 2: TREATMENT ROOM (Patient Selected)
# ==================================================
else:
    patient = st.session_state.selected_case
    
    # Header with Back Button
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("⬅️ Back"):
            st.session_state.selected_case = None
            st.rerun()
    with c2:
        st.subheader(f"Treating Patient #{patient['user_id']}")

    # 1. AI SUMMARY
    with st.expander("📄 **Read AI Diagnosis**", expanded=True):
        st.write(f"**Symptoms:** {patient['summary']}")

    st.write("---")

    # 2. TABS
    tab1, tab2 = st.tabs(["💊 **Prescribe**", "💬 **Chat**"])

    # --- PRESCRIPTION TAB ---
    with tab1:
        st.write("#### Build Prescription")
        
        # A. FUZZY SEARCH MEDICINE 🔍
        # 1. Type query
        search_query = st.text_input("🔍 Search Medicine (Type & Press Enter):", placeholder="e.g. para")
        
        # 2. Fetch options from API
        med_options = ["Type to search..."]
        if len(search_query) >= 2:
            try:
                api_res = requests.get(f"{API_URL}/search_medicine?query={search_query}")
                if api_res.status_code == 200:
                    data = api_res.json()
                    if data.get("options"):
                        med_options = data["options"]
                    else:
                        med_options = ["No matches found"]
            except:
                pass

        # 3. Select Result
        selected_med = st.selectbox("Select Result:", med_options)

        # B. DOSAGE
        c1, c2 = st.columns(2)
        with c1:
            freq = st.selectbox("Frequency", ["Morning", "Night", "Morning-Night", "Morning-Afternoon-Night", "SOS"])
        with c2:
            dur = st.text_input("Duration", "3 days")
        
        instruct = st.selectbox("Instruction", ["After Food", "Before Food", "With Water", "External Use"])

        # C. ADD BUTTON
        if st.button("➕ Add to Prescription", type="primary"):
            if selected_med and "search..." not in selected_med and "found" not in selected_med:
                st.session_state.cart.append({
                    "name": selected_med,
                    "timing": freq,
                    "instruction": instruct,
                    "duration": dur
                })
                st.success(f"Added {selected_med}")
            else:
                st.error("Please select a valid medicine first.")

        # D. REVIEW & SEND
        if st.session_state.cart:
            st.write("---")
            st.write("#### 📝 Prescription Preview")
            
            # Show readable sentence
            for idx, item in enumerate(st.session_state.cart):
                st.info(f"{idx+1}. Take **{item['name']}** ({item['timing']}) {item['instruction']} for {item['duration']}.")
            
            if st.button("🚀 Send to Patient", type="primary"):
                payload = {
                    "case_id": patient['case_id'],
                    "user_id": patient['user_id'],
                    "response_type": "MEDICINE",
                    "text": "Prescription Sent",
                    "prescription": st.session_state.cart
                }
                try:
                    r = requests.post(f"{API_URL}/doctor_reply", json=payload)
                    if r.status_code == 200:
                        st.balloons()
                        st.success("Sent Successfully!")
                        st.session_state.selected_case = None # Go back to queue
                        st.rerun()
                    else:
                        st.error("Failed to send.")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # --- CHAT TAB ---
    with tab2:
        question = st.text_input("Ask question:")
        if st.button("Send Query"):
            if question:
                payload = {
                    "case_id": patient['case_id'],
                    "user_id": patient['user_id'],
                    "response_type": "QUERY",
                    "text": question
                }
                requests.post(f"{API_URL}/doctor_reply", json=payload)
                st.success("Sent!")
                st.session_state.selected_case = None
                st.rerun()