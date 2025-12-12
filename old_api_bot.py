import google.generativeai as genai
import sqlite3

# --- SETUP 1: The Brain (Gemini) ---
# ⚠️ REPLACE THE TEXT BELOW WITH YOUR ACTUAL API KEY
GOOGLE_API_KEY = "AIzaSyBjTV6E4DwZqOEqGUdNKxJ7Rhnc1u3AT_8"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# --- SETUP 2: The "Cheat Sheet" (System Prompt) ---
# This tells the AI how to behave and when to stop.
SYSTEM_PROMPT = """
You are an experienced medical triage nurse assistant. 
Your goal is to efficiently gather patient information for a doctor.

--- PHASE 1: IMMEDIATE ANALYSIS (Internal Process) ---
1. Analyze the user's input.
2. IF this is the START of the chat and the user mentions a symptom (e.g., "My head hurts"), IMMEDIATELY mark "Chief Complaint" as DONE. DO NOT ask "What is the problem?".
3. Check for these 3 required data points:
   - Chief Complaint (The symptom)
   - Duration (How long?)
   - Severity (Mild, Moderate, Severe, Unbearable)

--- PHASE 2: CONVERSATION FLOW ---
- IF "Chief Complaint" is missing → Ask: "What seems to be the main problem today?"
- IF "Chief Complaint" is known but "Duration" is missing → Ask: "How long have you been feeling this way?"
- IF "Severity" is missing → Ask: "How would you describe the severity: Mild, Moderate, or Severe?"

--- PHASE 3: CLINICAL CHECK (Body Systems) ---
If you have the basic info, ask ONE relevant medical follow-up:
- STOMACH/GI: Ask about "Vomiting", "Diarrhea", or "Last meal".
- HEAD/NEURO: Ask about "Vision changes" or "Light sensitivity".
- CHEST: Ask about "Breathing difficulty".
- INJURY: Ask about "Swelling" or "Movement".
- GENERAL: Ask about "Fever".

--- GUIDELINES ---
1. EMPATHY: Start with a natural phrase like "I see," "That sounds uncomfortable," or "Okay" (Vary your responses).
2. ONE QUESTION RULE: Never ask two questions in one message.
3. HANDLING UNCERTAINTY: If the user says "I don't know" or is unsure, REASSURE them immediately (e.g., "That is okay, the doctor will check that" or "No problem, we can skip that").

--- TERMINATION & SUMMARY ---
Condition 1: Once you have Chief Complaint, Duration, and Severity, output a POLITE CLOSING phrase followed by the tag.
Example: "Understood. I have noted everything for the doctor. [SUMMARY_READY]"

Condition 2: If the user (or system) asks for a "Final Summary", you MUST output a structured report in this format:
"Patient Report:
- Chief Complaint: [Value]
- Duration: [Value]
- Severity: [Value]
- Notes: [Any extra info]"
"""

# --- LOGIC: The Database Function ---
def save_summary_to_db(summary):
    conn = sqlite3.connect('medical_bot.db')
    cursor = conn.cursor()
    
    # We create a new case for "Patient ID 1" (Simulated user)
    cursor.execute('''
        INSERT INTO cases (patient_id, summary, status)
        VALUES (?, ?, ?)
    ''', (1, summary, 'PENDING'))
    
    conn.commit()
    conn.close()
    print("\n✅ Case saved to Database! Doctor notified.")

# --- LOGIC: The Chat Loop ---
def start_chat():
    print("--- Medical Bot Started (Type 'quit' to stop) ---")
    
    # 1. Start the chat with the hidden instructions
    chat = model.start_chat(history=[
        {"role": "user", "parts": SYSTEM_PROMPT},
        {"role": "model", "parts": "Understood. I am ready to start the triage."}
    ])
    
    print("AI: Hello! I am the AI Medical Assistant. What seems to be the problem today?")

    while True:
        # 2. Get User Input
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            break
        try:

            # 3. Send to Gemini
            response = chat.send_message(user_input)
            ai_reply = response.text
        
            # 4. The "Spy" Check (Looking for the Trigger)
            if "[SUMMARY_READY]" in ai_reply:
            # Print the AI's actual polite message (but hide the ugly tag)
                clean_reply = ai_reply.replace("[SUMMARY_READY]", "")
                print(f"AI: {clean_reply}") 
            
                # Ask Gemini to generate the final summary cleanly
                summary_request = chat.send_message("Please generate the final summary now. Do not use the tag.")
                final_summary = summary_request.text
            
                print(f"\n📝 FINAL SUMMARY:\n{final_summary}")
            
                # 5. Save and Exit
                save_summary_to_db(final_summary)
                break
            else:
                # Continue the chat
                print(f"AI: {ai_reply}")
        except Exception as e:

            # --- THE SAFETY NET DEPLOYED 🪂 ---

            # If Google says "429 Resource Exhausted", we catch it here!

            print(f"\n⚠️ The AI is busy (Error: {e}).")

            print("Please wait 30 seconds and try typing your answer again.")

            # We do NOT break the loop, so the user can retry.

if __name__ == "__main__":
    start_chat()