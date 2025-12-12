# api_fixed.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sqlite3
import json
import pandas as pd  # Needed to read the CSV
from rapidfuzz import process, fuzz
import os
import logging
import random
import re

# Global Memory Storage (Prevents Amnesia)
active_sessions = {}

#--- GLOBAL CONFIGURATION ---
# Paste your new pattern here! 👇
SEVERITY_PATTERN = r"(mild|moderate|severe|sharp|dull|excruciating|bad)|(\d+\s*/\s*10)|(level|score|pain)\s+(\d+)"

# ------------------------------
# Basic logging
# ------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("triage_api")

# Prevent oneDNN warnings on some setups
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

print("🔌 Starting Server...")

# ------------------------------
# 1. LOAD BRAIN & MEDICINES
# ------------------------------
model = None
tokenizer = None
label_encoder = None

try:
    model = tf.keras.models.load_model('triage_brain.keras')
    logger.info("✅ Brain model loaded.")
except Exception as e:
    logger.warning("⚠️ Brain model not found or failed to load: %s", e)

try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    logger.info("✅ Tokenizer loaded.")
except Exception as e:
    logger.warning("⚠️ Tokenizer not found: %s", e)

try:
    with open('label_encoder.pickle', 'rb') as handle:
        label_encoder = pickle.load(handle)
    logger.info("✅ Label encoder loaded.")
except Exception as e:
    logger.warning("⚠️ Label encoder not found: %s", e)

# Load medicines CSV (fallback to small list if not present)
try:
    df_meds = pd.read_csv('medicines.csv')
    MEDICINE_DB = df_meds['medicine_name'].astype(str).tolist()
    logger.info("✅ Loaded %d medicines.", len(MEDICINE_DB))
except Exception as e:
    MEDICINE_DB = ["Paracetamol 500mg", "Dolo 650mg", "Pantoprazole 40mg"]
    logger.warning("⚠️ 'medicines.csv' not found or failed to read: %s. Using fallback list.", e)

# ------------------------------
# 2. DATA MODELS
# ------------------------------
class UserSignup(BaseModel):
    name: str
    age: int
    gender: str
    phone: str

class PatientQuery(BaseModel):
    user_id: int
    text: str
    doctor_id: Optional[int] = None

class MedicineItem(BaseModel):
    name: str
    timing: str
    instruction: str
    duration: str

class DoctorReply(BaseModel):
    case_id: int
    response_type: str  # "MEDICINE" or "QUERY"
    text: str
    prescription: Optional[List[MedicineItem]] = None

# ------------------------------
# 3. HELPER FUNCTIONS
# ------------------------------
def check_negation(text, index):
    """
    Checks the text BEFORE a match to see if the user said 'no'.
    Returns True if negation is found.
    """
    # Look at the 20 chars leading up to the word
    preceding_text = text[max(0, index - 20):index]
    
    # Check for "no", "not", "don't", "without", "denies"
    if re.search(r"\b(no|not|dont|without|denies|never)\b", preceding_text):
        return True
    return False

def get_db_connection():
    # Add timeout so sqlite busy errors wait a bit
    conn = sqlite3.connect('hospital_app.db', timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def generate_patient_sentence(meds: List[MedicineItem]):
    sentences = []
    for m in meds:
        freq_map = {
            "Morning": "once a day (morning)",
            "Night": "once a day (night)",
            "Morning-Night": "2 times a day",
            "Morning-Afternoon-Night": "3 times a day",
            "Once": "once a day",
            "Twice": "twice a day"
        }
        frequency = freq_map.get(m.timing, m.timing)
        # ensure instruction and duration are not empty
        instruction = m.instruction if m.instruction else ""
        duration = m.duration if m.duration else ""
        # Build readable line
        line = f"Take {m.name} {frequency} {instruction} for {duration}.".strip()
        sentences.append(line)
    return "\n".join(sentences)

# ------------------------------
# 4. LOGIC CONFIGURATION
# ------------------------------
SPECIALIST_QUESTIONS = {}

GASTRO_KEYWORDS = {
    "STOMACH": ["vomit", "nausea", "puke", "throw up", "upper", "ulcer", "gastritis", "tummy"],
    "INTESTINES": ["diarrhea", "constipation", "poop", "stool", "bloat", "gas", "cramp", "lower"],
    "ESOPHAGUS": ["heartburn", "acid", "reflux", "chest", "burn", "swallow", "gerd"],
    "GENERAL": ["poison", "food", "ate", "sushi", "bad food", "flu"]
}

SPECIALIST_QUESTIONS = {
    "GASTROINTESTINAL": {
        "STOMACH": {
            "duration": [
                "How long have you had this stomach pain?",
                "When did the pain start?",
                "Has this been hurting for a while or did it just start?"
            ],
            "vomiting": [
                "Have you experienced any vomiting or nausea?",
                "Have you thrown up at all?",
                "Is there any nausea?"
            ],
            "severity": [
                "Does the pain get worse after eating?",
                "On a scale of 1-10, how severe is it?",
                "Is the pain sharp or dull?"
            ]
        },
        "INTESTINES": {
            "bowel": [
                "Have you noticed any changes in your bowel movements (diarrhea or constipation)?",
                "Is everything normal when you go to the bathroom?"
            ],
            "bloating": [
                "Is there any bloating or gas?",
                "Do you feel unusually full or bloated?"
            ],
            "cramps": [
                "On a scale of 1-10, how severe are the cramps?",
                "Are the cramps coming and going?"
            ]
        },
        "ESOPHAGUS": {
            "swallowing": [
                "Do you have difficulty swallowing?",
                "Does it hurt when you swallow?"
            ],
            "heartburn": [
                "Do you experience burning in the chest after meals?",
                "Does antacid relieve the burning sensation?"
            ],
            "regurgitation": [
                "Do you feel any acid or food coming up into your throat?",
                "Do you have a sour taste in your mouth?"
            ]
        },
        "GENERAL": {
            "food_history": [
                "Did you eat anything unusual or uncooked recently?",
                "Any recent travel or food exposures?"
            ],
            "fever": [
                "Do you have fever or chills?",
                "Have you noticed any sweating or night fevers?"
            ]
        },

        "DEFAULT": { # Fallback logic if keywords are found but don't match specific sub-cats
            "assessment": [ 
                "Where exactly is the pain located?",
                "How long have you felt this way?",
                "On a scale of 1-10, how severe is it?"
            ]
        }
    }
}

NEURO_KEYWORDS = {
    "HEADACHE": [
        "headache", "migraine", "head", "temple", "throb", "pounding", 
        "forehead", "skull", "pressure", "ache"
    ],
    "DIZZINESS": [
        "dizzy", "spin", "spinning", "lightheaded", "woozy", 
        "balance", "steady", "unsteady", "vertigo", "fall"
    ],
    "VISION": [
        "vision", "blur", "blurry", "double", "see", "sight", 
        "flash", "spots", "blind", "eye"
    ],
    "CONSCIOUSNESS": [
        "faint", "black out", "blackout", "passed out", "unconscious", 
        "seizure", "fit", "convulsion", "wake up"
    ],
    # "GENERAL" acts as a catch-all for other neuro symptoms not listed above
    "GENERAL": [
        "numb", "numbness", "tingle", "tingling", "weak", "weakness", 
        "confused", "confusion", "slur", "speak", "stroke", "face"
    ]
}

# B. QUESTION GROUPS (The "Checklist") 📋
# Structure: { SUB_GROUP: { SLOT_NAME: [List of Questions] } }
SPECIALIST_QUESTIONS = {
    
    "NEUROLOGICAL": {
        "HEADACHE": {
            "location": [
                "Where exactly is the pain? (Front, back, or just one side?)",
                "Is the pain all over your head or focused in one spot?"
            ],
            "duration": [
                "How long have you had this headache?",
                "When did the pain start?"
            ],
            "associated_symptoms": [
                "Do you feel nauseous or sick to your stomach?",
                "Are you sensitive to bright lights or loud noises right now?"
            ],
            "severity": [
                "On a scale of 1-10, how bad is the pain?",
                "Is it a throbbing pain, a tight band, or a sharp stabbing sensation?"
            ]
        },

        "DIZZINESS": {
            "sensation": [
                "Does the room feel like it is spinning around you (vertigo), or do you just feel lightheaded?",
                "Do you feel like you might pass out, or are you just unsteady on your feet?"
            ],
            "triggers": [
                "Does standing up quickly make it worse?",
                "Does moving your head or rolling over in bed trigger the dizziness?"
            ],
            "ears": [
                "Do you hear any ringing or buzzing in your ears?",
                "Does your ear feel full or blocked?"
            ]
        },

        "VISION": {
            "clarity": [
                "Is your vision blurry, or are you seeing double of everything?",
                "Have you lost vision in any part of your eye (like a curtain coming down)?"
            ],
            "disturbances": [
                "Are you seeing flashing lights, zig-zag lines, or spots?",
                "Do bright lights hurt your eyes?"
            ],
            "onset": [
                "Did this vision change happen suddenly or gradually?",
                "Is it affecting one eye or both eyes?"
            ]
        },

        "CONSCIOUSNESS": {
            "event": [
                "Did you actually lose consciousness (black out) or just feel faint?",
                "Do you remember falling, or did you wake up on the floor?"
            ],
            "warning": [
                "Did you feel anything before it happened, like nausea or sweating?",
                "Were you doing anything specific (like exercising or standing up) when it happened?"
            ],
            "aftermath": [
                "Did you feel confused or extremely tired immediately after waking up?",
                "Did you bite your tongue or lose bladder control?"
            ]
        },

        "GENERAL": { 
            "weakness": [
                "Do you have any numbness or tingling in your face, arms, or legs?",
                "Is one side of your body weaker than the other?"
            ],
            "cognition": [
                "Are you having trouble speaking or understanding words?",
                "Do you feel confused or disoriented?"
            ],
            "history": [
                "Have you ever had a seizure or stroke before?",
                "Are you currently on any medication?"
            ]
        },

        "DEFAULT": { 
            "assessment": [
                "Can you describe the sensation you are feeling in more detail?",
                "When did you first notice this symptom?",
                "Is the symptom constant, or does it come and go?",
                "On a scale of 1-10, how much is this affecting your daily activities?"
            ]
        }
    }
}
# --- LOGIC CONFIGURATION (RESPIRATORY) ---

# A. KEYWORD ROUTER (RESPIRATORY)
RESPIRATORY_KEYWORDS = {
    "BREATHING": [
        "breath", "breathing", "short of breath", "gasp", "wheeze", 
        "tight", "suffocate", "air", "pant", "asthma", "inhaler"
    ],
    "COUGH": [
        "cough", "hacking", "phlegm", "sputum", "mucus", 
        "spit", "blood", "dry cough", "wet cough", "bark"
    ],
    "INFECTION": [
        "pneumonia", "bronchitis", "fever", "chills", "flu", 
        "cold", "chest infection", "painful breath"
    ],
    # "GENERAL" acts as a catch-all for vague chest issues or allergies
    "GENERAL": [
        "chest", "congestion", "stuff", "stuffed", "allergy", 
        "sneeze", "runny nose", "nose", "sinus"
    ]
}

# B. QUESTION GROUPS (RESPIRATORY)
# Structure: { SUB_GROUP: { SLOT_NAME: [List of Questions] } }

SPECIALIST_QUESTIONS["RESPIRATORY"] = {
    "BREATHING": { # Covers Shortness of Breath, Asthma, Wheezing
        "onset": [
            "Did this shortness of breath start suddenly or has it been getting worse over time?",
            "Does it happen when you are resting, or only when you exert yourself (like walking)?"
        ],
        "severity": [
            "On a scale of 1-10, how difficult is it to breathe right now?",
            "Do you feel like you can't get enough air into your lungs?"
        ],
        "sounds": [
            "Do you hear any wheezing or whistling sounds when you breathe?",
            "Is your chest feeling tight or heavy?"
        ]
    },

    "COUGH": { # Covers Dry/Wet Cough, Hemoptysis (Blood)
        "type": [
            "Is your cough dry and tickly, or are you bringing up mucus (wet cough)?",
            "Does the cough happen more at night or during the day?"
        ],
        "sputum": [ # Critical check for infection or serious issues
            "If you are coughing up mucus, what color is it (clear, yellow, green, or rusty)?",
            "Have you coughed up any blood or pink froth?"
        ],
        "duration": [
            "How long have you had this cough?",
            "Is it getting worse or staying the same?"
        ]
    },

    "INFECTION": { # Covers Pneumonia, Bronchitis, Flu-like chest issues
        "systemic": [
            "Do you have a high fever or chills along with the breathing issues?",
            "Do you feel generally weak or achy?"
        ],
        "pain": [
            "Does it hurt your chest when you take a deep breath (sharp stabbing pain)?",
            "Do your ribs hurt from coughing?"
        ],
        "history": [
            "Have you been around anyone sick recently?",
            "Have you had a cold or flu that moved into your chest?"
        ]
    },

    "GENERAL": { # Fallback for Congestion, Allergies, Vague Chest Discomfort
        "symptoms": [
            "Do you have a runny or blocked nose?",
            "Are your eyes itchy or watery?"
        ],
        "triggers": [
            "Do you notice these symptoms more around pets, dust, or pollen?",
            "Does the weather change make it worse?"
        ],
        "congestion": [
            "Does your chest feel 'full' or congested?",
            "Are you having trouble clearing your throat?"
        ]
    },

    "DEFAULT": { # Fallback logic if keywords are found but don't match specific sub-cats
        "assessment": [
            "Can you describe exactly what you are feeling in your chest?",
            "How long has this been bothering you?",
            "Does anything make it better (like sitting up) or worse (like lying down)?",
            "Have you ever had lung problems before?"
        ]
    }
}

# --- LOGIC CONFIGURATION (ORTHOPEDIC) ---

# A. KEYWORD ROUTER (ORTHOPEDIC)
ORTHO_KEYWORDS = {
    "SPINE_BACK": [
        "back", "spine", "neck", "lumbar", "disc", "sciatica", 
        "tailbone", "vertebrae", "stiff neck"
    ],
    "JOINTS": [
        "knee", "hip", "shoulder", "elbow", "joint", "arthritis", 
        "socket", "rotator cuff", "meniscus"
    ],
    "EXTREMITIES": [
        "hand", "wrist", "finger", "thumb", "foot", "ankle", 
        "toe", "heel", "plantar", "carpal"
    ],
    "TRAUMA": [
        "break", "broken", "fracture", "fall", "fell", "twist", 
        "sprain", "accident", "hit", "crash", "pop", "snap"
    ],
    "GENERAL": [
        "muscle", "ache", "sore", "stiffness", "swelling", 
        "swell", "bruise", "cramp", "knot"
    ]
}

# B. QUESTION GROUPS (ORTHOPEDIC)
SPECIALIST_QUESTIONS["ORTHOPEDIC"] = {
    "SPINE_BACK": { # Covers Herniated Discs, Sciatica, Strains
        "radiation": [ # Critical for nerve involvement
            "Does the pain shoot down your legs or arms?",
            "Do you feel any electric shock sensations traveling away from your back?"
        ],
        "numbness": [ # Red Flag check (Cauda Equina)
            "Do you have any numbness, especially in your groin or buttocks area?",
            "Are you having any trouble controlling your bladder or bowels?"
        ],
        "triggers": [
            "Does it hurt more when you bend forward or lift things?",
            "Is the pain worse in the morning when you wake up?"
        ]
    },

    "JOINTS": { # Covers Arthritis, Bursitis, Wear & Tear
        "stiffness": [
            "Is the joint stiff in the morning? If so, how long does it take to loosen up?",
            "Does the joint feel 'locked' or stuck?"
        ],
        "swelling": [
            "Is the area swollen, red, or hot to the touch?",
            "Can you see fluid buildup around the joint?"
        ],
        "sounds": [
            "Do you hear a clicking, grinding, or popping sound when you move it?",
            "Does it feel like 'bone on bone'?"
        ]
    },

    "TRAUMA": { # Covers Fractures, Sprains, Dislocations
        "mechanism": [
            "How exactly did the injury happen (e.g., fell on outstretched hand, twisted knee)?",
            "Did you hear a loud 'pop' or 'snap' when it happened?"
        ],
        "deformity": [ # Urgency Check
            "Does the limb look bent, crooked, or misshapen?",
            "Is there any bone sticking out or a deep open wound?"
        ],
        "function": [
            "Can you put weight on it or move it at all?",
            "Are you able to walk, or is it too painful?"
        ]
    },

    "EXTREMITIES": { # Covers Carpal Tunnel, Plantar Fasciitis, Ankle Sprains
        "usage": [
            "Does the pain get worse with specific activities like typing or walking?",
            "Is the pain worse when you take your first steps in the morning?"
        ],
        "sensation": [
            "Do you feel pins and needles or numbness in your fingers or toes?",
            "Does it feel like you are walking on a pebble?"
        ]
    },

    "GENERAL": {
        "location": [
            "Where exactly is the pain located?",
            "Is it deep in the bone or more in the muscle?"
        ],
        "impact": [
            "Is this stopping you from working or exercising?",
            "Does the pain wake you up at night?"
        ]
    },

    "DEFAULT": {
        "assessment": [
            "How long have you had this pain?",
            "On a scale of 1-10, how severe is it?",
            "Did you do any heavy lifting or exercise recently?"
        ]
    }
}

# --- LOGIC CONFIGURATION (DERMATOLOGICAL) ---

# A. KEYWORD ROUTER (DERMATOLOGICAL)
DERMA_KEYWORDS = {
    "RASH_ALLERGY": [
        "rash", "hives", "eczema", "redness", "bump", "blister", 
        "spot", "pimple", "acne", "breakout", "psoriasis"
    ],
    "TRAUMA_BURN": [
        "cut", "burn", "bleed", "wound", "scrape", "scald", 
        "fire", "knife", "injury", "tear", "laceration", "hot"
    ],
    "BITES": [
        "bite", "sting", "spider", "bug", "mosquito", "tick", 
        "bee", "wasp", "insect"
    ],
    "GENERAL": [
        "itch", "dry", "peel", "flake", "skin", "scab", 
        "mole", "lump", "growth"
    ]
}

# B. QUESTION GROUPS (DERMATOLOGICAL)
SPECIALIST_QUESTIONS["DERMATOLOGICAL"] = {
    "RASH_ALLERGY": { # Covers Contact Dermatitis, Hives, Infections
        "spread": [
            "Is the rash spreading to other parts of your body?",
            "Is it staying in one spot or moving?"
        ],
        "triggers": [
            "Did you use any new soaps, lotions, or detergents recently?",
            "Have you eaten anything new or been out in the woods?"
        ],
        "sensation": [
           "Does the rash burn, itch, or feel hot to the touch?" 
        ]
    },

    "TRAUMA_BURN": { # Covers Cuts, Burns, Open Wounds
        "depth": [
            "How deep is the wound? Can you see fatty tissue or bone?",
            "For a burn, did the skin blister or turn white/black?"
        ],
        "bleeding": [ # Urgency Check
            "Is it bleeding heavily? Does it stop if you press on it?",
            "Is there any pulsing blood?"
        ],
        "infection_signs": [
            "Do you see any yellow pus or red streaks coming from the wound?",
            "Is the area swollen and painful?"
        ]
    },

    "BITES": { # Covers Insects, Spiders, Ticks
        "appearance": [
            "What does the bite look like? (e.g., Bullseye pattern, two puncture marks)",
            "Is the area around the bite turning black or purple?"
        ],
        "systemic": [ # Anaphylaxis Check
            "Are you having any trouble breathing or swallowing?",
            "Do you feel dizzy or nauseous?"
        ]
    },

    "GENERAL": {
        "duration": [
            "How long have you had this skin issue?",
            "Has it changed in size or color recently?"
        ],
        "location": [
            "Where exactly on your body is the problem?",
            "Is it all over or just in one specific area?"
        ]
    },

    "DEFAULT": {
        "assessment": [
            "Can you describe what the skin looks like right now?",
            "Is it causing you pain or just discomfort?",
            "Have you applied any creams or medications to it?"
        ]
    }
}

# --- LOGIC CONFIGURATION (GENERAL/SYSTEMIC) ---

# A. KEYWORD ROUTER (GENERAL)
GENERAL_KEYWORDS = {
    "SUMMER_HYDRATION": [
        "sun", "heat", "hot", "sweat", "dry", "thirsty", "water", 
        "urine", "pee", "burn", "faint", "dizzy", "dehydrated"
    ],
    "MONSOON_VECTOR": [
        "mosquito", "bite", "shiver", "chill", "cold", "shake", 
        "bone pain", "joint pain", "eye pain", "dengue", "malaria"
    ],
    "WINTER_VIRAL": [
        "flu", "cold", "sneeze", "runny", "nose", "ache", "body pain", 
        "sore throat", "cough", "congestion", "viral"
    ],
    "CHRONIC_METABOLIC": [
        "tired", "fatigue", "weak", "weight", "loss", "gain", "hair", 
        "hungry", "thirst", "sugar", "thyroid", "pale", "sleep"
    ],
    "FLU_SYMPTOMS": [
        "fever", "chill", "shiver", "temperature", "high temp", "sweat", 
        "hot", "cold", "body ache", "muscle ache", "sore body", "weak", "flu", "viral"
    ],
    "FATIGUE": [
        "tired", "fatigue", "exhausted", "low energy", "sleepy", "draining", 
        "lethargic", "no energy", "worn out"
    ],
    "WEIGHT_APPETITE": [
        "weight loss", "weight gain", "lost weight", "thin", "heavy", 
        "no appetite", "hungry", "not eating", "eating too much"
    ]
}

# B. QUESTION GROUPS (GENERAL)
SPECIALIST_QUESTIONS["GENERAL_SYSTEMIC"] = {
    "SUMMER_HYDRATION": { # Covers Heatstroke, UTI, Dehydration
        "intake": [
            "How much water have you been drinking today?",
            "Have you been out in the sun or working in the heat recently?"
        ],
        "urine_output": [ # UTI & Dehydration check
            "is your urine dark yellow or does it burn when you pee?",
            "When was the last time you urinated?"
        ],
        "mental_state": [ # Heatstroke Red Flag
            "Do you feel confused or like you might pass out?",
            "Has the sweating stopped completely (dry skin)?"
        ]
    },

    "MONSOON_VECTOR": { # Covers Dengue, Malaria, Chikungunya
        "fever_pattern": [
            "Does the fever come and go (like every other day), or is it constant?",
            "Do you have severe shivering (rigors) before the fever spikes?"
        ],
        "pain_specifics": [ # Dengue specific
            "Do you have severe pain behind your eyes?",
            "Does it feel like your bones are breaking (severe body ache)?"
        ],
        "bleeding_check": [ # Dengue Hemorrhagic Flag
            "Have you noticed any bleeding from your gums or nose?",
            "Do you see any tiny red spots (rashes) on your skin?"
        ]
    },

    "WINTER_VIRAL": { # Covers Flu, Common Cold
        "respiratory_check": [ # Checking if it's actually Respiratory category
            "Do you have a runny nose or sore throat?",
            "Is there a cough accompanying the body aches?"
        ],
        "severity": [
            "Do you feel like you've been 'hit by a truck' (sudden extreme fatigue)?",
            "Is the fever mild or very high?"
        ]
    },

    "CHRONIC_METABOLIC": { # Covers Diabetes, Thyroid, Anemia
        "weight_energy": [
            "Have you noticed any unexplained weight loss or weight gain?",
            "Do you feel tired even after a full night's sleep?"
        ],
        "classic_signs": [
            "Are you feeling thirsty or hungry all the time?",
            "Are you experiencing hair loss or dry skin?"
        ],
        "timeline": [
            "How long have you been feeling this 'general' weakness?",
            "Has anyone in your family had thyroid or sugar issues?"
        ]
    },

    "FLU_SYMPTOMS": {
            "temperature": [
                "Have you measured your temperature? If so, how high is it?",
                "Do you feel hot or feverish to the touch?"
            ],
            "duration": [
                "How many days have you been feeling this way?",
                "Did these symptoms start suddenly or gradually?"
            ],
            "other_symptoms": [
                "Do you have a cough, sore throat, or runny nose?", 
                "Are you experiencing any other specific pains besides body aches?"
            ]
        },
    "FATIGUE": {
            "sleep": [
                "How have you been sleeping lately? Is it restful?",
                "Are you sleeping more than usual?"
            ],
            "impact": [
                "Is the tiredness affecting your daily work or activities?",
                "Do you feel tired even after waking up?"
            ],
            "duration": [
                "How long have you been feeling this low energy?",
                "Has this been going on for weeks or just a few days?"
            ]
        },
    "WEIGHT_APPETITE": {
            "amount": [
                "Have you noticed a significant change in your weight recently?",
                "Are you eating more or less than usual?"
            ],
            "timeline": [
                "Over what period of time has this change happened?",
                "Did this change happen quickly?"
            ]
        },

    "DEFAULT": { # Fallback for "I just feel sick"
        "assessment": [
            "Can you tell me your main symptom right now?",
            "Do you have a fever? If so, how high is it?",
            "Are you eating and drinking normally?"
        ]
    }
}

def get_best_subgroup(text, category):

    """

    The Mini-Router: Finds the best sub-group (e.g., 'STOMACH') 

    based on the input text and the active category.

    Uses Dynamic Looping to check the configuration dictionaries.

    """

    text = text.lower()
    print(f"   [DEBUG ROUTER] Checking '{text}' inside Category: {category}")
    target_dict = {}



    # 1. Select the correct dictionary based on the category

    if category == "GASTROINTESTINAL":

        target_dict = GASTRO_KEYWORDS

    elif category == "NEUROLOGICAL":

        target_dict = NEURO_KEYWORDS

    elif category == "RESPIRATORY":

        target_dict = RESPIRATORY_KEYWORDS

    elif category == "ORTHOPEDIC":

        target_dict = ORTHO_KEYWORDS

    elif category == "DERMATOLOGICAL":

        target_dict = DERMA_KEYWORDS

    elif category == "GENERAL_SYSTEMIC":

        target_dict = GENERAL_KEYWORDS



    # 2. The Dynamic Loop 🔄

    # This automatically checks every sub-group in the chosen dictionary.

    # Check if dictionary is found
    if not target_dict:
        print("   [DEBUG ROUTER] ⚠️ Empty Dictionary! No keywords found.") # <--- ADD THIS
        return "DEFAULT"

    # The Loop
    for sub_group, keywords in target_dict.items():
        if any(w in text for w in keywords):
            print(f"   [DEBUG ROUTER] ✅ Match Found! Subgroup: {sub_group}") # <--- ADD THIS
            return sub_group

    print("   [DEBUG ROUTER] ❌ No match found. Returning DEFAULT.") # <--- ADD THIS
    return "DEFAULT"

def get_next_question(category: str, sub_group: str, user_text: str, current_clipboard: dict,last_asked_slot=None):
    """
    Acts as the Secretary: updates the clipboard and decides the next question.
    Returns a question string if a slot is missing; returns None when all done.
    """
    # Lowercase text for simple cues
    text_lower = user_text.lower()
    # --- 1. CONTEXT CHECK (The "Human" Logic) 🧠 ---
    # If we just asked a question, and the user answers "no" or "yes", 
    # we assume they are answering that specific question.
    
    # --- 2. CONTEXT AWARENESS (The "YES/NO" Handler) ---
    if last_asked_slot:
        
        # HANDLE "NO" ❌
        if re.search(r"\b(no|nah|not|nope)\b", text_lower):
            
            if last_asked_slot == "severity":
                current_clipboard["severity"] = "mild symptoms"
            
            # [NEW] Handle Triggers (e.g., "Have you eaten new food?" -> "No")
            elif last_asked_slot == "triggers":
                current_clipboard["triggers"] = "no known triggers"

            # [NEW] Handle Sensation (e.g., "Is it hot?" -> "No")
            elif last_asked_slot == "sensation":
                current_clipboard["sensation"] = "no pain or burning"

            # [NEW] Handle Spread (e.g., "Is it spreading?" -> "No")
            elif last_asked_slot == "spread":
                current_clipboard["spread"] = "no spread"

            else:
                # Default behavior: Save what they said (e.g., "No")
                current_clipboard[last_asked_slot] = user_text 
        
        # HANDLE "YES" ✅
        elif re.search(r"\b(yes|yeah|yep|sure)\b", text_lower):
            current_clipboard[last_asked_slot] = user_text

    # Duration: Looks for Number + Unit (Smart Check)
    # Matches: "2 days", "one week", "48 hours"
    # 1. Capture the match object
    match = re.search(r"(\d+|one|two|three|few|several)\s*(day|week|month|hour|min)", text_lower)

    if match:
    # 2. Save ONLY the matching text (e.g., "3 days")
       current_clipboard["duration"] = match.group(0)

    # Severity: Looks for specific descriptive words OR a score
    # Matches: "severe", "8/10", "sharp pain"
    match = re.search(SEVERITY_PATTERN, text_lower) 
    
    if match:
        current_clipboard["severity"] = match.group(0)

    #-----


    # [GASTROINTESTINAL LOGIC] 🍔
    if category == "GASTROINTESTINAL":
        
        # 1. VOMITING (Specific to Gastro)
        match = re.search(r"(vomit|nausea|puke|throw up|queasy|dry heave)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["vomiting"] = prefix + match.group(0)

        # 2. BOWEL MOVEMENTS
        match = re.search(r"(diarrhea|constipation|poop|stool|loose|runny)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["bowel"] = prefix + match.group(0)

        # 3. BLOATING / GAS
        match = re.search(r"(bloat|gas|fart|fullness|air|burp)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["bloating"] = prefix + match.group(0)
            
        # 4. TRIGGERS (Food Poisoning check)
        match = re.search(r"(ate|food|meal|restaurant|spicy|oily|sushi|chicken)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["triggers"] = prefix + match.group(0)

        # 5. STOOL CHARACTERISTICS (Urgency Check)
        match = re.search(r"(blood|red|black|tar|dark stool|coffee)", text_lower)
        if match:
            is_negated = check_negation(text_lower, match.start())
            prefix = "no " if is_negated else ""
            current_clipboard["stool_color"] = prefix + match.group(0)
            
            # Only trigger CRITICAL if NOT negated
            if not is_negated:
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Possible GI Bleed. Urgency set to CRITICAL.")

        # 6. HYDRATION (Dehydration check)
        match = re.search(r"(water|drink|thirsty|dry mouth|pee|urine)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["hydration"] = prefix + match.group(0)

    # [RESPIRATORY LOGIC] 🫁
    elif category == "RESPIRATORY":

        # BREATHING Slots (Context: Resting vs Exertion)
        match = re.search(r"(sudden|slow|gradual|rest|walk|run|exercise|exert)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["onset"] = prefix + match.group(0)

        # BREATHING SOUNDS
        match = re.search(r"(wheeze|whistle|gasp|squeak|noisy|stridor)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["sounds"] = prefix + match.group(0)

        # COUGH TYPE
        match = re.search(r"(dry|wet|hack|tickle|productive|bark)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["type"] = prefix + match.group(0)

        # MUCUS / PHLEGM
        match = re.search(r"(mucus|phlegm|sputum|spit|green|yellow|clear|blood|red|pink)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["sputum"] = prefix + match.group(0)
            
            # MVP FEATURE: Instant Red Flag for Hemoptysis (Check Negation!)
            if re.search(r"(blood|red|pink)", text_lower) and not check_negation(text_lower, match.start()):
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Hemoptysis detected. Urgency set to CRITICAL.")

        # INFECTION Slots (Systemic signs)
        match = re.search(r"(fever|hot|temp|chill|shiver|sweat|ache|weak)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["systemic"] = prefix + match.group(0)

        # CHEST PAIN (Pleuritic)
        match = re.search(r"(hurt|pain|stab|sharp|rib|burn)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["pain"] = prefix + match.group(0)

        # GENERAL/ALLERGY Triggers
        match = re.search(r"(dust|pollen|cat|dog|pet|smoke|weather|season)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["triggers"] = prefix + match.group(0)

        # CONGESTION
        match = re.search(r"(stuff|block|full|clog|drip)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["congestion"] = prefix + match.group(0)

    # [NEUROLOGICAL LOGIC] 🧠
    elif category == "NEUROLOGICAL":
        
        # HEADACHE Slots
        match = re.search(r"(front|back|side|temple|forehead|skull|left|right|spot|all over)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["location"] = prefix + match.group(0)

        match = re.search(r"(nausea|sick|vomit|puke|light|bright|sound|noise|loud|eye hurt)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["associated_symptoms"] = prefix + match.group(0)

        # DIZZINESS Slots
        match = re.search(r"(spin|room|round|lightheaded|woozy|faint|unsteady|balance|fall)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["sensation"] = prefix + match.group(0)

        match = re.search(r"(stand|up|rise|bed|roll|turn|move head|lying)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["triggers"] = prefix + match.group(0)

        match = re.search(r"(ring|buzz|ear|full|pop|muffle|hear|tinnitus)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["ears"] = prefix + match.group(0)

        # VISION Slots
        match = re.search(r"(blur|double|blind|curtain|dark|see|focus|fog)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["clarity"] = prefix + match.group(0)

        match = re.search(r"(flash|spot|zig|zag|line|star|halo|spark)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["disturbances"] = prefix + match.group(0)

        match = re.search(r"(sudden|instant|gradual|slow|woke up)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["onset"] = prefix + match.group(0)

        # CONSCIOUSNESS Slots
        match = re.search(r"(black|faint|pass out|floor|wake|remember|conscious)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["event"] = prefix + match.group(0)

        match = re.search(r"(aura|smell|sweat|hot|nausea|dizzy before)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["warning"] = prefix + match.group(0)

        match = re.search(r"(confuse|tired|sleepy|bite|tongue|wet|urine|sore)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["aftermath"] = prefix + match.group(0)

        # GENERAL NEURO Slots
        match = re.search(r"(numb|tingle|weak|pin|needle|feel|arm|leg|face|droop)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["weakness"] = prefix + match.group(0)

        match = re.search(r"(speak|slur|word|talk|understand|confuse|memory|disorient)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["cognition"] = prefix + match.group(0)

        match = re.search(r"(stroke|seizure|epilepsy|medication|drug|history|before)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["history"] = prefix + match.group(0)

    # [ORTHOPEDIC LOGIC] 🦴
    elif category == "ORTHOPEDIC":

        # SPINE_BACK Slots
        match = re.search(r"(shoot|leg|arm|travel|down|radiate|electric|shock)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["radiation"] = prefix + match.group(0)

        match = re.search(r"(numb|groin|butt|bladder|bowel|toilet|control)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["numbness"] = prefix + match.group(0)

        # JOINTS Slots
        match = re.search(r"(lock|stuck|click|grind|pop|noise|crunch)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["sounds"] = prefix + match.group(0)

        match = re.search(r"(swell|swollen|puff|red|hot|warm|fluid|balloon)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["swelling"] = prefix + match.group(0)

        # TRAUMA Slots
        match = re.search(r"(fall|fell|trip|hit|twist|land|crush|accident)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["mechanism"] = prefix + match.group(0)

        match = re.search(r"(walk|stand|weight|move|step|lift)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["function"] = prefix + match.group(0)

        # [!!!] URGENCY LOGIC
        match = re.search(r"(bent|crooked|shape|bone|sticking|out|deformed|angle)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["deformity"] = prefix + match.group(0)
            
            # MVP FEATURE: Instant Red Flag for Open Fracture
            if re.search(r"(bone sticking|bone out|white|open wound)", text_lower) and not check_negation(text_lower, match.start()):
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Possible Compound Fracture. Urgency set to CRITICAL.")

        # EXTREMITIES Slots
        match = re.search(r"(type|computer|morning|first step|walk|run|shoe)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["usage"] = prefix + match.group(0)

    # [DERMATOLOGICAL LOGIC] 🧴
    elif category == "DERMATOLOGICAL":

        # RASH_ALLERGY Slots
        match = re.search(r"(soap|lotion|food|plant|woods|detergent)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["triggers"] = prefix + match.group(0)

        # SENSATION (The missing block!)
        # matches: itch, itchy, itching, burn, burning, sting, pain
        match = re.search(r"(itch|burn|sting|pain|hot|fire)", text_lower) 
        if match:
            # Check if they said "no itching"
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["sensation"] = prefix + match.group(0)

        match = re.search(r"(spread|move|growing|bigger|body|all over)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["spread"] = prefix + match.group(0)

        # TRAUMA_BURN Slots
        match = re.search(r"(deep|bone|fat|white|charred|blister|open)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["depth"] = prefix + match.group(0)

        # BLEEDING CHECKS
        match = re.search(r"(blood|bleed|gush|soak|pulsing|stop)", text_lower)
        if match:
            is_negated = check_negation(text_lower, match.start())
            prefix = "no " if is_negated else ""
            current_clipboard["bleeding"] = prefix + match.group(0)
            
            # MVP FEATURE: Instant Red Flag for Hemorrhage
            if re.search(r"(gush|won't stop|pulsing|heavy)", text_lower) and not is_negated:
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Severe Bleeding detected. Urgency set to CRITICAL.")

        # INFECTION SIGNS
        match = re.search(r"(pus|yellow|ooze|streak|line|hot|smell)", text_lower)
        if match:
            is_negated = check_negation(text_lower, match.start())
            prefix = "no " if is_negated else ""
            current_clipboard["infection_signs"] = prefix + match.group(0)
            
            # MVP FEATURE: Red Flag for Sepsis (Red Streaks)
            if re.search(r"(streak|line)", text_lower) and not is_negated:
                current_clipboard["URGENCY_OVERRIDE"] = "HIGH"
                print("⚠️ SYSTEM ALERT: Possible Infection Spread (Lymphangitis). Urgency set to HIGH.")

        # BITES Slots
        # Checks for Anaphylaxis (Systemic reaction)
        match = re.search(r"(breath|throat|swallow|dizzy|faint|tongue|swell)", text_lower)
        if match:
            is_negated = check_negation(text_lower, match.start())
            prefix = "no " if is_negated else ""
            current_clipboard["systemic"] = prefix + match.group(0)
            
            # MVP FEATURE: Instant Red Flag for Anaphylaxis
            if not is_negated:
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Possible Anaphylaxis. Urgency set to CRITICAL.")

        # GENERAL Slots
        match = re.search(r"(face|arm|leg|back|hand|foot|stomach)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["location"] = prefix + match.group(0)

    # [GENERAL_SYSTEMIC LOGIC]
    elif category == "GENERAL_SYSTEMIC":
        
        # --- A. STANDARD SLOT FILLING (Upgraded to Regex) 👂 ---
        
        # SUMMER / HYDRATION
        match = re.search(r"(sun|heat|work|outside|sweat|hot|dry|faint|dizzy)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["intake"] = prefix + match.group(0)
        
        match = re.search(r"(urine|pee|bathroom|burn|yellow|dark)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["urine_output"] = prefix + match.group(0)
            
            # Heatstroke Red Flag (Critical)
            if re.search(r"(stopped sweating|no sweat|dry skin|confused)", text_lower) and not check_negation(text_lower, match.start()):
                current_clipboard["URGENCY_OVERRIDE"] = "CRITICAL"
                print("⚠️ SYSTEM ALERT: Possible Heatstroke. Urgency set to CRITICAL.")

        # MONSOON / VECTOR
        match = re.search(r"(fever|temperature|sick|ill|unwell|symptom)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["assessment"] = prefix + match.group(0)

        match = re.search(r"(shiver|chill|shake|cold|night)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["fever_pattern"] = prefix + match.group(0)
        
        match = re.search(r"(eye|bone|joint|break|muscle)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["pain_specifics"] = prefix + match.group(0)

        match = re.search(r"(bleed|gum|nose|spot|rash|red)", text_lower)
        if match:
            is_negated = check_negation(text_lower, match.start())
            prefix = "no " if is_negated else ""
            current_clipboard["bleeding_check"] = prefix + match.group(0)
            
            # Dengue Hemorrhagic Flag
            if "bleed" in text_lower and not is_negated:
                current_clipboard["URGENCY_OVERRIDE"] = "HIGH"

        # WINTER / VIRAL
        match = re.search(r"(nose|throat|sneeze|cough|chest|congestion)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["respiratory_check"] = prefix + match.group(0)

        # CHRONIC / METABOLIC
        match = re.search(r"(weight|thin|fat|loss|gain|sleep|tired)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["weight_energy"] = prefix + match.group(0)
        
        match = re.search(r"(thirsty|hungry|eat|drink|hair|skin)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["classic_signs"] = prefix + match.group(0)
        
        match = re.search(r"(family|mom|dad|genetic|sugar|thyroid)", text_lower)
        if match:
            prefix = "no " if check_negation(text_lower, match.start()) else ""
            current_clipboard["timeline"] = prefix + match.group(0)

         #------------   
        # 1. Check for GASTRO specific keywords
        if re.search(r"(stomach|vomit|puke|diarrhea|nausea|poop)", text_lower):
            current_clipboard["category_redirect"] = "GASTROINTESTINAL"
            return None, "redirect"  # <--- Signal to Main Loop

        # 2. Check for RESPIRATORY specific keywords
        if re.search(r"(wheeze|short of breath|asthma|lung)", text_lower):
            current_clipboard["category_redirect"] = "RESPIRATORY"
            return None, "redirect"

        # 3. Check for NEURO specific keywords
        if re.search(r"(seizure|blind|vision|double|slur)", text_lower):
            current_clipboard["category_redirect"] = "NEUROLOGICAL"
            return None, "redirect"
            
        # 4. Check for DERMA specific keywords
        if re.search(r"(rash|hives|itch|skin|bump)", text_lower):
             current_clipboard["category_redirect"] = "DERMATOLOGICAL"
             return None, "redirect"


    # --- 3. PICK NEXT QUESTION 📋 ---
    required_slots = SPECIALIST_QUESTIONS.get(category, {}).get(sub_group, {})
    
    for slot, question_list in required_slots.items():
        if current_clipboard.get(slot) is None:
            # We found a missing piece of info! 
            # We return the Question AND the Slot Name so we remember it.
            return random.choice(question_list), slot

    return None, None # Signals that we are done

def generate_summary(clipboard, final_category):
    """
    Generates a natural language summary tailored to the specific disease category.
    """
    
    # 1. SAFE EXTRACTION 🛡️
    duration = clipboard.get('duration', 'unknown duration')
    severity = clipboard.get('severity', 'undetermined severity')

    # 2. SELECT TEMPLATE BASED ON CATEGORY 🧠
    
    if final_category == "ORTHOPEDIC":
        location = clipboard.get('location', 'an extremity')
        mechanism = clipboard.get('mechanism', 'an injury')
        
        summary = f"Patient presents with {location} injury following {mechanism}, reporting {severity} pain."
        
        if "deformity" in clipboard:
            summary += f" Noting {clipboard['deformity']}."
        if "function" in clipboard:
            summary += f" Functionality is {clipboard['function']}."
            
        return summary

    elif final_category == "GASTROINTESTINAL":
        # Collect relevant symptoms specifically for Gastro
        symptoms = []
        if "vomiting" in clipboard: symptoms.append(clipboard['vomiting'])
        if "bowel" in clipboard: symptoms.append(clipboard['bowel'])
        if "bloating" in clipboard: symptoms.append(clipboard['bloating'])
        if "stool_color" in clipboard: symptoms.append(clipboard['stool_color'])
        
        # remove "Filled" placeholders
        clean_symptoms = [s for s in symptoms if s != "Filled"]
        symptoms_text = ", ".join(clean_symptoms) if clean_symptoms else "GI symptoms"
        
        triggers = clipboard.get('triggers', 'unknown causes')
        
        return f"Patient reports {symptoms_text} triggered by {triggers}, persisting for {duration} with {severity}."

    elif final_category == "RESPIRATORY":
        onset = clipboard.get('onset', 'onset')
        cough_type = clipboard.get('type', 'cough')
        sputum = clipboard.get('sputum', 'no sputum')
        
        summary = f"Patient presents with {onset} respiratory symptoms, characterized by {cough_type}."
        
        if "sounds" in clipboard:
            summary += f" Breath sounds described as {clipboard['sounds']}."
        if "sputum" in clipboard and clipboard['sputum'] != "Filled":
             summary += f" Sputum is {sputum}."
             
        return summary

    elif final_category == "NEUROLOGICAL":
        location = clipboard.get('location', 'head/body')
        sensation = clipboard.get('sensation', 'neurological sensation')
        
        summary = f"Patient reports {sensation} involving {location} for {duration}."
        
        if "associated_symptoms" in clipboard:
             summary += f" Associated with {clipboard['associated_symptoms']}."
        if "event" in clipboard: 
             summary += f" Reports consciousness event: {clipboard['event']}."
             
        return summary

    elif final_category == "DERMATOLOGICAL":
        location = clipboard.get('location', 'skin')
        triggers = clipboard.get('triggers', 'unknown triggers')
        
        # --- SMART GRAMMAR FIX ---
        if triggers == "no known triggers":
            # If they said "No", make it a separate, clean sentence.
            summary = f"Patient presents with cutaneous symptoms on {location}. No triggers reported."
        else:
            # If they have a trigger (e.g. "soap"), use the original flow.
            summary = f"Patient presents with cutaneous symptoms on {location} triggered by {triggers}."
        # -------------------------

        if "sensation" in clipboard:
            summary += f" Reports sensation of {clipboard['sensation']}."

        if "spread" in clipboard:
            summary += f" Noting spread: {clipboard['spread']}."
            
        if "infection_signs" in clipboard:
            summary += f" Possible infection signs: {clipboard['infection_signs']}."
            
        return summary

    else:
        # GENERAL / FALLBACK TEMPLATE 📋
        # Collect all symptoms that aren't metadata
        symptoms_list = [
            (v if v != "Filled" else k.replace("_", " ").capitalize()) 
            for k, v in clipboard.items() 
            if k not in ["duration", "severity", "category_redirect", "URGENCY_OVERRIDE"]
        ]
        
        symptoms_text = ", ".join(symptoms_list) if symptoms_list else "general symptoms"
        
        return f"Patient reports {symptoms_text} for {duration} with {severity}."
    

# ------------------------------
# 5. FASTAPI APP & ENDPOINTS
# ------------------------------
app = FastAPI()

@app.post("/register")
def create_user(user: UserSignup):
    conn = get_db_connection()
    try:
        cursor = conn.execute(
            "INSERT INTO users (full_name, age, gender, phone_number) VALUES (?, ?, ?, ?)",
            (user.name, user.age, user.gender, user.phone)
        )
        conn.commit()
        new_id = cursor.lastrowid
        conn.close()
        return {"message": "User Created!", "user_id": new_id}
    except Exception as e:
        conn.close()
        logger.exception("Error creating user")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search_medicine")
def search_medicine(query: str):
    # Fuzzy Search in the CSV list
    try:
        results = process.extract(query, MEDICINE_DB, scorer=fuzz.WRatio, limit=5)
        return {"options": [item[0] for item in results]}
    except Exception as e:
        logger.exception("Error searching medicines")
        raise HTTPException(status_code=500, detail=str(e))

# [IMPORTANT] Add this line at the VERY TOP of your file (under imports)
active_sessions = {} 

@app.post("/predict")
def predict_disease(query: PatientQuery):
    conn = get_db_connection()

    # -------------------------------------------
    # 1. CHECK FOR EXISTING DATABASE LOCKS 🔒
    # -------------------------------------------
    try:
        existing_case = conn.execute('''
            SELECT case_id, status FROM consultations 
            WHERE user_id = ? AND status IN ('PENDING', 'NEEDS_INFO')
            ORDER BY created_at DESC LIMIT 1
        ''', (query.user_id,)).fetchone()
    except Exception as e:
        conn.close()
        logger.exception("DB error checking locks")
        raise HTTPException(status_code=500, detail=str(e))

    # If the case is already sent to the doctor (PENDING), stop the chat.
    if existing_case and existing_case['status'] == 'PENDING':
        conn.close()
        return {"message": "🔒 Chat Locked. Waiting for Doctor Review.", "locked": True}

    # -------------------------------------------
    # 2. MANAGE SESSION (The Memory) 🧠
    # -------------------------------------------
    
    # Check if we already have an active chat session in RAM
    if query.user_id not in active_sessions:
        
        # --- START NEW SESSION ---
        # A. Predict Category (Keyword Heuristics - Faster & More Reliable for Demo)
        text_lower = query.text.lower()
        if any(k in text_lower for k in ("cough", "breath", "chest", "wheeze", "lung")):
            category = "RESPIRATORY"
        elif any(k in text_lower for k in ("head", "dizzy", "migraine", "seizure", "vision", "faint")):
            category = "NEUROLOGICAL"
        elif any(k in text_lower for k in ("skin", "rash", "itch", "blister", "burn", "hives")):
            category = "DERMATOLOGICAL"
        elif any(k in text_lower for k in ("bone", "fracture", "knee", "back", "joint", "swollen")):
            category = "ORTHOPEDIC"
        elif any(k in text_lower for k in ("stomach", "vomit", "puke", "diarrhea", "nausea", "pain")):
            category = "GASTROINTESTINAL"
        else:
            # Fallback for "I feel sick" or unknown symptoms
            category = "GENERAL_SYSTEMIC"

        # B. Initialize Memory
        active_sessions[query.user_id] = {
            "clipboard": {},
            "subgroup": "DEFAULT",
            "last_slot": None,
            "category": category,
            "status": "ACTIVE"
        }
    
    # Load the session
    session = active_sessions[query.user_id]
    
    # Unlock if the doctor requested info (NEEDS_INFO state)
    if existing_case and existing_case['status'] == 'NEEDS_INFO':
         # If doctor asked a question, we just save the user's answer and notify doctor
         try:
            conn.execute(
                "UPDATE consultations SET ai_summary = ?, status = 'PENDING' WHERE case_id = ?",
                (f"PATIENT REPLIED: {query.text}", existing_case['case_id'])
            )
            conn.commit()
            conn.close()
            # Clear session to reset logic
            del active_sessions[query.user_id] 
            return {"message": "✅ Reply sent to doctor! Please wait.", "locked": True}
         except:
             pass

    # Load variables for logic
    current_category = session["category"]
    clipboard = session["clipboard"]
    sub_group = session["subgroup"]
    last_slot = session["last_slot"]

    # -------------------------------------------
    # 3. RUN LOGIC ENGINE ⚙️
    # -------------------------------------------

    # A. Mini-Router (Find Specific Subgroup)
    if sub_group == "DEFAULT":
        new_subgroup = get_best_subgroup(query.text, current_category)
        if new_subgroup != "DEFAULT":
            session["subgroup"] = new_subgroup
            sub_group = new_subgroup
            # Pre-fill slots with the first sentence
            get_next_question(current_category, sub_group, query.text, clipboard, None)

    # B. Get the Next Question
    next_q, slot = get_next_question(
        current_category, 
        sub_group, 
        query.text, 
        clipboard, 
        last_slot
    )

    # C. Handle Redirects (Silent Switch)
    # Example: User says "rash" while in Gastro mode
    if slot == "redirect":
        new_cat = clipboard["category_redirect"]
        session["category"] = new_cat
        current_category = new_cat
        
        # Router for new category
        new_sub = get_best_subgroup(query.text, new_cat)
        session["subgroup"] = new_sub
        sub_group = new_sub
        
        # Re-process input and get new question
        get_next_question(new_cat, sub_group, query.text, clipboard, None)
        next_q, slot = get_next_question(new_cat, sub_group, "", clipboard, None)

    # -------------------------------------------
    # 4. DECIDE OUTPUT 📤
    # -------------------------------------------

    # SCENARIO A: More Questions to Ask
    if next_q:
        session["last_slot"] = slot # Remember what we asked
        conn.close()
        return {"message": next_q, "locked": False}

    # SCENARIO B: Diagnosis Complete (Finish)
    else:
        # 1. Generate Summary
        summary = generate_summary(clipboard, current_category)
        
        # 2. Assign Specialist
        specialist_map = {
            "GASTROINTESTINAL": "Gastroenterologist",
            "NEUROLOGICAL": "Neurologist",
            "RESPIRATORY": "Pulmonologist",
            "ORTHOPEDIC": "Orthopedist",
            "DERMATOLOGICAL": "Dermatologist",
            "GENERAL_SYSTEMIC": "General Physician"
        }
        specialist = specialist_map.get(current_category, "General Physician")
        
        # 3. Save to Real Database
        try:
            conn.execute(
                "INSERT INTO consultations (user_id, ai_summary, predicted_category, urgency_score, doctor_assigned, status, created_at) VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
                (query.user_id, summary, current_category, "Normal", specialist, "PENDING")
            )
            conn.commit()
            
            # Clean up memory since case is closed
            del active_sessions[query.user_id]
            
        except Exception as e:
            logger.exception("DB error saving consultation")
            conn.close()
            raise HTTPException(status_code=500, detail=str(e))

        conn.close()
        return {
            "message": f"Diagnosis Complete. I have sent your report to the {specialist}.",
            "locked": True
        }
    
@app.post("/doctor_reply")
def doctor_reply(reply: DoctorReply):
    conn = get_db_connection()

    try:
        if reply.response_type == "MEDICINE":
            new_status = "COMPLETED"
            raw_json = json.dumps([item.dict() for item in (reply.prescription or [])])
            readable_text = generate_patient_sentence(reply.prescription or [])
            final_message = readable_text
        else:
            new_status = "NEEDS_INFO"
            final_message = reply.text

        conn.execute(
            "UPDATE consultations SET doctor_response = ?, status = ? WHERE case_id = ?",
            (final_message, new_status, reply.case_id)
        )
        conn.commit()
    except Exception as e:
        logger.exception("Error in doctor_reply")
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))

    conn.close()
    return {"message": "Reply sent!"}

@app.get("/check_status/{user_id}")
def check_status(user_id: int):
    conn = get_db_connection()
    try:
        case = conn.execute(
            "SELECT status, doctor_response, predicted_category FROM consultations WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        ).fetchone()
    finally:
        conn.close()

    if not case:
        return {"status": "NO_CASES"}
    return {"status": case['status'], "doctor_response": case['doctor_response'], "disease": case['predicted_category']}

@app.get("/doctors")
def get_doctors():
    conn = get_db_connection()
    try:
        doctors_data = conn.execute("SELECT doctor_id, name, specialty FROM doctors").fetchall()
        doctors_list = [{"id": row["doctor_id"], "name": row["name"], "specialty": row["specialty"]} for row in doctors_data]
        return doctors_list
    except Exception as e:
        logger.exception("Error fetching doctors")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

# --- ADD THIS TO API.PY ---

# --- UPDATE THIS FUNCTION IN API.PY ---

@app.get("/pending_cases")
def get_pending_cases():
    conn = get_db_connection()
    try:
        # Fetch status so we can show "Completed" or "Pending" in the UI
        rows = conn.execute('''
            SELECT case_id, user_id, ai_summary, predicted_category, created_at, status 
            FROM consultations 
            ORDER BY created_at DESC LIMIT 20
        ''').fetchall()
        
        cases = []
        for r in rows:
            cases.append({
                "case_id": r["case_id"],
                "user_id": r["user_id"],
                "summary": r["ai_summary"],
                "category": r["predicted_category"],
                "status": r["status"], # <--- Added Status
                "time": r["created_at"]
            })
        return cases
    except Exception as e:
        return []
    finally:
        conn.close()

# ------------------------------
# --- . MAIN EXECUTION LOOP 🚀 ---

def main():
    print("----- AI MEDICAL ASSISTANT STARTED -----")
    print("Bot: Hi! I am your AI Medical Assistant. Please describe your symptoms.")
    
    # 1. Initialize State
    # We start everyone in 'General' because we don't know what they have yet.
    current_category = "GENERAL_SYSTEMIC"
    current_subgroup = "DEFAULT" 
    
    # The Clipboard remembers everything the user says across all categories
    clipboard = {} 
    
    # Keeps track of the last slot we tried to fill (for Yes/No answers)
    last_asked_slot = None

    # 2. The Infinite Loop
    while True:
        # A. Get User Input
        try:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Bot: Take care! Exiting system.")
                break
        except EOFError:
            break

        # --- 🟢 NEW CODE: ROUTER LOGIC START 🟢 ---
        # "LOYALTY CHECK": Only run the router if we are still looking for a topic.
        # If we are already in 'FLU_SYMPTOMS', we skip this and stay focused.
        
        if current_subgroup == "DEFAULT": 
            detected_subgroup = get_best_subgroup(user_input, current_category) # Note: arguments swapped correctly now!
            
            if detected_subgroup != "DEFAULT":
                current_subgroup = detected_subgroup
                print(f"   [DEBUG] Sub-Group updated to: {current_subgroup}")
        # --- 🟢 NEW CODE END 🟢 ---

        # B. The "Secretary" Logic (Layer 1)
        # We ask the logic engine what to do next based on our current state
        next_q, slot = get_next_question(
            current_category, 
            current_subgroup, 
            user_input, 
            clipboard, 
            last_asked_slot
        )

        # --- C. THE SILENT SWITCH LOGIC (Layer 3 Magic) 🤫 ---
        if slot == "redirect":
            # 1. Grab the new category found by Layer 1
            new_category = clipboard["category_redirect"]
            print(f"   [DEBUG] System switching from {current_category} -> {new_category}")

            # 2. Update the State
            current_category = new_category
            
            # 3. Use the Mini-Router to find the best subgroup
            current_subgroup = get_best_subgroup(user_input, current_category)
            print(f"   [DEBUG] New Subgroup determined: {current_subgroup}")

            # 4. THE FIX: Re-Process input to fill slots (like 'arm'), THEN ASK NEXT QUESTION!
            
            # First, run logic to FILL slots (ignoring the question it returns)
            get_next_question(current_category, current_subgroup, user_input, clipboard, None)
            
            # Now, actually FETCH the next question to show the user
            next_q, next_slot = get_next_question(current_category, current_subgroup, "", clipboard, None)
            
            if next_q:
                print(f"Bot: {next_q}")
                last_asked_slot = next_slot # Remember what we just asked
            else:
                # If no questions are left, loop back to let the main logic handle completion
                pass 

            # 5. Reset context
            # last_asked_slot = None <--- DELETE THIS LINE (We just set it above!)
            
            # 6. NOW we can loop back. Since we just printed "Bot: ...", 
            # the next thing the user sees is the input prompt.
            continue

        # D. Standard Output
        if next_q:
            # We found a question to ask!
            last_asked_slot = slot
            print(f"Bot: {next_q}")
        else:
            # No more questions returned (None) -> We are done!
            print("\n" + "="*40)
            print("----- ✅ DIAGNOSIS COMPLETE -----")
            
            # 1. Print the Category Info
            print(f"**Category:** {current_category}")
            print(f"**Sub-Group:** {current_subgroup}")
            
            # 2. Generate and Print the Human-Readable Summary
            # This calls the function you just added!
            final_report = generate_summary(clipboard, current_category)
            print(final_report) 
            
            print("="*40 + "\n")
            print("Bot: Thank you. Connecting you to a doctor now...")
            break

# Run the program
# Run the FastAPI Server
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Medical AI Server...")
    # This starts the API that your frontends can talk to
    # 0.0.0.0 means "Listen to everyone on the network"
    uvicorn.run(app, host="127.0.0.1", port=8000)