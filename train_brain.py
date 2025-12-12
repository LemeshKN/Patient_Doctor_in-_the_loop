import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
DATA_FILE = 'synthetic_patient_triage_records.csv' # Your colorful file
MODEL_FILE = 'triage_brain.keras'           # Where we save the brain
TOKENIZER_FILE = 'tokenizer.pickle'         # Where we save the translator
LABEL_ENCODER_FILE = 'label_encoder.pickle' # Where we save the category list

# 1. LOAD DATA 📥
print("Loading data...")
try:
    # We use header=None because your raw text might not have a header
    # If your file HAS a header, change to header=0
    df = pd.read_csv(DATA_FILE, header=0, names=['text', 'age', 'gender', 'category', 'urgency', 'specialist', 'symptoms'])
    
    # Basic cleanup: Drop rows where text or category is missing
    df = df.dropna(subset=['text', 'category'])
    
    print(f"✅ Loaded {len(df)} rows of data.")
    print("Example data:")
    print(df[['text', 'category']].head())

except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit()

# 2. PREPROCESSING (Text -> Numbers) 🔢
print("\nTranslating text to numbers...")

# A. The Targets (Categories)
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(df['category']) # Converts "GASTRO" -> 0, "NEURO" -> 1
num_classes = len(label_encoder.classes_)
print(f"Detected {num_classes} Disease Categories: {label_encoder.classes_}")

# B. The Features (Patient Text)
vocab_size = 1000   # We only keep the top 1000 most common words
max_length = 50     # We cut off sentences longer than 50 words
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"   # If we see a new word, we call it <OOV> (Out of Vocabulary)

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# 3. SPLIT DATA (Train vs Test) ✂️
# We hide 20% of the data to test the student (Model) later
X_train, X_test, y_train, y_test = train_test_split(padded, Y, test_size=0.2, random_state=42)

# 4. BUILD THE BRAIN (Neural Network) 🧠
print("\nBuilding the Model...")
model = tf.keras.Sequential([
    # Layer 1: Embedding (Turns numbers into meaning vectors)
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    
    # Layer 2: Global Average Pooling (Simplifies the data)
    tf.keras.layers.GlobalAveragePooling1D(),
    
    # Layer 3: Hidden Layer (The thinking part)
    tf.keras.layers.Dense(64, activation='relu'),
    
    # Layer 4: Output Layer (The decision)
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5. TRAIN THE BRAIN 🏋️‍♂️
print("\nStarting Training (teaching the model)...")
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), verbose=1)

# 6. SAVE EVERYTHING 💾
print("\nSaving the brain to disk...")
model.save(MODEL_FILE)

with open(TOKENIZER_FILE, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(LABEL_ENCODER_FILE, 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ DONE! Model saved as 'triage_brain.keras'")