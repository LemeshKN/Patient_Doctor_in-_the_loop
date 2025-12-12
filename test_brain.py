import os
# Hide TensorFlow warning messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences

# 1. LOAD THE BRAIN & TOOLS 🧠
print("Loading the saved brain...")
model = keras.models.load_model('triage_brain.keras')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

print("✅ Brain loaded successfully!")

# 2. THE PREDICTION FUNCTION 🔮
def predict_disease(text):
    # A. Translate text to numbers (using the same Tokenizer as training)
    sequences = tokenizer.texts_to_sequences([text])
    # We use the same max_length=50 as we did in training
    padded = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    
    # B. Ask the Model
    prediction = model.predict(padded, verbose=0)
    
    # C. Decode the answer
    class_index = np.argmax(prediction)     # Which neuron fired the strongest?
    category = label_encoder.inverse_transform([class_index])[0]
    confidence = np.max(prediction) * 100   # How sure is it? (0-100%)
    
    return category, confidence

# 3. INTERACTIVE LOOP 🔁
print("\n--- AI TRIAGE TESTER (Type 'quit' to exit) ---")
print("Try typing symptoms like: 'I have a splitting headache' or 'My knee is swollen'")

while True:
    user_input = input("\nPatient: ")
    if user_input.lower() in ['quit', 'exit']:
        break
    
    # Get Prediction
    category, confidence = predict_disease(user_input)
    
    # Show Result
    print(f"🤖 Brain Prediction: {category}")
    print(f"📊 Confidence: {confidence:.2f}%")