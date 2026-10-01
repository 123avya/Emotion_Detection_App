from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from datetime import datetime
import random
import os

# --- Configuration and Initialization ---
# Download NLTK resources
nltk.download('stopwords', quiet=True)

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- Helper Functions ---
def clean_text(text):
    """Preprocess text exactly as during training"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = [w for w in text.split() if w not in stop_words]
    words = [stemmer.stem(w) for w in words]
    return ' '.join(words)

# Load trained models
with open('naive_bayes_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


app = Flask(__name__)

# NOTE: A secret key is ESSENTIAL for using Flask sessions.
# In a real application, this should be a complex, securely stored value.
app.secret_key = os.environ.get('SECRET_KEY', 'a_secret_key_for_dev_only_12345')

# Emotion-based responses for chatbot (using your existing list)
EMOTION_RESPONSES = {
    'joy': [
        "That's wonderful! 😊 I'm glad you're feeling happy!",
        "Your joy is contagious! Keep spreading that positivity!",
        "It's great to hear you're in such a good mood!"
    ],
    'sadness': [
        "I sense some sadness in your words. Remember, it's okay to feel this way. 💙",
        "I'm here for you. Would you like to talk about what's bothering you?",
        "Tough times don't last, but tough people do. You've got this!"
    ],
    'anger': [
        "I can tell you're upset. Take a deep breath. What's troubling you?",
        "It's okay to feel angry. Let's work through this together.",
        "Your feelings are valid. Would it help to talk about what's frustrating you?"
    ],
    'fear': [
        "I sense some worry in your message. You're not alone in this. 🤝",
        "Fear is natural, but you're stronger than you think!",
        "What's making you anxious? Sometimes talking helps."
    ],
    'love': [
        "How beautiful! Love is such a powerful emotion! ❤️",
        "That's so heartwarming! Love makes the world better.",
        "Your affection shines through your words!"
    ],
    'surprise': [
        "Oh wow! That sounds unexpected! 😮",
        "Life is full of surprises! Tell me more!",
        "That's quite surprising! How do you feel about it?"
    ]
}

def get_emotion_response(emotion):
    """Get a contextual response based on detected emotion"""
    return random.choice(EMOTION_RESPONSES.get(emotion, ["I understand how you're feeling."]))

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def home():
    """Initializes and displays the chat history."""
    if 'conversation_history' not in session:
        session['conversation_history'] = [{
            'sender': 'bot',
            'text': "Hello! I'm your emotion detection assistant. Type anything and I'll analyze the emotion in your message. Try expressing how you feel today!",
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }]
    
    # Handle POST request for prediction within the home route for cleaner interaction
    if request.method == 'POST':
        return predict() 
        
    return render_template('index.html', history=session.get('conversation_history', []))

@app.route('/predict', methods=['POST'])
def predict():
    """Handles emotion prediction requests and updates history."""
    text = request.form.get('text', '').strip()
    
    # Initialize history if not present (shouldn't happen if they start at home)
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    # 1. Add user message to history
    user_message = {
        'sender': 'user',
        'text': text,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }
    session['conversation_history'].append(user_message)
    session.modified = True # Tell Flask the session data was changed

    if not text:
        # If no text, return to home with error message (though HTML form required=true should prevent this)
        error_message = "Please enter some text to analyze."
        session['conversation_history'].append({'sender': 'error', 'text': error_message})
        session.modified = True
        return render_template('index.html', history=session['conversation_history'])


    # --- Prediction Logic ---
    cleaned = clean_text(text)
    
    if not cleaned:
        bot_response = "After processing, no meaningful words remain. Try different text."
        bot_message = {
            'sender': 'bot',
            'text': bot_response,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        session['conversation_history'].append(bot_message)
        session.modified = True
        return render_template('index.html', history=session['conversation_history'])
    
    try:
        X = vectorizer.transform([cleaned])
        
        # Predict
        pred_enc = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_enc])[0]
        
        # Get probabilities for all emotions
        top_preds = []
        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            top_idx = probs.argsort()[::-1]
            confidence = float(probs[top_idx[0]])
            for idx in top_idx:
                lbl = label_encoder.inverse_transform([idx])[0]
                top_preds.append((lbl, float(probs[idx])))
        
        # Get chatbot response
        bot_response = get_emotion_response(pred_label)
        
        # 2. Add bot message with prediction details to history
        bot_message = {
            'sender': 'bot',
            'text': bot_response,
            'prediction_text': pred_label.capitalize(),
            'confidence': f"{confidence*100:.1f}",
            'top_preds': top_preds[:6],
            'cleaned_input': cleaned,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        session['conversation_history'].append(bot_message)
        session.modified = True
        
    except Exception as e:
        # Generic error handling for model issues
        error_message = f"An error occurred during prediction: {e}"
        session['conversation_history'].append({'sender': 'error', 'text': error_message})
        session.modified = True

    # Redirect to GET request to clear form data and re-render the page
    return redirect(url_for('home'))

@app.route('/api/predict', methods=['POST'])
# The API route remains the same, as it's not part of the chat interface
def api_predict():
    """API endpoint for programmatic access"""
    # ... (Keep the original api_predict function content here) ...
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    cleaned = clean_text(text)
    if not cleaned:
        return jsonify({'error': 'No meaningful content after preprocessing'}), 400
    
    X = vectorizer.transform([cleaned])
    pred_enc = model.predict(X)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    
    result = {
        'emotion': pred_label,
        'original_text': text,
        'cleaned_text': cleaned
    }
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)[0]
        emotions = {}
        for idx, prob in enumerate(probs):
            lbl = label_encoder.inverse_transform([idx])[0]
            emotions[lbl] = float(prob)
        result['probabilities'] = emotions
    
    return jsonify(result)

@app.route('/reset')
def reset():
    """Clear the conversation history."""
    session.pop('conversation_history', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)