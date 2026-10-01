# 🎭 Emotion Detection Chatbot

An **Emotion Detection Chatbot** built using **Python, Flask, NLTK, and Machine Learning**.

The application analyzes the emotion expressed in a user's text and identifies emotions such as **Joy, Sadness, Anger, Fear, Love, and Surprise**. After detecting the emotion, the chatbot provides a contextual response and displays the prediction confidence and emotion probabilities.

## ✨ Features

* 🎭 Detects emotions from user text
* 💬 Interactive chatbot interface
* 🤖 Machine Learning-based emotion classification
* 📊 Displays prediction confidence
* 📈 Displays probability for different emotions
* 🧹 Text preprocessing using NLTK
* 🔤 Stopword removal
* 🌱 Porter stemming
* 💾 Maintains conversation history using Flask sessions
* 🔄 New Chat / Reset option
* 🌐 REST API endpoint for emotion prediction
* 📱 Responsive and modern web interface

## 🧠 Supported Emotions

The chatbot can detect the following emotions:

* 😊 Joy
* 😢 Sadness
* 😠 Anger
* 😨 Fear
* ❤️ Love
* 😮 Surprise

## 🛠️ Technologies Used

### Backend

* Python
* Flask
* NLTK
* Scikit-learn
* Pickle

### Frontend

* HTML5
* CSS3
* JavaScript

### Machine Learning

* TF-IDF Vectorizer
* Naive Bayes Classifier
* Label Encoder

## 📂 Project Structure

```text
Emotion-Detection-Chatbot/
│
├── app.py
├── index.html
│
├── naive_bayes_model.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
│
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

> Place `index.html` inside the `templates` folder because Flask uses `render_template()` to load the HTML page.

## ⚙️ How It Works

The application follows these steps:

```text
User enters a message
        ↓
Text preprocessing
        ↓
Remove URLs and HTML tags
        ↓
Remove punctuation and numbers
        ↓
Remove stopwords
        ↓
Apply Porter Stemming
        ↓
TF-IDF Vectorization
        ↓
Naive Bayes Model
        ↓
Emotion Prediction
        ↓
Confidence & Probability Calculation
        ↓
Chatbot Response
```

## 🧹 Text Preprocessing

Before prediction, the user's message is processed using NLTK.

The application:

1. Converts text to lowercase
2. Removes URLs
3. Removes HTML tags
4. Removes punctuation
5. Removes numbers
6. Removes English stopwords
7. Applies Porter stemming

Example:

```text
Original:
"I am really HAPPY about my new job!"

Processed:
"realli happi new job"
```

## 🤖 Machine Learning Model

The project uses a trained **Naive Bayes classifier**.

The text is converted into numerical features using a **TF-IDF Vectorizer**.

The following trained model files are required:

```text
naive_bayes_model.pkl
tfidf_vectorizer.pkl
label_encoder.pkl
```

### Model Files

| File                    | Purpose                                       |
| ----------------------- | --------------------------------------------- |
| `naive_bayes_model.pkl` | Trained emotion classification model          |
| `tfidf_vectorizer.pkl`  | Converts text into TF-IDF features            |
| `label_encoder.pkl`     | Converts encoded labels back to emotion names |

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-detection-chatbot.git
```

### 2. Open the project folder

```bash
cd emotion-detection-chatbot
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install flask nltk scikit-learn
```

Or, if `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask application:

```bash
python app.py
```

The application will run on:

```text
http://127.0.0.1:5000
```

Open the URL in your browser.

## 💬 Example

### User Input

```text
I am very happy today because I got a new job!
```

### Chatbot Output

```text
Detected Emotion: Joy
Confidence: 95.2%
```

The chatbot also displays the probability distribution for the detected emotions.

## 🌐 API

The application provides an API endpoint:

```text
POST /api/predict
```

### Request

Send JSON data:

```json
{
    "text": "I am feeling very happy today"
}
```

### Example Response

```json
{
    "emotion": "joy",
    "original_text": "I am feeling very happy today",
    "cleaned_text": "feel veri happi today",
    "probabilities": {
        "joy": 0.95,
        "sadness": 0.01,
        "anger": 0.01,
        "fear": 0.01,
        "love": 0.01,
        "surprise": 0.01
    }
}
```

The exact probability values depend on the trained model.

## 🔄 New Chat

The **New Chat** button clears the current conversation history.

The reset route is:

```text
/reset
```

## 🔐 Session Management

The chatbot uses **Flask sessions** to maintain conversation history.

Each conversation can contain:

* User messages
* Bot responses
* Detected emotions
* Confidence values
* Emotion probabilities
* Timestamps

## 📊 Emotion Analysis

For predictions where the machine-learning model provides probabilities, the chatbot displays the probability of each detected emotion.

Example:

```text
Joy        ████████████████████ 90%
Love       ██                   5%
Surprise   █                    3%
Sadness    █                    2%
```

## 📋 Requirements

Recommended environment:

```text
Python 3.x
Flask
NLTK
Scikit-learn
```

The application also downloads the NLTK English stopwords resource automatically when it starts.

## ⚠️ Important

The following files are required for the application to work:

```text
naive_bayes_model.pkl
tfidf_vectorizer.pkl
label_encoder.pkl
```

These files contain the trained machine-learning model and preprocessing components.

Make sure they are available in the same project location expected by `app.py`.

## 🔮 Future Improvements

Possible future improvements include:

* 🎤 Voice input
* 🔊 Voice responses
* 👤 User accounts
* 💾 Database-based chat history
* 📱 Improved mobile interface
* 📈 Emotion history charts
* 🤖 More advanced NLP models
* 🌍 Support for multiple languages
* ☁️ Deployment to a cloud platform

## 👩‍💻 Author

**Navya Narayan Gouda**

Developed as a Machine Learning / NLP project using Python and Flask.

## 📜 License

This project is intended for educational and demonstration purposes.
