from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib as jb
import uvicorn
import re


# Define length of sequence
sequence_length = 3
# Load the trained model
model = load_model('PrepareModel\model.keras')
# Load the tokenizer
tokenizer = jb.load('PrepareModel\\tokenizer.sav')


def preprocess(text):
    # Remove URLs (starting with 'www.' or 'http' or 'https')
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove digits, special characters, quotes, double quotes, carriage returns, and newlines
    text_cleaned = re.sub(r'[^\w\s]|\r|\n|\d|["\']', '', text)
    # Replace multiple spaces with a single space
    text_cleaned = re.sub(r'[\s]+', ' ', text_cleaned)
    return text_cleaned

# Handle text content before feed it into model 
def handle_input(text):
    strip_text = text.strip()
    cleaned_text = preprocess(strip_text)
    return cleaned_text

# Predictive system to predict one word each time based on last three words
def predict_next_word(text):
    handled_text = text.split(' ')[-sequence_length:]
    sequence = tokenizer.texts_to_sequences([handled_text])
    sequence = pad_sequences(sequence, maxlen = sequence_length, padding = 'pre')
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    
    predicted_word = ''
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    return predicted_word

# Create application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name = "static")
templates = Jinja2Templates(directory="templates")



# Route for the home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for the predict page
@app.post("/predict")
async def predict(request: Request, Text: str = Form(...), N_words: int = Form(...)):
    content = handle_input(Text)
    for _ in range(N_words):
        predicted_word = predict_next_word(content)
        content = content + ' ' + predicted_word
    return templates.TemplateResponse("index.html", {"request": request, "prediction": content})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")





