from fastapi import FastAPI
from pydantic import BaseModel
# from transformers import BertTokenizer, TFBertForSequenceClassification
import numpy as np

# ---------- LOAD MODEL ----------
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ---------- FASTAPI ----------
app = FastAPI()

class UserInput(BaseModel):
    text: str

# ---------- LABELS ----------
labels = {
    0: "normal",
    1: "mild",
    2: "moderate",
    3: "severe"
}

# ---------- CRISIS CHECK ----------
CRISIS_WORDS = [
    "suicide", "kill myself", "end my life",
    "die", "better off dead", "no reason to live"
]

def crisis_detect(text):
    t = text.lower()
    return any(word in t for word in CRISIS_WORDS)

# ---------- RESPONSES ----------
def response(emotion, crisis=False):
    if crisis:
        return {
            "mode": "CRISIS",
            "reply": (
                "Mujhe bahut afsos hai ki tum aisa feel kar rahe ho 💔\n\n"
                "Tum akela nahi ho. Abhi kisi trusted person se baat karna bahut zaroori hai.\n\n"
                "🇮🇳 India Helpline: 9152987821 (AASRA)\n"
                "Agar possible ho to turant kisi ko phone karo.\n\n"
                "Main yahin hoon, tumhari baat sunne ke liye."
            )
        }

    replies = {
        "normal": "Tum thode stable lag rahe ho 🙂. Agar kuch share karna chaho toh bol sakte ho.",
        "mild": "Lagta hai tum thoda heavy feel kar rahe ho. Dheere dheere baat kar sakte hain.",
        "moderate": "Tum kaafi overwhelmed lag rahe ho 😔. Tumhare emotions valid hain.",
        "severe": "Tum bahut zyada struggle kar rahe ho 💔. Help lena bilkul theek hai."
    }

    return {
        "mode": emotion.upper(),
        "reply": replies[emotion]
    }

# ---------- API ----------
@app.post("/analyze")
def analyze(input: UserInput):
    text = input.text

    if crisis_detect(text):
        return response("severe", crisis=True)

    enc = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding=True,
        max_length=128
    )

    preds = model(enc).logits
    label = int(tf.argmax(preds, axis=1).numpy()[0])

    emotion = labels[label]
    return response(emotion)
