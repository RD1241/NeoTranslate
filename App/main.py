from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ==============================
# LOAD MODELS
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "Model")

encoder_model = load_model(os.path.join(MODEL_PATH, "encoder_attention_v3.h5"))
decoder_model = load_model(os.path.join(MODEL_PATH, "decoder_attention_v3.h5"))

with open(os.path.join(MODEL_PATH, "tokenizer_v3.pkl"), "rb") as f:
    tokenizer = pickle.load(f)

print("Models Loaded Successfully")

# ==============================
# GLOBAL VARIABLES
# ==============================

max_len = 35
lstm_units = 256
index_word = {v: k for k, v in tokenizer.word_index.items()}


# ==============================
# DECODE FUNCTION
# ==============================

def decode_sequence(input_text):

    seq = tokenizer.texts_to_sequences([input_text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post")

    enc_out, state_h, state_c = encoder_model.predict(seq, verbose=0)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get("<start>")

    decoded_sentence = ""

    while True:

        output_tokens, h, c = decoder_model.predict(
            [target_seq, enc_out, state_h, state_c],
            verbose=0
        )

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index_word.get(sampled_token_index, "")

        if sampled_word == "<end>" or len(decoded_sentence.split()) > 30:
            break

        decoded_sentence += " " + sampled_word

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        state_h, state_c = h, c

    return decoded_sentence.strip()


# ==============================
# ROUTES
# ==============================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/translate", response_class=HTMLResponse)
async def translate(
    request: Request,
    source_lang: str = Form(...),
    target_lang: str = Form(...),
    input_text: str = Form(...)
):

    # Build input format used during training
    formatted_input = f"<{source_lang}> <to_{target_lang}> {input_text}"

    translation = decode_sequence(formatted_input)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "translation": translation
        }
    )