import os
import re
from flask import Flask, render_template, request, flash, redirect, url_for
from google import genai

app = Flask(__name__)
app.secret_key = "dev-key"

GEMINI_MODEL = "gemini-2.5-flash"

def preprocess(text):
    """Lowercase, remove punctuation, tokenize"""
    lowered = text.lower()
    no_punct = re.sub(r"[^0-9a-zA-Z\s]", "", lowered)
    tokens = no_punct.split()
    return {
        "original": text,
        "lowered": lowered,
        "no_punct": no_punct,
        "tokens": tokens
    }

def call_gemini(question: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model=GEMINI_MODEL)
    
    # Only send the original question as string
    response = chat.send_message(question)
    return response.text

@app.route("/", methods=["GET", "POST"])
def index():
    processed = None
    answer = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()

        if not question:
            flash("Please enter a question.", "warning")
            return redirect(url_for("index"))

        processed = preprocess(question)

        try:
            answer = call_gemini(processed["original"])
        except Exception as e:
            flash(f"Error calling Gemini: {e}", "danger")

    return render_template("index.html",
                           question=question,
                           processed=processed,
                           answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
