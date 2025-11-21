import os
import re
from flask import Flask, render_template, request, flash, redirect, url_for
from google import genai  # Google Gemini client library

# -------------------------------
# Initialize Flask application
# -------------------------------
app = Flask(__name__)
app.secret_key = "dev-key"  # Needed for flashing messages (alerts)

# -------------------------------
# LLM configuration
# -------------------------------
GEMINI_MODEL = "gemini-2.5-flash"  # The Gemini model to use. Replace if necessary.

# -------------------------------
# Function: preprocess
# -------------------------------
def preprocess(text):
    """
    Basic text preprocessing:
    1. Convert to lowercase
    2. Remove punctuation (only keep letters, numbers, spaces)
    3. Tokenize into words
    Returns a dictionary containing:
        - 'original': original text
        - 'lowered': lowercase text
        - 'no_punct': lowercase text without punctuation
        - 'tokens': list of words
    """
    lowered = text.lower()  # Convert text to lowercase
    no_punct = re.sub(r"[^0-9a-zA-Z\s]", "", lowered)  # Remove punctuation
    tokens = no_punct.split()  # Split into words
    return {
        "original": text,
        "lowered": lowered,
        "no_punct": no_punct,
        "tokens": tokens
    }

# -------------------------------
# Function: call_gemini
# -------------------------------
def call_gemini(question: str) -> str:
    """
    Send a question to Google Gemini and get the LLM response.

    Steps:
    1. Retrieve GEMINI_API_KEY from environment variables.
       Raises RuntimeError if the key is not set.
    2. Initialize the Gemini client with the API key.
    3. Create a chat session with the specified model.
    4. Send the original question as a string.
    5. Return the LLM's response text.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")  # Ensure API key exists
    
    client = genai.Client(api_key=api_key)  # Initialize Gemini client
    chat = client.chats.create(model=GEMINI_MODEL)  # Create a chat session
    
    response = chat.send_message(question)  # Send only the original text
    return response.text  # Return LLM response

# -------------------------------
# Flask route: index
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Handles requests to the home page.
    GET: Display empty form
    POST: Process the user's question and display LLM response

    Variables:
    - processed: dictionary of processed question
    - answer: LLM response
    - question: original user input
    """
    processed = None
    answer = None
    question = ""

    if request.method == "POST":
        question = request.form.get("question", "").strip()  # Get input from form

        if not question:
            flash("Please enter a question.", "warning")  # Display warning if empty
            return redirect(url_for("index"))

        processed = preprocess(question)  # Preprocess text

        try:
            answer = call_gemini(processed["original"])  # Send original question to LLM
        except Exception as e:
            flash(f"Error calling Gemini: {e}", "danger")  # Display errors

    # Render the template with processed data and answer (if available)
    return render_template("index.html",
                           question=question,
                           processed=processed,
                           answer=answer)

# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    # Only execute when script is run directly, not imported
    app.run(debug=True)  # Start Flask development server in debug mode
