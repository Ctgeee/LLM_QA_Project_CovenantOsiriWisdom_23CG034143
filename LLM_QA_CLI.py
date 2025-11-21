#!/usr/bin/env python3
"""
LLM_QA_CLI.py — Google Gemini (GenAI SDK) version

This script implements a Command-Line Interface (CLI) for a Question-and-Answer system
that sends user questions to Google Gemini (GenAI SDK) and retrieves responses.
"""

# -------------------------------
# Import required libraries
# -------------------------------
import os      # For accessing environment variables (to get API key)
import re      # Regular expressions, used for text preprocessing
import json    # To display processed questions in readable JSON format
from google import genai  # Google Gemini client library for interacting with the LLM

# -------------------------------
# Configuration / Constants
# -------------------------------
GEMINI_MODEL = "gemini-2.5-flash"  # The LLM model to use. Replace with a model you have access to.

# -------------------------------
# Function: preprocess
# -------------------------------
def preprocess(text: str):
    """
    Basic preprocessing of the input text:
    - Convert text to lowercase
    - Remove punctuation
    - Tokenize into words
    
    Returns a dictionary with:
    - 'original': original input text
    - 'lowered': lowercase version
    - 'no_punct': lowercase text with punctuation removed
    - 'tokens': list of words (tokens)
    """
    lowered = text.lower()  # convert to lowercase
    no_punct = re.sub(r"[^0-9a-zA-Z\s]", "", lowered)  # remove all non-alphanumeric characters
    tokens = no_punct.split()  # split into individual words
    return {
        "original": text,
        "lowered": lowered,
        "no_punct": no_punct,
        "tokens": tokens
    }

# -------------------------------
# Function: build_prompt
# -------------------------------
def build_prompt(processed):
    """
    Constructs a prompt to send to the LLM.
    Here we simply instruct the model to be a helpful assistant and include
    the original question.
    """
    return f"You are a helpful assistant. Answer clearly.\nQuestion: {processed['original']}"

# -------------------------------
# Function: call_gemini
# -------------------------------
def call_gemini(prompt: str) -> str:
    """
    Sends the user question to the Gemini LLM and returns the model's answer.

    Steps:
    1. Retrieve the API key from environment variables.
    2. Initialize the Gemini client.
    3. Create a chat session using the specified model.
    4. Send the prompt string to the model.
    5. Return the text response.

    Notes:
    - Gemini expects the message as a string, not a dictionary.
    - Raises RuntimeError if the API key is not set.
    """
    api_key = os.getenv("GEMINI_API_KEY")  # get API key from environment variable
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    
    client = genai.Client(api_key=api_key)  # initialize Gemini client
    chat = client.chats.create(model=GEMINI_MODEL)  # create chat session
    
    response = chat.send_message(prompt)  # send prompt to LLM
    return response.text  # return the text response

# -------------------------------
# Function: main
# -------------------------------
def main():
    """
    Main CLI loop:
    1. Print welcome message
    2. Repeatedly ask the user for a question
    3. Preprocess and display the processed question
    4. Send the original question to Gemini
    5. Print the LLM's answer
    """
    print("Gemini LLM Q&A CLI\n")
    
    while True:
        question = input("Question: ").strip()  # get user input and remove leading/trailing spaces
        if not question:
            continue  # skip empty input

        # Preprocess the question for display and debugging
        processed = preprocess(question)
        print("\nProcessed Question:")
        print(json.dumps(processed, indent=2))  # display nicely formatted JSON

        # Build prompt (currently just includes the original question)
        prompt = build_prompt(processed)
        print("\nSending to Gemini LLM...")

        # Call Gemini API and handle errors
        try:
            answer = call_gemini(processed["original"])  # send only the original text
            print("\nFinal Answer:\n", answer)  # display LLM response
        except Exception as e:
            print(f"\nError calling Gemini LLM: {e}")  # display any errors

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    # Only execute main if the script is run directly (not imported)
    main()