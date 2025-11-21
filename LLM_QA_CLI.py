#!/usr/bin/env python3
"""
LLM_QA_CLI.py — Google Gemini (GenAI SDK) version
"""

import os
import re
import json
from google import genai  # Gemini client

GEMINI_MODEL = "gemini-2.5-flash"  # replace with a model you have access to

def preprocess(text: str):
    """Basic preprocessing: lowercase, remove punctuation, tokenize"""
    lowered = text.lower()
    no_punct = re.sub(r"[^0-9a-zA-Z\s]", "", lowered)
    tokens = no_punct.split()
    return {
        "original": text,
        "lowered": lowered,
        "no_punct": no_punct,
        "tokens": tokens
    }

def build_prompt(processed):
    """Optionally build prompt, can include extra instructions"""
    return f"You are a helpful assistant. Answer clearly.\nQuestion: {processed['original']}"

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")
    
    client = genai.Client(api_key=api_key)
    chat = client.chats.create(model=GEMINI_MODEL)
    
    # Send just the prompt string (Gemini expects str, not dict)
    response = chat.send_message(prompt)
    
    return response.text

def main():
    print("Gemini LLM Q&A CLI\n")
    while True:
        question = input("Question: ").strip()
        if not question:
            continue

        processed = preprocess(question)
        print("\nProcessed Question:")
        print(json.dumps(processed, indent=2))

        prompt = build_prompt(processed)
        print("\nSending to Gemini LLM...")

        try:
            answer = call_gemini(processed["original"])  # only original text sent
            print("\nFinal Answer:\n", answer)
        except Exception as e:
            print(f"\nError calling Gemini LLM: {e}")

if __name__ == "__main__":
    main()
