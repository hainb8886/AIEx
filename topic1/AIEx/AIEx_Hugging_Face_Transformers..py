#https://huggingface.co/docs/transformers/en/quicktour
#Hugging Face Transformers supports other tasks, such as:
#Translation
#Named Entity Recognition (NER)
#Summarization
#Text-to-Speech (TTS)

#Install
#pip install transformers
#pip install torch
#pip install tensorflow

import tf_keras as keras
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

#Ex: Sentiment Analysis
#This example uses a pre-trained model for sentiment analysis (distilbert-base-uncased-finetuned-sst-2-english).
from transformers import pipeline
# Load the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")
# Input text
texts = [
    "I love using Hugging Face models!",
    "This is the worst experience I've ever had."
]

# Analyze sentiments
results = classifier(texts)

# Display results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Label: {result['label']}, Confidence: {result['score']:.4f}\n")
