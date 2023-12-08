import streamlit as st
import gradio as gr
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import os

HF_TOKEN = os.environ.get('HF_TOKEN')

model_checkpoint = "besijar/dspa_review_classification"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_auth_token=HF_TOKEN)
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, use_auth_token=HF_TOKEN)

def review_classify(review):
    encoded_review = tokenizer.encode(review, return_tensors="tf")
    prediction = model.predict(encoded_review)
    predicted_class = int(prediction.logits.argmax())
    return predicted_class

# Streamlit app
st.title("Review Classification App")

review_input = st.text_area("Enter your review:")

if st.button("Classify"):
    if review_input:
        predicted_class = review_classify(review_input)
        st.success(f"The predicted class for the review is: {predicted_class}")
    else:
        st.warning("Please enter a review for classification")
