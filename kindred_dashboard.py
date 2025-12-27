import streamlit as st
import torch
from transformers import GPT2TokenizerFast
from kindred_llm import KindredLLM, KindredConfig
import yfinance as yf
import pytesseract
from PIL import Image
import io
import tempfile
import os
import asyncio
from gtts import gTTS
import pygame
import speech_recognition as sr
import pandas as pd
import plotly.express as px

# Initialize model
@st.cache_resource
def load_model():
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config = KindredConfig(vocab_size=len(tokenizer))
    model = KindredLLM(config)
    checkpoint_path = "checkpoints/kindred-epoch=00-val_loss=2.34.ckpt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# State
if 'revenue' not in st.session_state:
    st.session_state.revenue = 0.0
    st.session_state.portfolio = {'stocks': {}, 'nfts': {}}
    st.session_state.history = []

st.set_page_config(page_title="KAS-System", layout="wide")
st.title("🚀 KAS-System Dashboard")

tab1, tab2, tab3 = st.tabs(["Orchestration", "Multimodal", "Portfolio"])

with tab1:
    st.header("Task Orchestration")
    goal = st.text_input("Enter your goal", "Trade AAPL")
    if st.button("Execute"):
        with st.spinner("Thinking..."):
            # Simple orchestration simulation
            if "trade" in goal.lower():
                ticker = goal.split()[-1]
                data = yf.Ticker(ticker).history(period="1d")
                if not data.empty:
                    price = data['Close'].iloc[-1]
                    st.session_state.portfolio['stocks'][ticker] = price
                    st.session_state.revenue += price * 0.05
                    result = f"Bought {ticker} at ${price:.2f}. Revenue +${price*0.05:.2f}"
                else:
                    result = f"Could not fetch data for {ticker}"
            elif "generate image" in goal.lower():
                result = "Generated futuristic art. Simulated revenue +$5"
                st.session_state.revenue += 5.0
            else:
                result = f"Completed task: {goal}. Simulated analysis and execution."
            st.session_state.history.append(result)
            st.success("Done!")
            st.write(result)

with tab2:
    st.header("Multimodal Input")
    uploaded = st.file_uploader("Upload PDF/Image", type=["pdf", "png", "jpg"])
    if uploaded and st.button("Process"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded.name.split('.')[-1]}") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        if uploaded.type == "application/pdf":
            from pdf2image import convert_from_path
            images = convert_from_path(tmp_path)
            text = "\n".join([pytesseract.image_to_string(img) for img in images])
            st.write("Extracted Text:", text[:1000])
        else:
            img = Image.open(tmp_path)
            st.image(img, caption="Uploaded")
            text = pytesseract.image_to_string(img)
            st.write("OCR Result:", text)

with tab3:
    st.header("Portfolio")
    st.metric("Revenue", f"${st.session_state.revenue:.2f}")
    st.json(st.session_state.portfolio)
    if st.session_state.history:
        df = pd.DataFrame({"Task": range(1, len(st.session_state.history)+1), "Revenue": [st.session_state.revenue] * len(st.session_state.history)})
        fig = px.line(df, x="Task", y="Revenue", title="Revenue Growth")
        st.plotly_chart(fig)

st.success("KAS-System Running! Merry Christmas 2025! 🎄")