import os
import torch
from torchmetrics.text.bleu import BLEUScore
from transformers import GPT2TokenizerFast, pipeline
from kindred_llm import KindredLLM, KindredConfig
from datasets import load_dataset
from crewai import Agent, Task, Crew
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
from openai import OpenAI
import pytesseract
from pytesseract import image_to_string
import asyncio
from gtts import gTTS
import pygame
import speech_recognition as sr
from PIL import Image
import io
import tempfile
import json
import logging
from datetime import datetime
import numpy as np
import time
from deap import base, creator, tools

# Copyright
__copyright__ = "Copyright (c) 2025 Kindred Cox - KAS-System (Kindred Autonomous System)"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
config = KindredConfig()
kindred_model = KindredLLM(config)

checkpoint_path = "checkpoints/kindred-epoch=00-val_loss=2.34.ckpt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    kindred_model.load_state_dict(checkpoint["state_dict"], strict=False)
kindred_model.eval()

# State
orchestrator_state = {
    'revenue': 0.0,
    'portfolio': {'stocks': {}, 'nfts': {}},
    'metrics': {'revenue': [], 'bleu': []},
    'generations': []
}

# Simple LLM call
def generate_text(prompt: str) -> str:
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = kindred_model(inputs)
        logits = outputs["logits"][:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        return tokenizer.decode(next_token, skip_special_tokens=True)

async def simple_llm_call(prompt: str) -> str:
    return generate_text(prompt)

# Tools
async def finance_tool(query: str) -> str:
    try:
        ticker = yf.Ticker(query.split()[-1])
        data = ticker.history(period="1d")
        price = data['Close'].iloc[-1]
        orchestrator_state['portfolio']['stocks'][query] = price
        orchestrator_state['revenue'] += price * 0.05
        orchestrator_state['metrics']['revenue'].append(orchestrator_state['revenue'])
        return f"Bought {query} at ${price:.2f}. Revenue: ${orchestrator_state['revenue']:.2f}"
    except:
        return "Trade failed"

async def image_gen_tool(prompt: str) -> str:
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.images.generate(model="dall-e-3", prompt=prompt, n=1)
        url = response.data[0].url
        orchestrator_state['revenue'] += 5.0
        orchestrator_state['metrics']['revenue'].append(orchestrator_state['revenue'])
        return f"Generated: {url}"
    except:
        return "Image generation failed"

# Orchestration
async def orchestrate_task(goal: str):
    result = await simple_llm_call(f"Complete task: {goal}")
    orchestrator_state['generations'].append(result)
    return result

# Dashboard
def run_dashboard():
    st.set_page_config(page_title="KAS-System", layout="wide")
    st.title("KAS-System Dashboard")

    tab1, tab2, tab3 = st.tabs(["Orchestration", "Multimodal", "Portfolio"])

    with tab1:
        st.header("Task Orchestration")
        goal = st.text_input("Goal", "Trade AAPL")
        if st.button("Execute"):
            with st.spinner("Processing..."):
                result = asyncio.run(orchestrate_task(goal))
                st.success("Complete!")
                st.write(result)

    with tab2:
        st.header("Multimodal")
        uploaded = st.file_uploader("Upload file", type=["pdf", "png", "jpg", "mp4"])
        if uploaded and st.button("Process"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            if uploaded.type.startswith("image"):
                st.image(tmp_path)
                text = image_to_string(Image.open(tmp_path))
                st.write("OCR:", text)
            elif uploaded.type == "application/pdf":
                from pdf2image import convert_from_path
                images = convert_from_path(tmp_path)
                text = "\n".join([image_to_string(img) for img in images])
                st.write("PDF Text:", text[:1000])

    with tab3:
        st.header("Portfolio & Revenue")
        st.metric("Revenue", f"${orchestrator_state['revenue']:.2f}")
        st.json(orchestrator_state['portfolio'])
        if orchestrator_state['metrics']['revenue']:
            df = pd.DataFrame({"Revenue": orchestrator_state['metrics']['revenue']})
            st.line_chart(df)

if __name__ == "__main__":
    run_dashboard()