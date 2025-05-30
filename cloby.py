import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
from typing import List, Optional, Dict, Tuple
import hashlib
import gradio as gr
import numpy as np
import re
import json
import requests
from functools import lru_cache

# Constants for Ollama with open-source model
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"  # Recommended small, fast open-source model

class Cloby:
    def _init_(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.personality = {
            "name": "Cloby",
            "role": "AI School Assistant",
            "traits": ["knowledgeable", "friendly", "supportive", "scholarly"],
            "greeting": "Hi! I'm Cloby 📚, your AI school assistant. Ask me anything about studies, exams, research, or your school data!"
        }
        self.conversation_history = []

    def _query_llm(self, prompt: str) -> str:
        import time
        max_retries = 5
        retry_delay = 10  # seconds
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    OLLAMA_API_URL,
                    json={"model": MODEL_NAME, "prompt": prompt, "stream": False},
                    timeout=120
                )
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                print(f"LLM error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return "I'm having trouble accessing my knowledge base right now."

    def _summarize_data(self) -> str:
        if self.df is None:
            return ""
        try:
            total_students = len(self.df)
            class_summary = self.df.groupby('Class')['Result In Percentage'].mean().round(2).to_dict()
            gender_dist = self.df['Gender'].str.lower().str.strip().value_counts().to_dict()
            at_risk = len(self.df[self.df['Result In Percentage'] < 50])
            avg_score = self.df['Result In Percentage'].mean().round(2)
            top_students = self.df.sort_values(by='Result In Percentage', ascending=False).head(5)
            top_summary = top_students[['Class', 'Result In Percentage']].to_dict(orient='records')

            return json.dumps({
                "total_students": total_students,
                "average_score": avg_score,
                "class_averages": class_summary,
                "gender_distribution": gender_dist,
                "at_risk_students": at_risk,
                "top_students": top_summary
            }, indent=2)
        except Exception as e:
            return f"Error summarizing data: {e}"

    def generate_response(self, query: str) -> str:
        try:
            history = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.conversation_history[-3:]])
            data_summary = self._summarize_data()

            prompt = f"""
You are Cloby, a school AI assistant. You support students and teachers with academic help, data insights, and learning tools.

Your abilities:
- Subject support (math, science, literature, etc.)
- Solve homework problems with steps
- Generate quizzes and flashcards
- Create study plans and summaries
- Help with citations and research
- Analyze student data if provided

Data Summary:
{data_summary}

Conversation so far:
{history}

Current Question:
{query}

Instructions:
- Be helpful, respectful, and clear
- Give explanations, not just answers
- Use emojis when helpful (📘📊🧠✅)
</s>
<|assistant|>"""
            response = self._query_llm(prompt)
            self.conversation_history.append((query, response.strip()))
            return response.strip()
        except Exception as e:
            return f"Oops! Something went wrong: {e}"

def process_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna('0')
    numeric_cols = ['Result In Percentage']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df

def create_chat_interface(df: Optional[pd.DataFrame] = None):
    cloby = Cloby(df)
    with gr.Blocks(title="Cloby - Academic AI Assistant") as interface:
        gr.Markdown("# 🤖 Cloby - Your AI Study Buddy")
        gr.Markdown(cloby.personality["greeting"])

        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(label="Ask Anything 📚", placeholder="e.g., Solve x^2 - 5x + 6 = 0 or Create a study plan for Biology.", lines=2)
                submit_btn = gr.Button("Ask Cloby")
            with gr.Column():
                response_output = gr.Textbox(label="Cloby's Response", lines=12, interactive=False)

        def process_query(query):
            if not query.strip():
                return "Please enter a question."
            return cloby.generate_response(query)

        submit_btn.click(fn=process_query, inputs=query_input, outputs=response_output)

    return interface

if _name_ == "_main_":
    try:
        student_data = process_data("Sample Students Data.csv")
        interface = create_chat_interface(student_data)
        interface.launch(share=True)
    except Exception as e:
        print(f"Error: {e}")
