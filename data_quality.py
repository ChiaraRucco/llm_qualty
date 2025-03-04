from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import textstat
from fastapi import FastAPI
import uvicorn
import streamlit as st
import pandas as pd
import torch

# Inizializza FastAPI
app = FastAPI()

# Carica un modello NLP locale per analisi testuale
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def llm_evaluate(text, prompt):
    inputs = tokenizer(prompt + text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    score = torch.nn.functional.softmax(outputs.logits, dim=-1).max().item()
    return score * 100  # Normalizza il punteggio su scala 0-100

# Funzione per calcolare metriche di leggibilità e qualità
def analyze_text_quality(text):
    readability_score = textstat.flesch_reading_ease(text)
    coherence_score = llm_evaluate(text, "Evaluate coherence:")
    
    return {
        "Readability Score": readability_score,
        "Coherence Score": coherence_score,
    }

# Funzione per valutare metriche di qualità dei dati
def evaluate_data_quality(text):
    metrics = {
        "Correctness": llm_evaluate(text, "Evaluate correctness of the text labels and structure:"),
        "Intrinsic Duplication": llm_evaluate(text, "Check for duplicated instances:"),
        "Trustworthiness": llm_evaluate(text, "Assess factual correctness based on source reliability:"),
        "Class Imbalance": llm_evaluate(text, "Analyze class distribution balance:"),
        "Completeness": llm_evaluate(text, "Identify missing values or incomplete data:"),
        "Contextual Comprehensiveness": llm_evaluate(text, "Ensure the dataset covers all relevant aspects:"),
        "Unbiasedness": llm_evaluate(text, "Detect potential biases in data distribution:"),
        "Variety": llm_evaluate(text, "Ensure diversity of examples in training and test datasets:"),
        "Conformity": llm_evaluate(text, "Check adherence to standard formats:"),
        "Representational Consistency": llm_evaluate(text, "Ensure consistent representation across data:"),
        "Accessibility Availability": llm_evaluate(text, "Evaluate access controls and availability:")
    }
    return pd.DataFrame(metrics.items(), columns=["Metric", "Score"])

@app.post("/analyze/")
def analyze_text(data: dict):
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}
    
    result = analyze_text_quality(text)
    dq_metrics = evaluate_data_quality(text)
    return {"quality": result, "data_quality": dq_metrics.to_dict()}

# Streamlit UI
st.title("Text Quality Checker with LLM")
text_input = st.text_area("Enter text to analyze:")
if st.button("Analyze"):
    if text_input:
        result = analyze_text_quality(text_input)
        dq_metrics = evaluate_data_quality(text_input)
        st.write("### Results")
        st.write(f"**Readability Score:** {result['Readability Score']}")
        st.write(f"**Coherence Score:** {result['Coherence Score']}")
        st.write("### Data Quality Metrics")
        st.dataframe(dq_metrics)
    else:
        st.warning("Please enter some text.")

# Avvio server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

