import gradio as gr
from src.preprocess import segment_clauses
from src.inference import check_risk

def audit_contract(text):
    clauses = segment_clauses(text)
    results = []
    for c in clauses:
        risk = check_risk(c, ["Standard indemnification text...", "Standard termination..."])
        results.append(f"Clause: {c[:50]}... | Result: {risk}")
    return "\n".join(results)

gr.Interface(fn=audit_contract, inputs="textbox", outputs="text").launch()
