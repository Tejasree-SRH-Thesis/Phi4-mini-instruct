import gradio as gr
import fitz  # PyMuPDF for PDF text extraction
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
import html

# Load Hugging Face token securely
# os.environ["HF_TOKEN"] = "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"
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
# if not hf_token:
#     raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable.")

hf_token = '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'
model_path = hf_hub_download(
    repo_id="MaziyarPanahi/Phi-4-mini-instruct-GGUF",
    filename="Phi-4-mini-instruct.Q4_K_S.gguf",
    local_dir="/content/models/phi",
    local_dir_use_symlinks=False
)

def load_model(model_path):
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=0,
            use_mmap=False,
            verbose=True
        )
        return llm
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def extract_json(text):
    text =re.sub(r'^json\s*', '', text.strip())
    try:
        return json.loads(text)
    except Exception as e:
        return {"Error": f"Failed to extract JSON: {str(e)}"}


def build_prompt(text):
    instruction = """
You are an information extraction engine. Return ONLY valid JSON, no explanations.

JSON Structure:
{
  "Title": "Paper title",
  "Authors": ["Author 1", "Author 2"],
  "DOI": "DOI if available",
  "Keywords": ["Keyword1", "Keyword2"],
  "Abstract": "Abstract text",
  "Document Type": "Research Paper, Thesis, etc.",
  "Number of References": 10
}

Extract metadata from the following scientific paper:
"""
    return f"<|user|>\n{instruction.strip()}\n{text[:2000]}\n<|assistant|>"


def extract_metadata(generator, paper_text):
    prompt = build_prompt(paper_text)
    response = generator.create_completion(
        prompt,
        max_tokens=1024,
        temperature=0,
        top_p=1.0,
        stop=["<|end|>", "</s>"], 
    )
    raw_output = response["choices"][0]["text"]
    return extract_json(raw_output)

# Extract raw text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join(page.get_text("text") for page in doc)
    return text if text.strip() else "Error: No extractable text found in PDF."

def process_pdf(pdf_file):
    extracted_text = extract_text_from_pdf(pdf_file.name)
    if extracted_text.startswith("Error:"):
        return {"Error": "No extractable text found in the PDF."}
    metadata = extract_metadata(model, extracted_text)
    return metadata

def main():
    model_path = "/content/models/phi/Phi-4-mini-instruct.Q4_K_S.gguf"
    global model
    model = load_model(model_path)
    #Gradio interface
    iface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF"),
    outputs="json",
    title="Metadata Extractor",
    description="Upload only a PDF to extract metadata"
    )
    # Launch the interface
    iface.launch()

if __name__ == "__main__":
    main()