
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_prompt(self, context, question):
        return f"Answer the question based on the context.

Context:
{context}

Question:
{question}"

    def generate_answer(self, context, question, max_tokens=100):
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

