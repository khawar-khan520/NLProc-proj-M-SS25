
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def build_prompt(self, context, question):
        prompt = (
            "Answer the question based on the context.\n\n"
            "Context:\n" + context + "\n\n"
            "Question:\n" + question + "\n"
        )
        return prompt

    def generate_answer(self, context, question):
        prompt = self.build_prompt(context, question)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
