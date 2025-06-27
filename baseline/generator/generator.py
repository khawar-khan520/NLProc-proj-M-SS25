from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class Generator:
    def __init__(self, model_name='google/flan-t5-base'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, query, context, max_length=256):
        input_text = (
            "You are a medical assistant. Based only on the following medical context, "
            "provide a short and informative answer to the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer (in 2-3 sentences):"
        )

        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
