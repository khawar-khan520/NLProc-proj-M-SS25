import pandas as pd
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
from baseline.prepare_data.prepare_data import load_and_prepare_medquad

import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# Token-based F1 Score
def compute_f1(true_answer, pred_answer):
    true_tokens = [t.lower() for t in word_tokenize(true_answer) if t.isalnum() and t.lower() not in stop_words]
    pred_tokens = [t.lower() for t in word_tokenize(pred_answer) if t.isalnum() and t.lower() not in stop_words]

    common = set(true_tokens) & set(pred_tokens)

    if len(common) == 0:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def main():
    # Load MedQuAD
    medquad_path = 'baseline/data/medquad.csv'
    df = load_and_prepare_medquad(medquad_path)
    print("[DEBUG] Type of df:", type(df))
    questions = df['question'].tolist()
    answers = df['answer'].tolist()

    # Initialize models
    retriever = Retriever()
    retriever.load_index()
    generator = Generator()

    f1_scores = []

    print(f"[Eval] Running evaluation on {len(questions)} questions...")

    for q, true_ans in zip(questions, answers):
        context_chunks = retriever.retrieve(q, top_k=3)
        if not context_chunks:
            pred_ans = ""
        else:
            context = " ".join(context_chunks)
            pred_ans = generator.generate(q, context)

        f1 = compute_f1(true_ans, pred_ans)
        f1_scores.append(f1)

    avg_f1 = sum(f1_scores) / len(f1_scores)
    print(f"\n✅ Average F1 Score: {avg_f1:.4f}")


if __name__ == "__main__":
    main()
