import os
import json
import re
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def calculate_f1(prediction, ground_truth):
    pred_tokens = tokenize(prediction)
    gt_tokens = tokenize(ground_truth)
    common = set(pred_tokens) & set(gt_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    from pipeline import MedicalAssistant
    assistant = MedicalAssistant()

    test_set_path = "baseline/data/test_inputs.json"
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    total_f1 = 0.0
    count = 0

    for item in test_set:
        question = item["question"]
        ground_truth = item["answer"]

        print(f"Question: {question}")

        context_chunks, generated_answer = assistant.answer(question)

        if context_chunks:
            print("\n📚 Top 3 Relevant Chunks:")
            for i, chunk in enumerate(context_chunks, 1):
                print(f"\n🔹 Chunk {i}:\n{chunk.strip()}")
        else:
            print("\n⚠️ No relevant chunks found.")

        print("\n🤖 Generated Answer:")
        print(generated_answer)

        f1 = calculate_f1(generated_answer, ground_truth)
        print(f"\n📊 F1 score: {f1:.3f}")
        print("-" * 80)

        total_f1 += f1
        count += 1

    avg_f1 = total_f1 / count if count > 0 else 0.0
    print(f"\n=== Average F1 score over {count} questions: {avg_f1:.3f} ===")

if __name__ == "__main__":
    main()
