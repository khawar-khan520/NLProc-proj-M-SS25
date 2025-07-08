import os
import json
import nltk
from nltk.tokenize import word_tokenize
import re

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

    # Load your test set with questions and answers for evaluation
    test_set_path = "baseline/data/test_inputs.json"
    with open(test_set_path, 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    print("Interactive QA Evaluation - type 'exit' to quit\n")

    while True:
        question = input("Enter question: ")
        if question.lower() == 'exit':
            break

        # Retrieve top 3 chunks AND generated answer
        context_chunks, answer = assistant.answer(question)

        if not context_chunks:
            print("\n🤖 Generated Answer:")
            print(answer)
            print("\n⚠️ No relevant medical information found for this question.")
            print("-" * 60)
            continue

        print("\n📚 Top 3 Relevant Chunks:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"\n🔹 Chunk {i}:\n{chunk.strip()}")

        print("\n🤖 Generated Answer:")
        print(answer)

        # Find closest matching ground truth question (simple textual similarity)
        def find_closest_question(user_q, test_questions):
            user_tokens = set(tokenize(user_q))
            max_overlap = 0
            best_match = None
            for item in test_questions:
                test_tokens = set(tokenize(item["question"]))
                overlap = len(user_tokens & test_tokens)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match = item
            return best_match

        matched_item = find_closest_question(question, test_set)

        if matched_item:
            gt_answer = matched_item["answer"]
            f1 = calculate_f1(answer, gt_answer)
            print(f"\n📊 F1 score (vs. ground truth): {f1:.3f}")
            print(f"📝 Matched ground truth question: '{matched_item['question']}'")
        else:
            print("\n⚠️ No matching ground truth question found in test set.")

        print("-" * 60)


if __name__ == "__main__":
    main()
