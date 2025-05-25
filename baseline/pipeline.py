import json
from datetime import datetime
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator
import os

# Resolve document_path relative to this script (avoid hardcoded relative path issues)
current_dir = os.path.dirname(os.path.abspath(__file__))
document_path = os.path.join(current_dir, "data", "winnie_the_pooh.txt")
test_input_path = os.path.join(current_dir, "data", "test_inputs.json")
log_path = os.path.join(current_dir, "..", "logs", "log.jsonl")  # Adjust path so logs folder is inside your repo root

def log_interaction(log_path, question, retrieved_chunks, prompt, generated_answer, group_id=0):
    log_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": generated_answer
    }
    os.makedirs(os.path.dirname(log_path), exist_ok=True)  # ensure logs folder exists
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")

def run_pipeline_with_logging(document_path, question, group_id=0, log_path=log_path):
    if not os.path.isfile(document_path):
        raise FileNotFoundError(f"Document file not found: {document_path}")
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    retriever = Retriever()
    retriever.add_documents([text])
    retrieved_chunks = retriever.query(question)
    context = "\n\n".join(retrieved_chunks)

    generator = Generator()
    prompt = generator.build_prompt(context, question)
    answer = generator.generate_answer(context, question)

    # Log the interaction (question, context, prompt, answer, timestamp, group_id)
    log_interaction(
        log_path=log_path,
        question=question,
        retrieved_chunks=retrieved_chunks,
        prompt=prompt,
        generated_answer=answer,
        group_id=group_id
    )

    return answer

def run_batch_tests(document_path, test_input_path=test_input_path, log_path=log_path):
    if not os.path.isfile(test_input_path):
        raise FileNotFoundError(f"Test input file not found: {test_input_path}")

    with open(test_input_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Running {len(test_data)} test questions...\n")

    for i, item in enumerate(test_data):
        question = item["question"]
        expected_answer = item.get("expected_answer", "N/A")
        print(f"[Test {i + 1}] Question: {question}")

        answer = run_pipeline_with_logging(document_path, question, group_id=i, log_path=log_path)

        print(f"Expected: {expected_answer}")
        print(f"Generated: {answer}\n{'-'*50}\n")

if __name__ == "__main__":
    # Run single test example
    question = "Who is always sad?"
    print("Single test result:\n", run_pipeline_with_logging(document_path, question))

    # Run batch tests with logging
    run_batch_tests(document_path)
