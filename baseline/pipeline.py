import json
from datetime import datetime
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator


def log_interaction(log_path, question, retrieved_chunks, prompt, generated_answer, group_id=0):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": generated_answer
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")


def run_pipeline_with_logging(document_path, question, group_id=0, log_path="logs/log.jsonl"):
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    retriever = Retriever()
    retriever.add_documents([text])
    retrieved_chunks = retriever.query(question)
    context = "\n\n".join(retrieved_chunks)

    generator = Generator()
    prompt = generator.build_prompt(context, question)
    answer = generator.generate_answer(context, question)

    # here to Log the interaction
    log_interaction(
        log_path=log_path,
        question=question,
        retrieved_chunks=retrieved_chunks,
        prompt=prompt,
        generated_answer=answer,
        group_id=group_id
    )

    return answer


def run_batch_tests(document_path, test_input_path="baseline/data/test_inputs.json", log_path="logs/log.jsonl"):
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
    # Run single test
    document_path = "baseline/data/winnie_the_pooh.txt"
    question = "Who is always sad?"
    print(run_pipeline_with_logging(document_path, question))

    # this code is to run  the whole batch tests
    run_batch_tests(document_path)
