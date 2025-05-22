import json
import os
from datetime import datetime

def log_output(question, retrieved_chunks, prompt, generated_answer, log_path="baseline/data/logs.jsonl", group_id="Team_Winnie"):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "group_id": group_id,
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "prompt": prompt,
        "generated_answer": generated_answer
    }

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
