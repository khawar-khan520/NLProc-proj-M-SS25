
import json
import os
from datetime import datetime

class Logger:
    def __init__(self, logfile="logs.jsonl"):
        self.logfile = logfile
        os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None

    def log(self, question, retrieved_chunks, prompt, generated_answer, group_id):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "group_id": group_id,
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "prompt": prompt,
            "generated_answer": generated_answer
        }
        with open(self.logfile, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
