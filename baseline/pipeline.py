
from baseline.retriever.retriever import Retriever
from baseline.generator.generator import Generator

def run_pipeline(document_path, question):
    with open(document_path, "r", encoding="utf-8") as f:
        text = f.read()

    retriever = Retriever()
    retriever.add_documents([text])
    results = retriever.query(question)

    context = "\n\n".join(results)

    generator = Generator()
    answer = generator.generate_answer(context, question)


    return answer

if __name__ == "__main__":
    document_path = "baseline/data/winnie_the_pooh.txt"
    question = "Who is always sad?"
    print(run_pipeline(document_path, question))
