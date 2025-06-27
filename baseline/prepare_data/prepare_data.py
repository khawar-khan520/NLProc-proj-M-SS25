import pandas as pd

def load_and_prepare_medquad(csv_path):
    df = pd.read_csv(csv_path)
    df.dropna(subset=['question', 'answer'], inplace=True)

    # clean text
    df['question'] = df['question'].astype(str).str.strip()
    df['answer'] = df['answer'].astype(str).str.strip()

    return df  # it will Return full DataFrame (not chunks) for evaluation

