from rag import *
from llm import StatelessLLM

MODEL_NAME_FINAL = MODEL

def test_case():
    rag = RAGInstance(verbose=True)

    rag.add_file("./example_data/corpus.txt")

    task = "Who is Kitayama Tou?"
    formatted_chunks = rag.retrieve(task, iterations=3)

    prompt = f"CONTEXT:\n{formatted_chunks}\nPROMPT:\n{task}"

    print(prompt)
    
    llm = StatelessLLM(
        model=MODEL_NAME_FINAL,
        max_tokens=3200,
    )

    print(llm.answer(prompt))



if __name__ == "__main__":
    test_case()