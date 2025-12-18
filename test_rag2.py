from rag import *
from llm import StatelessLLM

MODEL_NAME_FINAL = "granite4:1b"

def test_case():
    rag = RAGInstance(verbose=True)
    task = "Who is Kitayama Tou?"
    formatted_chunks = rag.retrieve(task, iterations=5)

    prompt = f"CONTEXT:\n{formatted_chunks}\nPROMPT:\n{task}"

    print(prompt)
    
    llm = StatelessLLM(
        model=MODEL_NAME_FINAL,
        max_tokens=3200,
    )

    print(llm.answer(prompt))



if __name__ == "__main__":
    test_case()