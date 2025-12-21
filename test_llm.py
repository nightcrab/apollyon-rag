from llm import StatelessLLM
from config import MODEL

def test_case():
    
    llm = StatelessLLM(
        model=MODEL,
        temperature=0.1,
        max_tokens=32,
    )

    print(llm.answer("hello"))


if __name__ == "__main__":
    test_case()