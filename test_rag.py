from rag import *
from hdb import SearchContext

def test_case():
    
    llm = StatelessLLM(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=32,
    )

    db = load_or_create_db("./example_data/corpus.txt")

    ctx = SearchContext(db, top_k=3)

    task = "Who is Kitayama Tou?"
    
    chunks = []
    queries = []

    chunks, _ = ctx.search(task)
    queries.append(task)

    prompt = generate_prompt(
        task,
        chunks=chunks,
        queries=queries,
        titles=db.titles
    )

    for x in range(5):
        query = llm.answer(prompt)
        print(query)   
        queries.append(query)
        
        chunks = combine_chunks(chunks, ctx.search(query)[0])
        
        prompt = generate_prompt(
            task,
            chunks=chunks,
            queries=queries,
            titles=db.titles
        )

    llm.stop()
    
    llm = StatelessLLM(
        model=MODEL_NAME_FINAL,
        temperature=0.1,
        max_tokens=3200,
    )


    print(f"{len(chunks)} chunks retrieved")
    
    prompt = f"CONTEXT:\n{format_chunks(chunks)}\nPROMPT:\n{task}"

    print(prompt)
    print(llm.answer(prompt))

    llm.stop()


if __name__ == "__main__":
    test_case()