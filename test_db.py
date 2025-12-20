
from hdb import HybridDB, SearchContext
import os

def load_or_create_db(path):
    db_path = path + ".db"
    if os.path.exists(db_path):
        print("using existing db")
        return HybridDB.load(db_path)
    print("creating new db")
    return HybridDB(path)


def format_chunks(chunks):
    ret = ""
    for i in range(len(chunks)):
        ret += "----------- CHUNK START -----------\n"
        ret += chunks[i] + "\n"
        ret += "----------- CHUNK END -----------\n"
        ret += "\n"
    return ret

db = load_or_create_db("./example_data/corpus.txt")

ctx = SearchContext(db, top_k=10)

task = "Who is Lilya Polonskaya?"

print(format_chunks(ctx.search(task)[0]))