from fastapi import FastAPI


app = FastAPI()


@app.post("/process_documents")
def process_documents(conversation_id: str, documents):
    pass


@app.post("/get_chunks")
def get_chunks(conversation_id: str, query: str, top_k: int) -> list[tuple[str, float]]:
    pass
