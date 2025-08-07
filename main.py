from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import run_pipeline, initialize_vectorstore
import uvicorn

app = FastAPI(title="Chatbot API", description="API for chatbot interactions")

class QueryInput(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    initialize_vectorstore()

@app.get("/")
async def read_root():
    return {"message": "Welcome to LangGraph + FastAPI chatbot"}

@app.post("/query")
async def get_answer(input: QueryInput):
    try:
        response = run_pipeline(input.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)