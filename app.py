from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot import chat 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    message = chat_req.message.strip()
    session_id = chat_req.session_id.strip()
    if not message or not session_id:
        return {"response": "Both message and session_id are required."}
    try:
        response = chat(message, session_id)
        return {"response": response}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}
