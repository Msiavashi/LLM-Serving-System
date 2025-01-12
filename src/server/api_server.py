from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
import uvicorn
import uuid
from src.config import ConfigManager
from src.queues.redis_fcfs_queue import RedisFCFSQueue

app = FastAPI()

# Initialize Redis queue
queue = RedisFCFSQueue("prefill_queue", None, None)

class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        request_id = str(uuid.uuid4())
        prompt = "\n".join([msg.content for msg in request.messages])
        
        # Send to Redis queue
        queue.enqueue({
            "id": request_id,
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        })
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "status": "processing",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Request accepted and queued"
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    config = ConfigManager()
    uvicorn.run(
        app, 
        host=config.get('server.host', '0.0.0.0'), 
        port=config.get('server.port', 8000)
    )
