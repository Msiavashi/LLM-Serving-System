from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Literal, Optional
import uvicorn
import uuid
from src.queues.fcfs_queue import FCFSQueue
from src.config import ConfigManager
from src.queues.storage.redis_storage import RedisQueueStorage
from src.sequence import Stage

app = FastAPI()

# Initialize Redis queues for different priorities
ls_queue = FCFSQueue(RedisQueueStorage("ls_prefill_queue", None, Stage.PREFILL))
non_ls_queue = FCFSQueue(RedisQueueStorage("non_ls_prefill_queue", None, Stage.PREFILL))

class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.7
    priority: int = 0  # 0 for normal, 1 for high priority

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    priority: Optional[int] = 0

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = 50
    temperature: Optional[float] = 0.7
    priority: Optional[int] = 0  # 0 for normal, 1 for high priority

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    try:
        request_id = str(uuid.uuid4())
        messages = request.messages
        prompt = "\n".join([msg.content for msg in messages])
        # Extract priority from the first message if available
        priority = getattr(messages[0], 'priority', 0) if messages else 0
        
        # Create request payload
        payload = {
            "id": request_id,
            "prompt": prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "priority": priority
        }
        # Route to appropriate queue based on priority
        if priority == 1:
            ls_queue.enqueue(payload)
        else:
            non_ls_queue.enqueue(payload)
        
        return {
            "id": request_id,
            "object": "chat.completion",
            "status": "processing",
            "priority": priority,
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

@app.get("/v1/queue/status")
async def get_queue_status():
    return {
        "high_priority_queue_size": ls_queue.size(),
        "normal_priority_queue_size": non_ls_queue.size()
    }

if __name__ == "__main__":
    config = ConfigManager()
    uvicorn.run(
        app, 
        host=config.get('server.host', '0.0.0.0'), 
        port=config.get('server.port', 8000)
    )
