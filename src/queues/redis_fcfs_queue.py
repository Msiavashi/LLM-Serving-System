import redis
import json
from typing import Optional, Any
from src.sequence.stage import Stage
from src.queues.fcfs_queue import FCFSQueue
from src.config import ConfigManager
from src.sequence.sequence import Sequence

class RedisFCFSQueue(FCFSQueue):
    """Redis-backed First-Come, First-Served (FCFS) queue implementation."""
    
    _redis_client: Optional[redis.Redis] = None
    
    @classmethod
    def get_redis_client(cls):
        if cls._redis_client is None:
            config = ConfigManager()
            try:
                cls._redis_client = redis.Redis(
                    host=config.get('redis.host', 'localhost'),
                    port=config.get('redis.port', 6379),
                    db=config.get('redis.db', 0),
                    password=config.get('redis.password'),
                    ssl=config.get('redis.ssl', False),
                    socket_timeout=config.get('redis.socket_timeout', 5),
                    socket_connect_timeout=config.get('redis.socket_connect_timeout', 5),
                    retry_on_timeout=config.get('redis.retry_on_timeout', True),
                    max_connections=config.get('redis.max_connections', 10),
                    decode_responses=True
                )
                cls._redis_client.ping()
            except redis.ConnectionError as e:
                raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
            except Exception as e:
                raise Exception(f"Error initializing Redis client: {str(e)}")
        return cls._redis_client

    def __init__(self, queue_name: str, tokenizer, stage: Stage):
        super().__init__()
        self.queue_name = queue_name
        self.redis = self.get_redis_client()
        self.tokenizer = tokenizer
        self.stage: Stage = stage

    def _storage_enqueue(self, item: Any):
        """Store item in Redis as JSON"""
        try:
            self.redis.lpush(self.queue_name, json.dumps(item))
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Enqueue error: {str(e)}")

    def _storage_dequeue(self) -> Optional[tuple]:
        """Retrieve and unpack item from Redis"""
        try:
            packed_str = self.redis.rpop(self.queue_name)
            if not packed_str:
                return None
            packed_item = json.loads(packed_str)
            deserialized_request = packed_item[0]  # Get the item part of the packed tuple
            sequence = Sequence(deserialized_request["prompt"], self.tokenizer, self.stage)
            return (sequence, packed_item[1])  # Return as a packed tuple (item, timestamp)
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Dequeue error: {str(e)}")

    def _storage_peek(self) -> Optional[tuple]:
        """Peek at the next item in Redis"""
        try:
            packed_str = self.redis.lindex(self.queue_name, -1)
            if not packed_str:
                return None
            packed_item = json.loads(packed_str)
            deserialized_request = packed_item[0]  # Get the item part of the packed tuple
            sequence = Sequence(deserialized_request["prompt"], self.tokenizer, self.stage)
            return (sequence, packed_item[1])  # Return as a packed tuple (item, timestamp)
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Peek error: {str(e)}")

    def is_empty(self) -> bool:
        return self.redis.llen(self.queue_name) == 0

    def size(self) -> int:
        return self.redis.llen(self.queue_name)
