import json
import redis
from typing import Optional, Any, Tuple
from src.config import ConfigManager
from src.sequence import Sequence, Stage
from src.queues.storage.base_queue_storage import BaseQueueStorage

class RedisQueueStorage(BaseQueueStorage):
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
        return cls._redis_client

    def __init__(self, queue_name: str, tokenizer, stage: Stage):
        self.queue_name = queue_name
        self.redis = self.get_redis_client()
        self.tokenizer = tokenizer
        self.stage = stage

    def enqueue(self, packed_item: Tuple[Any, float]) -> None:
        try:
            # packed_item is expected to be a tuple (payload, arrival_time)
            self.redis.lpush(self.queue_name, json.dumps(packed_item))
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Enqueue error: {str(e)}")

    def dequeue(self) -> Optional[Tuple[Any, float]]:
        try:
            packed_str = self.redis.rpop(self.queue_name)
            return self._deserialize(packed_str) if packed_str else None
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Dequeue error: {str(e)}")

    def peek(self) -> Optional[Tuple[Any, float]]:
        try:
            packed_str = self.redis.lindex(self.queue_name, -1)
            return self._deserialize(packed_str) if packed_str else None
        except (redis.RedisError, ValueError) as e:
            raise RuntimeError(f"Peek error: {str(e)}")

    def _deserialize(self, packed_str):
        packed_item = json.loads(packed_str)
        # packed_item is a list/tuple where:
        #   index 0: payload (a dict)
        #   index 1: arrival_time (a float timestamp)
        deserialized_request = packed_item[0]
        arrival_time = packed_item[1]
        # Pass the arrival_time to the Sequence constructor
        sequence = Sequence(
            deserialized_request["prompt"],
            self.tokenizer,
            self.stage,
            priority=deserialized_request["priority"],
            arrival_time=arrival_time
        )
        return (sequence, arrival_time)

    def is_empty(self) -> bool:
        return self.redis.llen(self.queue_name) == 0

    def size(self) -> int:
        return self.redis.llen(self.queue_name)
