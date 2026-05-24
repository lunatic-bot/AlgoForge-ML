import hashlib
import json
import redis
from typing import Optional

# Using decode_responses=True so values come back as strings instead of bytes
redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True) 

def generate_cache_key(model_id: str, input_data: dict) -> str:
    """Creates a unique, deterministic hash key based on model ID and input features."""
    serialized_data = json.dumps(input_data, sort_keys=True)
    raw_key = f"{model_id}:{serialized_data}"
    return hashlib.md5(raw_key.encode('utf-8')).hexdigest()


def get_cached_prediction(cache_key: str) -> Optional[dict]:
    """Retrieves a prediction from cache if it exists."""
    try:
        cached_val = redis_client.get(cache_key)
        if cached_val:
            return json.loads(cached_val)
    except redis.RedisError as e:
        print(f"Redis Error: {e}")

    return None


def set_cached_prediction(cached_key: str, data: dict, expire_seconds: int = 3600):
    """Stores a prediction result in Redis with an expiration limit (1 hour default)."""
    try:
        redis_client.set(cached_key, json.dumps(data), ex=expire_seconds)
    except redis.RedisError as e:
        print(f"Redis Error: {e}")
    
