# import hashlib
# import json
# import redis
# from typing import Optional
# # Import your centralized logger utility
# from .logger import setup_logger

# # Initialize a dedicated logger for cache operational layers
# logger = setup_logger("api.cache")

# logger.info("Initializing Redis client connection connection pools pointing to host: 'redis' on port 6379.")

# try:
#     # Using decode_responses=True so values come back as strings instead of bytes
#     redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True) 
#     logger.info("Redis socket connection client initialized successfully.")
# except Exception as init_err:
#     logger.error(f"Failed to compile initial Redis socket config: {str(init_err)}", exc_info=True)


# def generate_cache_key(model_id: str, input_data: dict) -> str:
#     """Creates a unique, deterministic hash key based on model ID and input features."""
#     logger.debug(f"Generating deterministic MD5 cache hash signature for Model ID: {model_id}")
#     try:
#         # Sorting keys ensures that {'a': 1, 'b': 2} and {'b': 2, 'a': 1} produce identical hashes
#         serialized_data = json.dumps(input_data, sort_keys=True)
#         raw_key = f"{model_id}:{serialized_data}"
#         cache_key = hashlib.md5(raw_key.encode('utf-8')).hexdigest()
#         logger.debug(f"Cache key successfully generated: {cache_key}")
#         return cache_key
#     except Exception as hash_err:
#         logger.error(f"Failed to generate deterministic cache hash signature: {str(hash_err)}", exc_info=True)
#         # Fallback to avoid breaking prediction thread, returns a random UUID string representation
#         import uuid
#         return f"fallback:{uuid.uuid4().hex}"


# def get_cached_prediction(cache_key: str) -> Optional[dict]:
#     """Retrieves a prediction from cache if it exists."""
#     logger.debug(f"Querying Redis in-memory lookup cache layer for key: {cache_key}")
#     try:
#         cached_val = redis_client.get(cache_key)
#         if cached_val:
#             logger.debug(f"Cache lookup successful. Parsing string payloads into structured JSON maps.")
#             return json.loads(cached_val)
        
#         logger.debug(f"Cache lookups executed cleanly but returned empty state for key: {cache_key}")
#     except redis.RedisError as e:
#         # Caught specifically so if the Redis container drops offline, the system degrades gracefully
#         logger.error(f"Fail-Safe Intercept: Redis read cluster threw an error lookup operation: {str(e)}", exc_info=True)
#     except json.JSONDecodeError as jde:
#         logger.error(f"Failed parsing cached string elements into structured dictionaries for key {cache_key}: {str(jde)}")
#     except Exception as unexpected_err:
#         logger.error(f"Unexpected exception reading memory state values from cache: {str(unexpected_err)}", exc_info=True)

#     return None


# def set_cached_prediction(cached_key: str, data: dict, expire_seconds: int = 3600):
#     """Stores a prediction result in Redis with an expiration limit (1 hour default)."""
#     logger.debug(f"Attempting cache write persistence block for key: {cached_key} with TTL expiration: {expire_seconds}s")
#     try:
#         serialized_payload = json.dumps(data)
#         redis_client.set(cached_key, serialized_payload, ex=expire_seconds)
#         logger.info(f"Successfully recorded prediction payload cache elements under memory key signature: {cached_key}")
#     except redis.RedisError as e:
#         logger.error(f"Fail-Safe Intercept: Redis write cluster threw an error save operation: {str(e)}", exc_info=True)
#     except TypeError as type_err:
#         logger.error(f"Serialization failed. Provided model metrics are not JSON serializable: {str(type_err)}", exc_info=True)
#     except Exception as unexpected_write_err:
#         logger.error(f"Unexpected operational crash committing data packet keys to Redis: {str(unexpected_write_err)}", exc_info=True)



# # import hashlib
# # import json
# # import redis
# # from typing import Optional

# # # Using decode_responses=True so values come back as strings instead of bytes
# # redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True) 

# # def generate_cache_key(model_id: str, input_data: dict) -> str:
# #     """Creates a unique, deterministic hash key based on model ID and input features."""
# #     serialized_data = json.dumps(input_data, sort_keys=True)
# #     raw_key = f"{model_id}:{serialized_data}"
# #     return hashlib.md5(raw_key.encode('utf-8')).hexdigest()


# # def get_cached_prediction(cache_key: str) -> Optional[dict]:
# #     """Retrieves a prediction from cache if it exists."""
# #     try:
# #         cached_val = redis_client.get(cache_key)
# #         if cached_val:
# #             return json.loads(cached_val)
# #     except redis.RedisError as e:
# #         print(f"Redis Error: {e}")

# #     return None


# # def set_cached_prediction(cached_key: str, data: dict, expire_seconds: int = 3600):
# #     """Stores a prediction result in Redis with an expiration limit (1 hour default)."""
# #     try:
# #         redis_client.set(cached_key, json.dumps(data), ex=expire_seconds)
# #     except redis.RedisError as e:
# #         print(f"Redis Error: {e}")
    
