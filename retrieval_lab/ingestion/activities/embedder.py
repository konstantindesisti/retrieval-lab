"""
Temporal activity for generating embeddings with Redis caching.

Flow for each chunk:
  1. cache_key = sha256(model + ":" + content)
  2. Redis GET → cache hit: deserialize and return immediately
  3. Redis MISS → collect into a batch and call the OpenAI API
  4. Store results in Redis (pipeline, single round-trip)
  5. Return EmbeddedDocument

Why use Redis pipeline for cache writes:
  N chunks → N SET commands → one TCP round-trip instead of N.

Note about Temporal payload size:
  1536 floats * ~17 chars (JSON) * 50 chunks ≈ 1.3MB per workflow step.
  Temporal's default limit is 4MB, so this is acceptable for typical articles.

  For larger documents (books, documentation), consider having the embedder
  write embeddings directly to the database and return only chunk IDs.
"""
import hashlib
import json
import structlog
from temporalio import activity


