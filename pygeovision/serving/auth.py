"""Authentication — API key and JWT for the inference server."""
from __future__ import annotations
import hashlib, hmac, logging, os, time
from typing import Any, Dict, Optional
logger = logging.getLogger(__name__)


class APIKeyAuth:
    """API key authentication middleware.

    Example::

        auth = APIKeyAuth(keys={"user1": "key_abc123", "user2": "key_xyz789"})
        if auth.verify("key_abc123"):
            # authorised
    """

    def __init__(self, keys: Optional[Dict[str, str]] = None,
                 env_var: str = "PGV_API_KEYS") -> None:
        self._keys: Dict[str, str] = {}
        if keys:
            for user, key in keys.items():
                self.add_key(user, key)
        # Load from environment
        env_keys = os.environ.get(env_var, "")
        for pair in env_keys.split(","):
            if ":" in pair:
                user, key = pair.split(":", 1)
                self.add_key(user.strip(), key.strip())

    def add_key(self, user: str, key: str) -> None:
        self._keys[self._hash(key)] = user

    def _hash(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def verify(self, api_key: str) -> Optional[str]:
        """Verify API key. Returns username on success, None on failure."""
        return self._keys.get(self._hash(api_key))

    def generate_key(self, user: str, length: int = 32) -> str:
        """Generate and register a new API key."""
        import secrets
        key = "pgv_" + secrets.token_hex(length)
        self.add_key(user, key)
        return key


class JWTAuth:
    """JWT authentication for stateless API access.

    Example::

        auth = JWTAuth(secret="my-secret-key")
        token = auth.create_token({"user": "alice", "role": "admin"})
        payload = auth.verify_token(token)
    """

    def __init__(self, secret: Optional[str] = None, algorithm: str = "HS256",
                 expiry_hours: int = 24) -> None:
        self.secret = secret or os.environ.get("PGV_JWT_SECRET", "pgv-default-secret-change-me")
        self.algorithm = algorithm
        self.expiry_hours = expiry_hours

    def create_token(self, payload: Dict[str, Any]) -> str:
        try:
            import jwt
            payload = {**payload, "exp": time.time() + self.expiry_hours * 3600,
                       "iat": time.time()}
            return jwt.encode(payload, self.secret, algorithm=self.algorithm)
        except ImportError:
            raise ImportError("pip install PyJWT")

    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            import jwt
            return jwt.decode(token, self.secret, algorithms=[self.algorithm])
        except ImportError:
            raise ImportError("pip install PyJWT")
        except Exception as exc:
            logger.warning("JWT verification failed: %s", exc)
            return None
