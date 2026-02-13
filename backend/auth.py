import hashlib
from passlib.context import CryptContext
from itsdangerous import (
    URLSafeTimedSerializer,
    BadSignature,
    SignatureExpired,
)

SECRET_KEY = "CHANGE_THIS_SECRET_KEY"
RESET_SALT = "password-reset"

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

serializer = URLSafeTimedSerializer(SECRET_KEY)


def _normalize(value: str) -> str:
    """
    Ensures bcrypt never receives >72 bytes
    """
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    return pwd_context.hash(_normalize(password))


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_normalize(password), hashed_password)


def create_reset_token(email: str) -> str:
    return serializer.dumps(email, salt=RESET_SALT)


def verify_reset_token(token: str, max_age: int = 3600) -> str | None:
    try:
        return serializer.loads(
            token,
            salt=RESET_SALT,
            max_age=max_age
        )
    except (BadSignature, SignatureExpired):
        return None
