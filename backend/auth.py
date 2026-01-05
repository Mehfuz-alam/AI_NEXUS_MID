import hashlib
from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

serializer = URLSafeTimedSerializer("CHANGE_THIS_SECRET_KEY")


def _normalize(value: str) -> str:
    """
    Ensures bcrypt NEVER sees >72 bytes
    """
    if not isinstance(value, str):
        value = str(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    return pwd_context.hash(_normalize(password))


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(_normalize(password), hashed_password)


def create_reset_token(email: str) -> str:
    return serializer.dumps(email)


def verify_reset_token(token: str, max_age: int = 3600) -> str:
    return serializer.loads(token, max_age=max_age)
