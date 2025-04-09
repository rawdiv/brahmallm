import os
import jwt
import uuid
import json
import time
import secrets
from datetime import datetime, timedelta
from passlib.context import CryptContext
from typing import Dict, List, Optional, Union
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from pydantic import BaseModel

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Token settings
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# API key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Path for auth data storage
AUTH_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "auth")
os.makedirs(AUTH_DIR, exist_ok=True)
USERS_FILE = os.path.join(AUTH_DIR, "users.json")
API_KEYS_FILE = os.path.join(AUTH_DIR, "api_keys.json")

# Models
class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: Optional[bool] = False
    is_admin: bool = False
    hashed_password: str = ""

class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserUpdate(BaseModel):
    email: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    disabled: Optional[bool] = None
    is_admin: Optional[bool] = None

class Token(BaseModel):
    access_token: str
    token_type: str
    user: Dict

class TokenData(BaseModel):
    username: Optional[str] = None

class ApiKey(BaseModel):
    id: str
    key: str
    name: str
    user_id: str
    created_at: str
    last_used: Optional[str] = None
    usage_count: int = 0
    enabled: bool = True
    permissions: List[str] = ["generate", "chat"]
    rate_limit: int = 100  # Requests per day

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def load_users():
    """Load users from file or create default admin if none exist."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
        return users
    else:
        # Create default admin user
        admin_password = secrets.token_urlsafe(12)
        users = {
            "admin": {
                "username": "admin",
                "email": "admin@brahma.ai",
                "full_name": "Brahma Admin",
                "disabled": False,
                "is_admin": True,
                "hashed_password": get_password_hash(admin_password)
            }
        }
        save_users(users)
        print(f"\n=== Default Admin User Created ===")
        print(f"Username: admin")
        print(f"Password: {admin_password}")
        print(f"Please change this password after first login!\n")
        return users

def save_users(users):
    """Save users to file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def load_api_keys():
    """Load API keys from file or create empty dict if none exist."""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            api_keys = json.load(f)
        return api_keys
    else:
        api_keys = {}
        save_api_keys(api_keys)
        return api_keys

def save_api_keys(api_keys):
    """Save API keys to file."""
    with open(API_KEYS_FILE, "w") as f:
        json.dump(api_keys, f, indent=2)

def get_user(username: str):
    """Get user by username."""
    users = load_users()
    if username in users:
        user_data = users[username]
        return User(**user_data)
    return None

def authenticate_user(username: str, password: str):
    """Authenticate user."""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

async def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    """Check if user is active."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_admin_user(current_user: User = Depends(get_current_active_user)):
    """Check if user is admin."""
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Not authorized")
    return current_user

def create_api_key(user_id: str, name: str):
    """Create new API key for user."""
    api_keys = load_api_keys()
    
    # Generate unique key
    key = f"brahma_{secrets.token_hex(16)}"
    key_id = str(uuid.uuid4())
    
    new_key = {
        "id": key_id,
        "key": key,
        "name": name,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_used": None,
        "usage_count": 0,
        "enabled": True,
        "permissions": ["generate", "chat"],
        "rate_limit": 100
    }
    
    api_keys[key] = new_key
    save_api_keys(api_keys)
    
    return ApiKey(**new_key)

def delete_api_key(key: str):
    """Delete API key."""
    api_keys = load_api_keys()
    if key in api_keys:
        del api_keys[key]
        save_api_keys(api_keys)
        return True
    return False

def get_api_keys_for_user(user_id: str):
    """Get all API keys for a user."""
    api_keys = load_api_keys()
    user_keys = []
    
    for key, data in api_keys.items():
        if data["user_id"] == user_id:
            # Don't include the actual key in listings
            data_copy = data.copy()
            data_copy["key"] = key[-8:]  # Only show last 8 chars
            user_keys.append(ApiKey(**data_copy))
    
    return user_keys

async def validate_api_key(api_key: str = Depends(API_KEY_HEADER)):
    """Validate API key from header."""
    api_keys = load_api_keys()
    
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    key_data = api_keys[api_key]
    
    # Check if key is enabled
    if not key_data["enabled"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is disabled",
        )
    
    # Update usage statistics
    key_data["last_used"] = datetime.utcnow().isoformat()
    key_data["usage_count"] += 1
    api_keys[api_key] = key_data
    save_api_keys(api_keys)
    
    return key_data

def create_user(user_data: UserCreate):
    """Create a new user."""
    users = load_users()
    
    if user_data.username in users:
        return None  # User already exists
    
    # Create new user
    new_user = {
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "disabled": False,
        "is_admin": False,
        "hashed_password": get_password_hash(user_data.password)
    }
    
    users[user_data.username] = new_user
    save_users(users)
    
    return User(**new_user)

def update_user(username: str, user_data: UserUpdate):
    """Update user data."""
    users = load_users()
    
    if username not in users:
        return None  # User doesn't exist
    
    # Update fields
    if user_data.email is not None:
        users[username]["email"] = user_data.email
    
    if user_data.full_name is not None:
        users[username]["full_name"] = user_data.full_name
    
    if user_data.password is not None:
        users[username]["hashed_password"] = get_password_hash(user_data.password)
    
    if user_data.disabled is not None:
        users[username]["disabled"] = user_data.disabled
    
    if user_data.is_admin is not None:
        users[username]["is_admin"] = user_data.is_admin
    
    save_users(users)
    
    return User(**users[username])

def list_users():
    """List all users."""
    users = load_users()
    return [User(**user_data) for user_data in users.values()]

# Initialize auth data on import
load_users()
load_api_keys()
