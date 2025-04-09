import os
import sys
import json
import shutil
import asyncio
import logging
import tempfile
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form, Request, Depends, status, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import subprocess
import uuid
import time
from datetime import datetime

# Import auth module
from api.auth import (
    User, UserCreate, UserUpdate, Token, ApiKey,
    get_current_user, get_current_active_user, get_current_admin_user,
    authenticate_user, create_access_token, create_user, update_user, list_users,
    create_api_key, delete_api_key, get_api_keys_for_user, validate_api_key,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import BrahmaForCausalLM, BrahmaConfig
from inference.generator import BrahmaGenerator
from utils.tokenizer import BrahmaTokenizer
from utils.data_utils import set_seed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brahma LLM API",
    description="API for Brahma LLM - Your own large language model",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(templates_dir, exist_ok=True)
templates = Jinja2Templates(directory=templates_dir)

# Set up static files
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Create routes directory
os.makedirs("uploads", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Define models for API requests and responses
class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True
    seed: Optional[int] = None

class TextGenerationResponse(BaseModel):
    generated_text: str
    prompt: str
    config: Dict[str, Any]

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    config: Dict[str, Any]

class ModelConfig(BaseModel):
    model_path: str = "checkpoints/checkpoint-final.pt"
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None

class TrainingConfig(BaseModel):
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    tokenizer_config: Dict[str, Any]

class TrainingResponse(BaseModel):
    status: str
    message: str
    training_id: str

class TrainingStatusResponse(BaseModel):
    status: str
    message: str
    progress: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None

# Global variables for model and training state
model_instance = None
tokenizer_instance = None
active_trainings = {}

def get_model():
    """Get or initialize the model instance."""
    global model_instance, tokenizer_instance
    
    if model_instance is None:
        try:
            model_path = os.getenv("MODEL_PATH", "checkpoints/checkpoint-final.pt")
            
            # Check if model exists
            if not os.path.exists(model_path):
                available_checkpoints = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
                if available_checkpoints:
                    model_path = os.path.join("checkpoints", available_checkpoints[0])
                    logger.info(f"Using available checkpoint: {model_path}")
                else:
                    # No model available, return None
                    return None, None
            
            # Load tokenizer
            tokenizer_dir = os.path.dirname(model_path)
            try:
                from transformers import AutoTokenizer
                tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer_dir)
            except:
                # Fall back to custom tokenizer
                tokenizer_instance = BrahmaTokenizer.from_pretrained(tokenizer_dir)
            
            # Load config and model
            checkpoint = torch.load(model_path, map_location="cpu")
            
            if "config" in checkpoint:
                config = checkpoint["config"]
                if isinstance(config, dict):
                    config = BrahmaConfig.from_dict(config)
            else:
                config = BrahmaConfig()
            
            model = BrahmaForCausalLM(config)
            
            # Load weights
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            
            # Optimize model for inference
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(device)
            model.eval()
            
            model_instance = BrahmaGenerator(
                model_path=model_path,
                device=device,
                tokenizer=tokenizer_instance,
                config=config,
            )
            
            logger.info(f"Model loaded from {model_path}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    return model_instance, tokenizer_instance

async def run_training(
    training_id: str,
    config_path: str,
    resume_from_checkpoint: Optional[str] = None,
    use_wandb: bool = False
):
    """Run training in background."""
    active_trainings[training_id] = {
        "status": "running",
        "message": "Training started",
        "progress": 0.0,
        "metrics": {"loss": 0.0},
    }
    
    try:
        # Build command
        cmd = [
            sys.executable, "-m", "train",
            "--config", config_path
        ]
        
        if resume_from_checkpoint:
            cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])
        
        if use_wandb:
            cmd.append("--use_wandb")
        
        # Run training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Monitor process output
        while True:
            if process.poll() is not None:
                break
            
            line = process.stdout.readline()
            if line:
                # Parse progress from logs
                if "Step:" in line:
                    try:
                        step_info = line.split("Step:")[1].strip().split(",")
                        step = int(step_info[0])
                        loss = float(step_info[1].split(":")[1].strip())
                        
                        active_trainings[training_id]["progress"] = min(step / 100000, 0.99)
                        active_trainings[training_id]["metrics"]["loss"] = loss
                    except:
                        pass
            
            await asyncio.sleep(0.1)
        
        return_code = process.poll()
        
        if return_code == 0:
            active_trainings[training_id]["status"] = "completed"
            active_trainings[training_id]["message"] = "Training completed successfully"
            active_trainings[training_id]["progress"] = 1.0
        else:
            error_output = process.stderr.read()
            active_trainings[training_id]["status"] = "failed"
            active_trainings[training_id]["message"] = f"Training failed: {error_output}"
    
    except Exception as e:
        active_trainings[training_id]["status"] = "failed"
        active_trainings[training_id]["message"] = f"Training failed: {str(e)}"

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    # Try to load the model
    model, tokenizer = get_model()
    if model is None:
        logger.warning("No model found at startup. Please train or upload a model.")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the home page."""
    model, _ = get_model()
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "model_loaded": model is not None,
        }
    )

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get JWT token."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user": {
            "username": user.username,
            "email": user.email,
            "full_name": user.full_name,
            "is_admin": user.is_admin
        }
    }

@app.post("/users", response_model=User)
async def register_user(user_data: UserCreate):
    """Register a new user."""
    user = create_user(user_data)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    return user

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return current_user

@app.put("/users/me", response_model=User)
async def update_user_me(user_data: UserUpdate, current_user: User = Depends(get_current_active_user)):
    """Update current user information."""
    updated_user = update_user(current_user.username, user_data)
    return updated_user

@app.put("/users/me/password", status_code=status.HTTP_200_OK)
async def change_password(request: Request, current_user: User = Depends(get_current_active_user)):
    """Change user password."""
    try:
        data = await request.json()
        current_password = data.get("current_password")
        new_password = data.get("new_password")
        
        if not current_password or not new_password:
            raise HTTPException(status_code=400, detail="Both current and new password are required")
        
        # Verify current password
        if not authenticate_user(current_user.username, current_password):
            raise HTTPException(status_code=400, detail="Incorrect current password")
        
        # Update the password
        update_user(current_user.username, UserUpdate(password=new_password))
        
        return {"message": "Password updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating password: {str(e)}")

@app.put("/users/me/preferences", status_code=status.HTTP_200_OK)
async def update_preferences(request: Request, current_user: User = Depends(get_current_active_user)):
    """Update user preferences."""
    try:
        preferences = await request.json()
        
        # Here you would store user preferences in your database
        # For simplicity, we'll just return success
        # In a real application, you would have a preferences field in your User model
        # and a function to update it in your auth module
        
        return {"message": "Preferences updated successfully", "preferences": preferences}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating preferences: {str(e)}")

@app.get("/users", response_model=List[User])
async def read_users(current_user: User = Depends(get_current_admin_user)):
    """Get all users (admin only)."""
    return list_users()

@app.post("/api-keys", response_model=ApiKey)
async def create_new_api_key(name: str = Form(...), current_user: User = Depends(get_current_active_user)):
    """Create a new API key for the current user."""
    return create_api_key(current_user.username, name)

@app.get("/api-keys", response_model=List[ApiKey])
async def get_user_api_keys(current_user: User = Depends(get_current_active_user)):
    """Get all API keys for the current user."""
    return get_api_keys_for_user(current_user.username)

@app.delete("/api-keys/{key_id}")
async def remove_api_key(key_id: str, current_user: User = Depends(get_current_active_user)):
    """Delete an API key."""
    success = delete_api_key(key_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    return {"status": "success"}

@app.get("/api-status")
async def api_status():
    """API status endpoint for health checks."""
    model, _ = get_model()
    return {
        "status": "active",
        "model_loaded": model is not None,
        "info": "Brahma LLM API is running. See /docs for API documentation."
    }

@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest, api_key: dict = Depends(validate_api_key)):
    """Generate text from a prompt using API key authentication."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Please train or upload a model first.")
    
    # Check permission
    if "generate" not in api_key["permissions"]:
        raise HTTPException(status_code=403, detail="API key doesn't have permission to generate text")
    
    # Set seed if provided
    if request.seed is not None:
        set_seed(request.seed)
    
    try:
        generated_text = model.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )[0]
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "config": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Route for authenticated web users
# Web Routes
@app.get("/")
async def index_page(request: Request):
    """Render the index page."""
    # Check if a model is loaded
    model_loaded = get_model()[0] is not None
    return templates.TemplateResponse("index.html", {"request": request, "model_loaded": model_loaded})

@app.get("/login")
async def login_page(request: Request):
    """Render the login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register")
async def register_page(request: Request):
    """Render the registration page."""
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/dashboard")
async def dashboard_page(request: Request):
    """Render the dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api-keys")
async def api_keys_page(request: Request):
    """Render the API keys management page."""
    return templates.TemplateResponse("api_keys.html", {"request": request})

@app.get("/chat")
async def chat_page(request: Request):
    """Render the chat interface page."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/account")
async def account_page(request: Request):
    """Render the account settings page."""
    return templates.TemplateResponse("account.html", {"request": request})

@app.get("/playground")
async def playground_page(request: Request):
    """Render the model playground page."""
    return templates.TemplateResponse("playground.html", {"request": request})

@app.post("/api/demo-generate")
async def demo_generate(request: TextGenerationRequest):
    """Generate text for the homepage demo."""
    try:
        model, tokenizer = get_model()
        
        if model is None or tokenizer is None:
            raise HTTPException(status_code=404, detail="Model not loaded")
        
        # Setup generator
        generator = BrahmaGenerator(model, tokenizer)
        
        # Set seed if provided
        if request.seed is not None:
            set_seed(request.seed)
        
        # Generate text
        generated_text = generator.generate(
            prompt=request.prompt,
            max_new_tokens=min(request.max_new_tokens, 150),  # Limit for demo
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample
        )
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/web/models")
async def web_models(current_user: User = Depends(get_current_active_user)):
    """Get available models for web interface."""
    try:
        models = []
        for filename in os.listdir("checkpoints"):
            if filename.endswith(".pt"):
                model_path = os.path.join("checkpoints", filename)
                model_size = os.path.getsize(model_path)
                
                # Size in human-readable format
                if model_size < 1024 * 1024:
                    size_str = f"{model_size / 1024:.1f} KB"
                else:
                    size_str = f"{model_size / (1024 * 1024):.1f} MB"
                
                models.append({
                    "id": filename,
                    "name": filename.replace(".pt", "").replace("checkpoint-", ""),
                    "path": model_path,
                    "size": size_str,
                    "modified": datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y-%m-%d %H:%M")
                })
        
        return models
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/web/user-stats")
async def user_stats(current_user: User = Depends(get_current_active_user)):
    """Get usage statistics for the current user."""
    # This would normally come from a database
    # For demo purposes, we'll return sample data
    return {
        "tokens_generated": 125000,
        "api_calls": 142,
        "model_usage": [
            {"model": "brahma-7b", "tokens": 75000, "calls": 90},
            {"model": "brahma-13b", "tokens": 50000, "calls": 52}
        ],
        "daily_usage": [
            {"date": "2025-04-01", "tokens": 15000},
            {"date": "2025-04-02", "tokens": 22000},
            {"date": "2025-04-03", "tokens": 18000},
            {"date": "2025-04-04", "tokens": 25000},
            {"date": "2025-04-05", "tokens": 12000},
            {"date": "2025-04-06", "tokens": 8000},
            {"date": "2025-04-07", "tokens": 25000}
        ]
    }

@app.get("/account")
async def account_page(request: Request):
    """Render the account settings page."""
    return templates.TemplateResponse("account.html", {"request": request})

# Web API Routes
@app.get("/web/models")
async def get_models(current_user: User = Depends(get_current_active_user)):
    """Get all models for the current user."""
    # Implementation would list models from a directory or database
    models = []
    try:
        models_dir = os.path.join("checkpoints")
        if os.path.exists(models_dir):
            for model_folder in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_folder)
                if os.path.isdir(model_path):
                    # Get model metadata
                    metadata_file = os.path.join(model_path, "metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                            models.append({
                                "id": model_folder,
                                "name": metadata.get("name", model_folder),
                                "size": metadata.get("size", "Unknown"),
                                "created_at": metadata.get("created_at", "")
                            })
                    else:
                        # Add without metadata
                        models.append({
                            "id": model_folder,
                            "name": model_folder,
                            "size": "Unknown",
                            "created_at": ""
                        })
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
    
    return models

@app.get("/web/datasets")
async def get_datasets(current_user: User = Depends(get_current_active_user)):
    """Get all datasets for the current user."""
    # Implementation would list datasets from a directory or database
    datasets = []
    try:
        datasets_dir = os.path.join("uploads", "datasets")
        if os.path.exists(datasets_dir):
            for dataset_file in os.listdir(datasets_dir):
                if dataset_file.endswith(".jsonl"):
                    dataset_path = os.path.join(datasets_dir, dataset_file)
                    file_stats = os.stat(dataset_path)
                    
                    # Count samples in the file
                    sample_count = 0
                    with open(dataset_path, "r") as f:
                        for _ in f:
                            sample_count += 1
                    
                    # Get last modified time
                    modified_time = datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                    
                    datasets.append({
                        "id": dataset_file,
                        "name": os.path.splitext(dataset_file)[0],
                        "size_bytes": file_stats.st_size,
                        "sample_count": sample_count,
                        "uploaded_at": modified_time
                    })
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
    
    return datasets

@app.get("/web/training-jobs")
async def get_training_jobs(current_user: User = Depends(get_current_active_user)):
    """Get all training jobs for the current user."""
    # Implementation would list training jobs from a database
    # This is a placeholder - in a real implementation, you'd retrieve from a database
    return []

@app.post("/web/upload-model")
async def upload_model(
    name: str = Form(...),
    model_file: UploadFile = File(...),
    config_file: Optional[UploadFile] = None,
    current_user: User = Depends(get_current_active_user)
):
    """Upload a model file and optional config."""
    try:
        # Create model directory
        model_id = f"{int(time.time())}_{name.replace(' ', '_')}"
        model_dir = os.path.join("checkpoints", model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(model_dir, "model.bin")
        with open(model_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
        
        # Save config file if provided
        if config_file:
            config_path = os.path.join(model_dir, "config.json")
            with open(config_path, "wb") as f:
                shutil.copyfileobj(config_file.file, f)
        
        # Create metadata file
        metadata = {
            "name": name,
            "size": "Custom",
            "created_at": datetime.now().isoformat(),
            "created_by": current_user.username
        }
        
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        return {"success": True, "model_id": model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")

@app.post("/web/upload-dataset")
async def upload_dataset(
    name: str = Form(...),
    dataset_file: UploadFile = File(...),
    type: str = Form(...),
    current_user: User = Depends(get_current_active_user)
):
    """Upload a dataset file."""
    try:
        # Create datasets directory if it doesn't exist
        datasets_dir = os.path.join("uploads", "datasets")
        os.makedirs(datasets_dir, exist_ok=True)
        
        # Clean the filename and ensure it has the correct extension
        filename = f"{name.replace(' ', '_')}_{type}.jsonl"
        file_path = os.path.join(datasets_dir, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(dataset_file.file, f)
        
        return {"success": True, "dataset_id": filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")

@app.post("/web/start-training")
async def start_training(
    request: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Start a training job with the specified configuration."""
    try:
        # Validate request
        required_fields = ["name", "dataset_id", "model_size", "learning_rate", "batch_size"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        # Generate a training job ID
        job_id = str(uuid.uuid4())
        
        # In a real implementation, you would start the training job here
        # For now, we'll just return success
        return {"success": True, "job_id": job_id}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.post("/api/demo-generate")
async def demo_generate(request: Dict[str, Any]):
    """Generate text for the demo on the home page without authentication."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Please train or upload a model first.")
    
    try:
        prompt = request.get("prompt", "")
        max_new_tokens = min(request.get("max_new_tokens", 100), 100)  # Limit to 100 tokens for demo
        
        generated_text = model.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.0,
            do_sample=True,
        )[0]
        
        return {"generated_text": generated_text, "prompt": prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/web/generate", response_model=TextGenerationResponse)
async def generate_text_web(request: TextGenerationRequest, current_user: User = Depends(get_current_active_user)):
    """Generate text from a prompt for authenticated web users."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Please train or upload a model first.")
    
    # Set seed if provided
    if request.seed is not None:
        set_seed(request.seed)
    
    try:
        generated_text = model.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )[0]
        
        return {
            "generated_text": generated_text,
            "prompt": request.prompt,
            "config": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
                "do_sample": request.do_sample,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, api_key: dict = Depends(validate_api_key)):
    """Generate chat responses using API key authentication."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Please train or upload a model first.")
    
    # Check permission
    if "chat" not in api_key["permissions"]:
        raise HTTPException(status_code=403, detail="API key doesn't have permission to use chat")
    
    # Set seed if provided
    if request.seed is not None:
        set_seed(request.seed)
    
    try:
        response = model.chat(
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        return {
            "response": response,
            "config": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

# Route for authenticated web users
@app.post("/web/chat", response_model=ChatResponse)
async def chat_web(request: ChatRequest, current_user: User = Depends(get_current_active_user)):
    """Generate chat responses for authenticated web users."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded. Please train or upload a model first.")
    
    # Set seed if provided
    if request.seed is not None:
        set_seed(request.seed)
    
    try:
        response = model.chat(
            messages=request.messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
        
        return {
            "response": response,
            "config": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "top_k": request.top_k,
                "top_p": request.top_p,
                "repetition_penalty": request.repetition_penalty,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")

@app.post("/upload-model")
async def upload_model(
    model_file: UploadFile = File(...),
    tokenizer_file: Optional[UploadFile] = File(None),
    config_file: Optional[UploadFile] = File(None),
):
    """Upload a trained model."""
    global model_instance, tokenizer_instance
    
    try:
        # Reset model instance
        model_instance = None
        tokenizer_instance = None
        
        # Save model file
        model_path = os.path.join("checkpoints", "uploaded_model.pt")
        with open(model_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)
        
        # Save tokenizer if provided
        if tokenizer_file:
            tokenizer_dir = os.path.join("checkpoints")
            with open(os.path.join(tokenizer_dir, tokenizer_file.filename), "wb") as f:
                shutil.copyfileobj(tokenizer_file.file, f)
        
        # Save config if provided
        if config_file:
            config_path = os.path.join("checkpoints", "config.json")
            with open(config_path, "wb") as f:
                shutil.copyfileobj(config_file.file, f)
        
        # Load the model
        model, tokenizer = get_model()
        
        if model is None:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to load uploaded model"}
            )
        
        return {
            "status": "success",
            "message": "Model uploaded and loaded successfully"
        }
    
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")
    
    finally:
        # Close file objects
        model_file.file.close()
        if tokenizer_file:
            tokenizer_file.file.close()
        if config_file:
            config_file.file.close()

@app.post("/upload-dataset")
async def upload_dataset(
    train_file: UploadFile = File(...),
    validation_file: Optional[UploadFile] = File(None),
    dataset_name: str = Form(...),
):
    """Upload a dataset for training."""
    try:
        # Create dataset directory
        dataset_dir = os.path.join("data", dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save training file
        train_path = os.path.join(dataset_dir, "train.jsonl")
        with open(train_path, "wb") as f:
            shutil.copyfileobj(train_file.file, f)
        
        # Save validation file if provided
        if validation_file:
            validation_path = os.path.join(dataset_dir, "validation.jsonl")
            with open(validation_path, "wb") as f:
                shutil.copyfileobj(validation_file.file, f)
        
        return {
            "status": "success",
            "message": "Dataset uploaded successfully",
            "dataset_path": dataset_dir
        }
    
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload dataset: {str(e)}")
    
    finally:
        # Close file objects
        train_file.file.close()
        if validation_file:
            validation_file.file.close()

@app.post("/start-training", response_model=TrainingResponse)
async def start_training(
    config: TrainingConfig,
    background_tasks: BackgroundTasks,
    resume_from_checkpoint: Optional[str] = None,
    use_wandb: bool = False,
):
    """Start model training."""
    try:
        # Generate training ID
        import uuid
        training_id = str(uuid.uuid4())
        
        # Save configuration to file
        config_path = os.path.join("checkpoints", f"training_config_{training_id}.json")
        with open(config_path, "w") as f:
            json.dump(config.dict(), f, indent=2)
        
        # Start training in background
        background_tasks.add_task(
            run_training,
            training_id=training_id,
            config_path=config_path,
            resume_from_checkpoint=resume_from_checkpoint,
            use_wandb=use_wandb
        )
        
        return {
            "status": "started",
            "message": "Training started in background",
            "training_id": training_id
        }
    
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@app.get("/training-status/{training_id}", response_model=TrainingStatusResponse)
async def training_status(training_id: str):
    """Get training status."""
    if training_id not in active_trainings:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return active_trainings[training_id]

@app.get("/list-models")
async def list_models():
    """List available trained models."""
    try:
        models = []
        for filename in os.listdir("checkpoints"):
            if filename.endswith(".pt"):
                model_path = os.path.join("checkpoints", filename)
                models.append({
                    "name": filename,
                    "path": model_path,
                    "size": os.path.getsize(model_path),
                    "modified": os.path.getmtime(model_path),
                })
        
        return {"models": models}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@app.get("/list-datasets")
async def list_datasets():
    """List available datasets."""
    try:
        datasets = []
        for dirname in os.listdir("data"):
            dir_path = os.path.join("data", dirname)
            if os.path.isdir(dir_path):
                train_path = os.path.join(dir_path, "train.jsonl")
                validation_path = os.path.join(dir_path, "validation.jsonl")
                
                dataset_info = {
                    "name": dirname,
                    "path": dir_path,
                    "has_training": os.path.exists(train_path),
                    "has_validation": os.path.exists(validation_path),
                }
                
                if os.path.exists(train_path):
                    dataset_info["train_size"] = os.path.getsize(train_path)
                
                datasets.append(dataset_info)
        
        return {"datasets": datasets}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")

@app.get("/model-config")
async def model_config():
    """Get current model configuration."""
    model, _ = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_config": model.model.config.__dict__ if hasattr(model, "model") else {},
    }

@app.post("/reload-model")
async def reload_model(config: ModelConfig):
    """Reload model from a specified path."""
    global model_instance, tokenizer_instance
    
    # Reset model instance
    model_instance = None
    tokenizer_instance = None
    
    # Set environment variable for model path
    os.environ["MODEL_PATH"] = config.model_path
    
    # Try to load the model
    model, tokenizer = get_model()
    
    if model is None:
        raise HTTPException(status_code=404, detail="Failed to load model from specified path")
    
    return {
        "status": "success",
        "message": f"Model reloaded from {config.model_path}"
    }

@app.get("/default-config")
async def default_config():
    """Get default configuration for training."""
    default_config_path = os.path.join("config", "default_config.json")
    
    if not os.path.exists(default_config_path):
        raise HTTPException(status_code=404, detail="Default configuration not found")
    
    with open(default_config_path, "r") as f:
        config = json.load(f)
    
    return config

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
