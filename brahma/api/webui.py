import os
import sys
import json
import time
import requests
import gradio as gr
from pathlib import Path
import webbrowser
from typing import List, Dict, Optional, Any

# Constants
API_URL = "http://localhost:8000"  # API server URL

def check_api_status():
    """Check if API server is running."""
    try:
        response = requests.get(f"{API_URL}")
        if response.status_code == 200:
            data = response.json()
            return data.get("model_loaded", False)
        return False
    except:
        return False

def generate_text(prompt, max_tokens, temperature, top_p, top_k, repetition_penalty, do_sample):
    """Generate text via API."""
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["generated_text"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def chat_response(message, history):
    """Handle chat messages."""
    messages = []
    
    # Convert history to message format
    for h in history:
        messages.append({"role": "user", "content": h[0]})
        messages.append({"role": "assistant", "content": h[1]})
    
    # Add the new message
    messages.append({"role": "user", "content": message})
    
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "messages": messages,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return data["response"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def list_models():
    """Get list of available models."""
    try:
        response = requests.get(f"{API_URL}/list-models")
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return models
        return []
    except:
        return []

def list_datasets():
    """Get list of available datasets."""
    try:
        response = requests.get(f"{API_URL}/list-datasets")
        if response.status_code == 200:
            data = response.json()
            datasets = data.get("datasets", [])
            return datasets
        return []
    except:
        return []

def reload_model(model_name):
    """Reload a different model."""
    try:
        if not model_name:
            return "No model selected"
        
        models = list_models()
        selected_model = next((m for m in models if m["name"] == model_name), None)
        
        if not selected_model:
            return f"Model {model_name} not found"
        
        response = requests.post(
            f"{API_URL}/reload-model",
            json={
                "model_path": selected_model["path"]
            }
        )
        
        if response.status_code == 200:
            return f"Model {model_name} loaded successfully"
        else:
            return f"Error loading model: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def upload_model(model_file, tokenizer_file, config_file):
    """Upload a model to the server."""
    try:
        files = {
            "model_file": (os.path.basename(model_file.name), model_file, "application/octet-stream")
        }
        
        if tokenizer_file:
            files["tokenizer_file"] = (os.path.basename(tokenizer_file.name), tokenizer_file, "application/octet-stream")
        
        if config_file:
            files["config_file"] = (os.path.basename(config_file.name), config_file, "application/octet-stream")
        
        response = requests.post(f"{API_URL}/upload-model", files=files)
        
        if response.status_code == 200:
            return "Model uploaded successfully"
        else:
            return f"Error uploading model: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def upload_dataset(train_file, validation_file, dataset_name):
    """Upload a dataset to the server."""
    try:
        if not dataset_name:
            return "Please provide a dataset name"
        
        files = {
            "train_file": (os.path.basename(train_file.name), train_file, "application/octet-stream"),
            "dataset_name": (None, dataset_name)
        }
        
        if validation_file:
            files["validation_file"] = (os.path.basename(validation_file.name), validation_file, "application/octet-stream")
        
        response = requests.post(f"{API_URL}/upload-dataset", files=files)
        
        if response.status_code == 200:
            return f"Dataset '{dataset_name}' uploaded successfully"
        else:
            return f"Error uploading dataset: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def start_training(dataset_name, model_size, batch_size, learning_rate, epochs, use_wandb, resume_checkpoint):
    """Start training a model."""
    try:
        # Get default config
        response = requests.get(f"{API_URL}/default-config")
        if response.status_code != 200:
            return "Error getting default configuration"
        
        default_config = response.json()
        
        # Update config with user parameters
        if model_size == "Small (~1B params)":
            default_config["model_config"]["hidden_size"] = 2048
            default_config["model_config"]["num_hidden_layers"] = 24
            default_config["model_config"]["num_attention_heads"] = 16
        elif model_size == "Medium (~7B params)":
            default_config["model_config"]["hidden_size"] = 4096
            default_config["model_config"]["num_hidden_layers"] = 32
            default_config["model_config"]["num_attention_heads"] = 32
        elif model_size == "Large (~13B params)":
            default_config["model_config"]["hidden_size"] = 5120
            default_config["model_config"]["num_hidden_layers"] = 40
            default_config["model_config"]["num_attention_heads"] = 40
        
        # Set data config
        default_config["data_config"]["dataset_path"] = f"data/{dataset_name}"
        
        # Set training config
        default_config["training_config"]["batch_size"] = batch_size
        default_config["training_config"]["learning_rate"] = learning_rate
        default_config["training_config"]["num_train_epochs"] = epochs
        
        # Start training
        train_params = {
            "config": default_config,
            "use_wandb": use_wandb
        }
        
        if resume_checkpoint and resume_checkpoint != "None":
            train_params["resume_from_checkpoint"] = resume_checkpoint
        
        response = requests.post(f"{API_URL}/start-training", json=train_params)
        
        if response.status_code == 200:
            data = response.json()
            training_id = data.get("training_id", "unknown")
            return f"Training started with ID: {training_id}"
        else:
            return f"Error starting training: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def check_training_status(training_id):
    """Check the status of a training job."""
    try:
        if not training_id:
            return "No training ID provided"
        
        response = requests.get(f"{API_URL}/training-status/{training_id}")
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "unknown")
            message = data.get("message", "")
            progress = data.get("progress", 0.0)
            
            if progress is not None:
                progress_percent = f"{progress * 100:.2f}%"
            else:
                progress_percent = "unknown"
            
            if status == "completed":
                return f"Training completed: {message}"
            elif status == "failed":
                return f"Training failed: {message}"
            else:
                return f"Training in progress - {progress_percent} complete: {message}"
        else:
            return f"Error checking training status: {response.text}"
    except Exception as e:
        return f"Error: {str(e)}"

def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="Brahma LLM", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Brahma LLM Interface")
        
        # Check if API is running
        api_running = check_api_status()
        if not api_running:
            gr.Markdown("""
            ## ⚠️ API Server Not Running
            
            The Brahma API server is not running. Please start it with:
            ```
            python -m api.server
            ```
            """)
        
        with gr.Tabs():
            # Chat Interface
            with gr.TabItem("Chat"):
                chat_interface = gr.ChatInterface(
                    chat_response,
                    title="Chat with Brahma",
                    description="Have a conversation with the Brahma LLM model",
                    examples=[
                        "Tell me about yourself",
                        "What can you do?",
                        "Write a short story about a robot learning to paint",
                        "Explain quantum computing to a 10-year-old"
                    ]
                )
            
            # Text Generation
            with gr.TabItem("Text Generation"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=5
                        )
                        
                        with gr.Row():
                            max_tokens = gr.Slider(
                                minimum=1, 
                                maximum=1024, 
                                value=256, 
                                step=1, 
                                label="Max New Tokens"
                            )
                            temperature = gr.Slider(
                                minimum=0.1, 
                                maximum=2.0, 
                                value=0.7, 
                                step=0.1, 
                                label="Temperature"
                            )
                        
                        with gr.Row():
                            top_p = gr.Slider(
                                minimum=0.1, 
                                maximum=1.0, 
                                value=0.9, 
                                step=0.1, 
                                label="Top-p"
                            )
                            top_k = gr.Slider(
                                minimum=1, 
                                maximum=100, 
                                value=50, 
                                step=1, 
                                label="Top-k"
                            )
                        
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                minimum=1.0, 
                                maximum=2.0, 
                                value=1.1, 
                                step=0.1, 
                                label="Repetition Penalty"
                            )
                            do_sample = gr.Checkbox(
                                label="Use Sampling", 
                                value=True
                            )
                        
                        generate_btn = gr.Button("Generate")
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Generated Text",
                            lines=20,
                            interactive=False
                        )
                
                generate_btn.click(
                    generate_text,
                    inputs=[
                        prompt_input, 
                        max_tokens, 
                        temperature, 
                        top_p, 
                        top_k, 
                        repetition_penalty, 
                        do_sample
                    ],
                    outputs=output_text
                )
            
            # Model Management
            with gr.TabItem("Model Management"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Model")
                        
                        model_file = gr.File(label="Model Checkpoint (PT file)")
                        tokenizer_file = gr.File(label="Tokenizer (Optional)")
                        config_file = gr.File(label="Config (Optional)")
                        
                        upload_model_btn = gr.Button("Upload Model")
                        upload_model_output = gr.Textbox(label="Upload Status", interactive=False)
                        
                        upload_model_btn.click(
                            upload_model,
                            inputs=[model_file, tokenizer_file, config_file],
                            outputs=upload_model_output
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Model Selection")
                        
                        refresh_models_btn = gr.Button("Refresh Models")
                        
                        def get_model_dropdown():
                            models = list_models()
                            return gr.Dropdown.update(
                                choices=[m["name"] for m in models],
                                value=models[0]["name"] if models else None
                            )
                        
                        model_dropdown = gr.Dropdown(
                            label="Available Models",
                            choices=[],
                            interactive=True
                        )
                        
                        refresh_models_btn.click(
                            get_model_dropdown,
                            inputs=[],
                            outputs=model_dropdown
                        )
                        
                        load_model_btn = gr.Button("Load Selected Model")
                        load_model_output = gr.Textbox(label="Load Status", interactive=False)
                        
                        load_model_btn.click(
                            reload_model,
                            inputs=[model_dropdown],
                            outputs=load_model_output
                        )
            
            # Training
            with gr.TabItem("Training"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Dataset")
                        
                        train_file = gr.File(label="Training Data (JSONL)")
                        validation_file = gr.File(label="Validation Data (Optional)")
                        dataset_name = gr.Textbox(label="Dataset Name", placeholder="my_dataset")
                        
                        upload_dataset_btn = gr.Button("Upload Dataset")
                        upload_dataset_output = gr.Textbox(label="Upload Status", interactive=False)
                        
                        upload_dataset_btn.click(
                            upload_dataset,
                            inputs=[train_file, validation_file, dataset_name],
                            outputs=upload_dataset_output
                        )
                    
                    with gr.Column():
                        gr.Markdown("### Start Training")
                        
                        refresh_datasets_btn = gr.Button("Refresh Datasets")
                        
                        def get_dataset_dropdown():
                            datasets = list_datasets()
                            return gr.Dropdown.update(
                                choices=[d["name"] for d in datasets],
                                value=datasets[0]["name"] if datasets else None
                            )
                        
                        dataset_dropdown = gr.Dropdown(
                            label="Available Datasets",
                            choices=[],
                            interactive=True
                        )
                        
                        refresh_datasets_btn.click(
                            get_dataset_dropdown,
                            inputs=[],
                            outputs=dataset_dropdown
                        )
                        
                        model_size = gr.Dropdown(
                            label="Model Size",
                            choices=[
                                "Small (~1B params)",
                                "Medium (~7B params)", 
                                "Large (~13B params)"
                            ],
                            value="Small (~1B params)"
                        )
                        
                        with gr.Row():
                            batch_size = gr.Slider(
                                minimum=1, 
                                maximum=64, 
                                value=8, 
                                step=1, 
                                label="Batch Size"
                            )
                            learning_rate = gr.Slider(
                                minimum=1e-6, 
                                maximum=1e-3, 
                                value=3e-4, 
                                step=1e-6, 
                                label="Learning Rate"
                            )
                        
                        epochs = gr.Slider(
                            minimum=1, 
                            maximum=10, 
                            value=3, 
                            step=1, 
                            label="Epochs"
                        )
                        
                        use_wandb = gr.Checkbox(label="Use Weights & Biases", value=False)
                        
                        def get_checkpoints_dropdown():
                            models = list_models()
                            choices = ["None"] + [m["name"] for m in models]
                            return gr.Dropdown.update(choices=choices, value="None")
                        
                        resume_checkpoint = gr.Dropdown(
                            label="Resume from Checkpoint",
                            choices=["None"],
                            value="None"
                        )
                        
                        refresh_checkpoints_btn = gr.Button("Refresh Checkpoints")
                        refresh_checkpoints_btn.click(
                            get_checkpoints_dropdown,
                            inputs=[],
                            outputs=resume_checkpoint
                        )
                        
                        start_training_btn = gr.Button("Start Training")
                        training_output = gr.Textbox(label="Training Status", interactive=False)
                        
                        start_training_btn.click(
                            start_training,
                            inputs=[
                                dataset_dropdown,
                                model_size,
                                batch_size,
                                learning_rate,
                                epochs,
                                use_wandb,
                                resume_checkpoint
                            ],
                            outputs=training_output
                        )
                
                with gr.Row():
                    gr.Markdown("### Check Training Status")
                    
                    training_id = gr.Textbox(label="Training ID")
                    check_status_btn = gr.Button("Check Status")
                    training_status_output = gr.Textbox(label="Status", interactive=False)
                    
                    check_status_btn.click(
                        check_training_status,
                        inputs=[training_id],
                        outputs=training_status_output
                    )
        
        # Refresh data on load
        demo.load(
            get_model_dropdown,
            inputs=None,
            outputs=model_dropdown
        )
        
        demo.load(
            get_dataset_dropdown,
            inputs=None,
            outputs=dataset_dropdown
        )
        
        demo.load(
            get_checkpoints_dropdown,
            inputs=None,
            outputs=resume_checkpoint
        )
    
    return demo

def main():
    """Main function to start the UI."""
    demo = create_ui()
    demo.launch(share=True)

if __name__ == "__main__":
    main()
