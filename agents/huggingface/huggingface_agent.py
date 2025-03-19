from typing import Dict, Any, List, Optional
import os
import requests
import json
from dotenv import load_dotenv
from core.base import Agent, Message

load_dotenv()

class HuggingFaceAgent(Agent):
    """An agent that uses Hugging Face's inference API."""
    
    def __init__(self, 
                 name: str, 
                 model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 api_key: Optional[str] = None):
        super().__init__(name)
        self.model_id = model_id
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.system_prompt = f"You are {name}, a helpful AI assistant."
        self.specialization = ""
        
        # Some popular instruction-following models that work well on Hugging Face
        self.recommended_models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "HuggingFaceH4/zephyr-7b-beta",
            "google/flan-t5-xxl",
            "tiiuae/falcon-7b-instruct",
            "microsoft/DialoGPT-medium",
            "facebook/blenderbot-400M-distill"
        ]
        
        if model_id not in self.recommended_models:
            print(f"Note: {model_id} is not in the list of recommended models. Recommended models are: {', '.join(self.recommended_models)}")
    
    def initialize(self) -> None:
        """Initialize the Hugging Face agent."""
        pass
    
    def set_specialization(self, specialization: str) -> None:
        """Set the agent's specialization to guide its responses."""
        self.specialization = specialization
    
    def format_prompt(self, message: Message) -> str:
        """Format the message history into a prompt for the model."""
        # Start with system prompt and specialization
        formatted_prompt = f"{self.system_prompt}\n"
        if self.specialization:
            formatted_prompt += f"{self.specialization}\n\n"
        
        # Add conversation history (limited to last 5 messages to keep context manageable)
        for msg in self.message_history[-5:]:
            if msg.sender == self.name:
                formatted_prompt += f"Assistant: {msg.content}\n"
            else:
                formatted_prompt += f"User: {msg.content}\n"
        
        # Add the current message
        formatted_prompt += f"User: {message.content}\nAssistant:"
        
        return formatted_prompt
    
    def process_message(self, message: Message) -> Message:
        """Process a message using Hugging Face's inference API."""
        # Add the incoming message to history
        self.add_to_history(message)
        
        # Format the prompt based on message history
        prompt = self.format_prompt(message)
        
        try:
            # Set up headers (with API key if available)
            headers = {
                "Content-Type": "application/json",
            }
            
            if self.api_key and self.api_key != "your_huggingface_api_key_here":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            # Create the payload
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
            
            print(f"Sending request to Hugging Face Inference API for model {self.model_id}...")
            
            # Make the API call
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                # Process the successful response
                response_json = response.json()
                
                # Extract the generated text
                if isinstance(response_json, list) and len(response_json) > 0:
                    # The API returns a list of generated outputs
                    response_text = response_json[0].get("generated_text", "")
                    
                    # Clean up the response - sometimes it includes the prompt
                    if response_text.startswith(prompt):
                        response_text = response_text[len(prompt):].strip()
                else:
                    response_text = str(response_json)
                
                # Create response message
                response_message = Message(
                    content=response_text,
                    sender=self.name,
                    metadata={"model": self.model_id}
                )
            else:
                # Handle API error
                error_detail = "Unknown error"
                try:
                    error_json = response.json()
                    error_detail = json.dumps(error_json, indent=2)
                except:
                    error_detail = response.text
                
                print(f"API Error Details: {error_detail}")
                
                # Check if the error is related to the model still loading
                if "estimated_time" in response.text:
                    try:
                        error_data = response.json()
                        wait_time = error_data.get("estimated_time", "unknown")
                        response_message = Message(
                            content=f"I'm still warming up. The model is being loaded and will be ready in approximately {wait_time} seconds. Please try again shortly.",
                            sender=self.name,
                            metadata={"error": "model_loading", "wait_time": wait_time}
                        )
                    except:
                        response_message = Message(
                            content="The model is still being loaded. Please try again in a moment.",
                            sender=self.name,
                            metadata={"error": "model_loading"}
                        )
                else:
                    response_message = Message(
                        content=f"Sorry, I couldn't process your request. API error: {response.status_code}",
                        sender=self.name,
                        metadata={"error": f"HTTP {response.status_code}", "details": error_detail}
                    )
                
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            print(f"Exception: {str(e)}\n{traceback_str}")
            
            response_message = Message(
                content=f"I apologize, but I encountered an error when trying to process your query: {str(e)}",
                sender=self.name,
                metadata={"error": str(e)}
            )
        
        self.add_to_history(response_message)
        return response_message 