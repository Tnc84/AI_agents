from typing import Dict, Any
import os
import requests
import json
from dotenv import load_dotenv
from core.base import Agent, Message

load_dotenv()

class AnthropicAgent(Agent):
    """An agent that uses Anthropic's API directly via REST API."""
    
    def __init__(self, name: str, model: str = "claude-3-haiku-20240307"):
        super().__init__(name)
        self.model = model
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.system_prompt = f"You are {name}, a helpful AI assistant."
        self.specialization = ""
        
        # Print warning if model is not supported
        self.supported_models = [
            "claude-3-opus-20240229", 
            "claude-3-sonnet-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1", 
            "claude-2.0", 
            "claude-instant-1.2"
        ]
        
        if model not in self.supported_models:
            print(f"Warning: Model '{model}' might not be supported. Supported models are: {', '.join(self.supported_models)}")
    
    def initialize(self) -> None:
        """Initialize the Anthropic agent."""
        # Check if API key is set
        if not self.api_key or self.api_key == "your_anthropic_api_key_here":
            print(f"Warning: ANTHROPIC_API_KEY is not set or is using the default value for {self.name}.")
    
    def set_specialization(self, specialization: str) -> None:
        """Set the agent's specialization to guide its responses."""
        self.specialization = specialization
        
    def process_message(self, message: Message) -> Message:
        """Process an incoming message using Anthropic's API directly."""
        # Add the incoming message to history
        self.add_to_history(message)
        
        # Check if API key is available
        if not self.api_key or self.api_key == "your_anthropic_api_key_here":
            return Message(
                content="I'm sorry, but I need a valid Anthropic API key to work. Please update your .env file with your Anthropic API key.",
                sender=self.name,
                metadata={"error": "No API key"}
            )
        
        # Build the full system prompt with specialization if provided
        full_system_prompt = self.system_prompt
        if self.specialization:
            full_system_prompt += f"\n\n{self.specialization}"
        
        # Convert message history to messages for Anthropic
        messages = []
        
        for msg in self.message_history[-5:]:  # Limit to last 5 messages to avoid large context
            role = "assistant" if msg.sender == self.name else "user"
            messages.append({"role": role, "content": msg.content})
        
        # Ensure last message is the current one
        if not messages or messages[-1]["content"] != message.content:
            messages.append({"role": "user", "content": message.content})
        
        try:
            # Create the API request payload with correct format
            payload = {
                "model": self.model,
                "system": full_system_prompt,
                "messages": messages,
                "max_tokens": 1000
            }
            
            # Make the API request with updated headers
            headers = {
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.api_key
            }
            
            # Debug info
            print(f"Sending request to Anthropic API for {self.name}...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            # Debug info
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                # Debug the response structure
                if 'content' in response_json and len(response_json['content']) > 0 and 'text' in response_json['content'][0]:
                    response_text = response_json['content'][0]['text']
                else:
                    print(f"Unexpected response structure: {json.dumps(response_json, indent=2)}")
                    response_text = "Sorry, I received an unexpected response format from the API."
                
                # Create response message
                response_message = Message(
                    content=response_text,
                    sender=self.name,
                    metadata={"model": self.model}
                )
            else:
                # Handle API error with more details
                error_detail = "Unknown error"
                error_message = f"Sorry, I couldn't process your request. API error: {response.status_code}"
                
                try:
                    error_json = response.json()
                    error_detail = json.dumps(error_json, indent=2)
                    
                    # Check for specific error cases
                    if "error" in error_json and "message" in error_json["error"]:
                        if "credit balance is too low" in error_json["error"]["message"]:
                            error_message = """I apologize, but your Anthropic API account doesn't have enough credits to process this request.
                            
Please visit https://console.anthropic.com/ to purchase credits or upgrade your plan."""
                except:
                    error_detail = response.text
                
                print(f"API Error Details: {error_detail}")
                
                response_message = Message(
                    content=error_message,
                    sender=self.name,
                    metadata={"error": f"HTTP {response.status_code}", "details": error_detail}
                )
        except Exception as e:
            # If something goes wrong, provide a fallback response
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