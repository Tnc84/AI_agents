from typing import Dict, Any
import os
import requests
import json
from dotenv import load_dotenv
from core.base import Agent, Message

load_dotenv()

class OpenAIAgent(Agent):
    """An agent that uses OpenAI's API directly via REST API."""
    
    def __init__(self, name: str, model: str = "gpt-3.5-turbo"):
        super().__init__(name)
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.system_prompt = f"You are {name}, a helpful AI assistant."
        self.specialization = ""
        
        # Print warning if model is not supported
        self.supported_models = [
            "gpt-4", 
            "gpt-4-turbo", 
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        
        if model not in self.supported_models:
            print(f"Warning: Model '{model}' might not be supported. Supported models are: {', '.join(self.supported_models)}")
    
    def initialize(self) -> None:
        """Initialize the OpenAI agent."""
        # Check if API key is set
        if not self.api_key or self.api_key == "your_api_key_here":
            print(f"Warning: OPENAI_API_KEY is not set or is using the default value for {self.name}.")
    
    def set_specialization(self, specialization: str) -> None:
        """Set the agent's specialization to guide its responses."""
        self.specialization = specialization
        
    def process_message(self, message: Message) -> Message:
        """Process an incoming message using OpenAI's API directly."""
        # Add the incoming message to history
        self.add_to_history(message)
        
        # Check if API key is available
        if not self.api_key or self.api_key == "your_api_key_here":
            return Message(
                content="I'm sorry, but I need a valid OpenAI API key to work. Please update your .env file with your OpenAI API key.",
                sender=self.name,
                metadata={"error": "No API key"}
            )
        
        # Build the full system prompt with specialization if provided
        full_system_prompt = self.system_prompt
        if self.specialization:
            full_system_prompt += f"\n\n{self.specialization}"
        
        # Convert message history to messages for OpenAI
        messages = [
            {"role": "system", "content": full_system_prompt}
        ]
        
        for msg in self.message_history[-5:]:  # Limit to last 5 messages to avoid large context
            role = "assistant" if msg.sender == self.name else "user"
            messages.append({"role": role, "content": msg.content})
        
        # Ensure last message is the current one
        if len(messages) < 2 or messages[-1]["content"] != message.content:
            messages.append({"role": "user", "content": message.content})
        
        try:
            # Create the API request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # Make the API request
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Debug info
            print(f"Sending request to OpenAI API for {self.name}...")
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload
            )
            
            # Debug info
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Check if the response has the expected structure
                if 'choices' in response_json and len(response_json['choices']) > 0 and 'message' in response_json['choices'][0]:
                    response_text = response_json['choices'][0]['message']['content']
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
                        error_message = f"I apologize, but I encountered an error: {error_json['error']['message']}"
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