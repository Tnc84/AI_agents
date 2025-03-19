import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_anthropic_api():
    """Test the Anthropic API to verify the connection and API usage."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key_here":
        print("ERROR: ANTHROPIC_API_KEY is not set or is using the default value.")
        return
    
    api_url = "https://api.anthropic.com/v1/messages"
    model = "claude-3-haiku-20240307"  # Using a different, more available model
    
    # Create a simple payload
    payload = {
        "model": model,
        "system": "You are a helpful assistant that provides concise responses.",
        "messages": [
            {"role": "user", "content": "Hello, how are you today?"}
        ],
        "max_tokens": 500
    }
    
    # Set up headers with API key
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
        "x-api-key": api_key
    }
    
    print("Sending test request to Anthropic API...")
    
    try:
        response = requests.post(
            api_url,
            headers=headers,
            json=payload
        )
        
        print(f"Response status code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            print("\nAPI call successful!")
            print(f"Model used: {response_json.get('model', 'unknown')}")
            if 'content' in response_json and len(response_json['content']) > 0:
                print(f"Response content: {response_json['content'][0]['text']}")
            else:
                print(f"Unexpected response structure: {json.dumps(response_json, indent=2)}")
        else:
            print(f"API call failed with status code: {response.status_code}")
            try:
                error_json = response.json()
                print(f"Error details: {json.dumps(error_json, indent=2)}")
            except:
                print(f"Error response: {response.text}")
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")

if __name__ == "__main__":
    test_anthropic_api() 