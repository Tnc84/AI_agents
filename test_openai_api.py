import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_openai_api():
    """Test the OpenAI API to verify the connection and API usage."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print("ERROR: OPENAI_API_KEY is not set or is using the default value.")
        return
    
    api_url = "https://api.openai.com/v1/chat/completions"
    model = "gpt-3.5-turbo"  # Using a commonly available model
    
    # Create a simple payload
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides concise responses."},
            {"role": "user", "content": "Hello, how are you today?"}
        ],
        "max_tokens": 150
    }
    
    # Set up headers with API key
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    print("Sending test request to OpenAI API...")
    
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
            if 'choices' in response_json and len(response_json['choices']) > 0:
                print(f"Response content: {response_json['choices'][0]['message']['content']}")
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
    test_openai_api() 