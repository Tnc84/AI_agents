import os
from dotenv import load_dotenv
from agents import HuggingFaceAgent
from core.base import Message

def test_huggingface_agent():
    load_dotenv()
    
    # Get API key (optional)
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if api_key:
        print("Using provided Hugging Face API key")
    else:
        print("Using Hugging Face without API key (rate limited)")
    
    # Test with recommended model
    print("\nTesting with default recommended model...")
    agent = HuggingFaceAgent("TestAgent")
    agent.set_specialization("You are a helpful assistant that provides concise responses.")
    agent.initialize()
    
    # Process a message
    user_message = Message(content="What is machine learning in simple terms?", sender="User")
    print(f"User: {user_message.content}")
    
    response = agent.process_message(user_message)
    print(f"Agent: {response.content}")
    
    # Test with alternative model
    print("\nTesting with alternative model (zephyr-7b-beta)...")
    agent2 = HuggingFaceAgent("TestAgent2", model_id="HuggingFaceH4/zephyr-7b-beta")
    agent2.set_specialization("You are a helpful assistant that provides concise responses.")
    agent2.initialize()
    
    # Process the same message
    print(f"User: {user_message.content}")
    response2 = agent2.process_message(user_message)
    print(f"Agent: {response2.content}")

if __name__ == "__main__":
    test_huggingface_agent() 