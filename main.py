from agents import OpenAIAgent, AnthropicAgent, HuggingFaceAgent
from core.base import Message
from core.coordinator import Coordinator
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    # Create coordinator
    coordinator = Coordinator()
    
    # Check available API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
    
    use_openai = openai_key and openai_key != "your_api_key_here"
    use_anthropic = anthropic_key and anthropic_key != "your_anthropic_api_key_here"
    use_huggingface = True  # Can be used without API key (with rate limits)
    
    # Print available APIs
    print("Available APIs:")
    print(f"- OpenAI: {'✓' if use_openai else '✗'}")
    print(f"- Anthropic: {'✓' if use_anthropic else '✗'}")
    print(f"- Hugging Face: {'✓ (API key provided)' if huggingface_key else '✓ (free tier)'}")
    
    # Choose general assistant agent based on available APIs
    if use_openai:
        print("Using OpenAI for general assistant...")
        general_assistant = OpenAIAgent("Assistant")
    elif use_anthropic:
        print("Using Anthropic for general assistant...")
        general_assistant = AnthropicAgent("Assistant")
    else:
        print("Using Hugging Face for general assistant...")
        general_assistant = HuggingFaceAgent("Assistant")
    
    # Set specialization for general assistant
    general_assistant.set_specialization("""You are a helpful general assistant. Answer questions on any topic except for 
    weather-related queries, which will be handled by a specialized agent. If a user asks about weather, 
    kindly inform them that a specialized weather agent can better assist with that question.""")
    
    # Choose weather assistant based on available APIs
    if use_anthropic:
        print("Using Anthropic for weather expert...")
        weather_assistant = AnthropicAgent("WeatherExpert")
    elif use_openai:
        print("Using OpenAI for weather expert...")
        weather_assistant = OpenAIAgent("WeatherExpert")
    else:
        print("Using Hugging Face for weather expert...")
        # Use a different model for the weather agent
        weather_assistant = HuggingFaceAgent("WeatherExpert", model_id="HuggingFaceH4/zephyr-7b-beta")
    
    # Set specialization for weather assistant
    weather_assistant.set_specialization("""You are a weather specialist. When asked about weather in a location, provide detailed 
    information about temperature, conditions, humidity, and forecasts. If you don't have real-time 
    weather data, explain that you're providing general climate information about the region based on historical patterns.
    
    Always try to identify the location in the user's query, even if it's not explicitly stated.
    If the location is ambiguous, ask for clarification. If no location is mentioned, ask which 
    city they're interested in.""")
    
    # Initialize agents
    general_assistant.initialize()
    weather_assistant.initialize()
    
    # Add agents to coordinator
    coordinator.add_agent(general_assistant)
    coordinator.add_agent(weather_assistant)
    
    print("\nMulti-Agent Chatbot System (Type 'exit' to quit)")
    print("Available agents: Assistant, WeatherExpert")
    print("You can switch agents by typing '@AgentName your message'")
    print("Weather-related queries will automatically be routed to the WeatherExpert")
    print("-" * 50)
    
    current_agent = "Assistant"  # Default agent
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            break
        
        # Check for agent switching command
        if user_input.lower().startswith("@"):
            parts = user_input.split(" ", 1)
            agent_name = parts[0][1:]  # Remove @ symbol
            
            if agent_name in coordinator.agents:
                current_agent = agent_name
                print(f"Switched to {current_agent}")
                
                # If there's a message after the agent name, process it
                if len(parts) > 1:
                    user_input = parts[1]
                else:
                    continue
            else:
                print(f"Agent '{agent_name}' not found. Available agents: {', '.join(coordinator.agents.keys())}")
                continue
        
        # Create user message
        user_message = Message(
            content=user_input,
            sender="User"
        )
        
        # Determine which agent to use based on content
        target_agent = current_agent
        weather_keywords = ["weather", "temperature", "forecast", "rain", "sunny", "climate", "humid", "cold", "hot"]
        
        if current_agent != "WeatherExpert" and any(keyword in user_input.lower() for keyword in weather_keywords):
            target_agent = "WeatherExpert"
            print(f"Routing to {target_agent} based on query content...")
        
        # Process message with appropriate agent
        try:
            print(f"Processing message with {target_agent}...")
            response = coordinator.process_message(user_message, target_agent)
            print(f"{response.sender}: {response.content}")
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
        
        print("-" * 50)

if __name__ == "__main__":
    main() 