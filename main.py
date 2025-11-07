"""
Main entry point for the Voice-to-Data Assistant.
Provides a simple CLI interface for processing transcriptions and queries.
"""

import json
import sys
import os
from dotenv import load_dotenv
from voice_assistant import VoiceToDataAssistant

# Load environment variables from .env file
load_dotenv()


def print_json(data: dict):
    """Pretty print JSON data."""
    print(json.dumps(data, indent=2))


def main():
    """Main CLI interface."""
    # Get API keys from environment or user input
    print("=" * 60)
    print("Voice-to-Data Assistant - API Setup")
    print("=" * 60)
    print("\nNote: API keys can be set in .env file or environment variables")
    print("See .env.example for reference\n")
    
    # HuggingFace API key for voice-to-text
    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        hf_key = input("Enter HuggingFace API key (for voice-to-text) or press Enter to skip voice features: ").strip()
        if not hf_key:
            print("Warning: Voice features will be disabled without HuggingFace API key.")
    
    # Ask user if they want to use LLM (optional)
    use_llm_input = input("\nUse LLM for enhanced summaries? (y/n, default=y): ").strip().lower()
    use_llm = use_llm_input != 'n'
    llm_backend = "groq"
    groq_key = None
    
    if use_llm:
        backend_input = input("LLM backend (groq/ollama/openai/huggingface/simple, default=groq): ").strip().lower()
        if backend_input in ['groq', 'ollama', 'openai', 'huggingface', 'simple']:
            llm_backend = backend_input
        
        # Get API key for Groq if selected
        if llm_backend == "groq":
            groq_key = os.getenv('GROQ_API_KEY')
            if not groq_key:
                groq_key = input("Enter Groq API key (or press Enter to use GROQ_API_KEY env var): ").strip()
                if not groq_key:
                    groq_key = None  # Will try env var
    
    assistant = VoiceToDataAssistant(
        use_llm=use_llm, 
        llm_backend=llm_backend,
        groq_api_key=groq_key,
        huggingface_api_key=hf_key if hf_key else None
    )
    
    print("\n" + "=" * 60)
    print("Voice-to-Data Assistant")
    print("=" * 60)
    print("\nCommands:")
    print("  1. Enter transcribed text to extract and save data")
    print("  2. Type 'voice' or 'record' to record from microphone")
    print("  3. Type 'file:<path>' to process an audio file")
    print("  4. Enter a question (starting with '?' or 'query:') to analyze data")
    print("  5. Type 'exit' or 'quit' to exit")
    print("\nExample transcriptions:")
    print("  - 'I made a profit of $500 on March 15th selling old items'")
    print("  - 'Lost $200 yesterday on groceries'")
    print("\nExample voice commands:")
    print("  - 'voice' or 'record' - Record from microphone")
    print("  - 'file:audio.mp3' - Process audio file")
    print("\nExample queries:")
    print("  - '? Show me all profit details for March'")
    print("  - '? How much loss did I have on 12 August?'")
    print("  - '? Generate a monthly profit/loss report'")
    print("\n" + "=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            # Check for voice recording command
            if user_input.lower() in ['voice', 'record', 'mic']:
                try:
                    duration = input("Recording duration in seconds (default=5): ").strip()
                    duration = int(duration) if duration else 5
                    result = assistant.process_voice_microphone(duration)
                    
                    print("\n" + "=" * 60)
                    print("Transcription:")
                    print("=" * 60)
                    print(result.get('transcribed_text', 'N/A'))
                    print("\n" + "=" * 60)
                    print("Extracted Data:")
                    print("=" * 60)
                    print_json(result['json'])
                    print(f"\nStatus: {result['message']}\n")
                except Exception as e:
                    print(f"\nError processing voice: {e}\n")
                continue
            
            # Check for audio file processing
            if user_input.lower().startswith('file:'):
                audio_path = user_input[5:].strip()
                try:
                    result = assistant.process_voice_file(audio_path)
                    
                    print("\n" + "=" * 60)
                    print("Transcription:")
                    print("=" * 60)
                    print(result.get('transcribed_text', 'N/A'))
                    print("\n" + "=" * 60)
                    print("Extracted Data:")
                    print("=" * 60)
                    print_json(result['json'])
                    print(f"\nStatus: {result['message']}\n")
                except Exception as e:
                    print(f"\nError processing audio file: {e}\n")
                continue
            
            # Check if it's a query (starts with '?' or 'query:')
            if user_input.startswith('?') or user_input.lower().startswith('query:'):
                query = user_input.lstrip('?').strip()
                if query.lower().startswith('query:'):
                    query = query[6:].strip()
                
                # Answer the query
                result = assistant.answer_query(query)
                
                print("\n" + "=" * 60)
                print("Response:")
                print("=" * 60)
                print(result['text'])
                
                if result.get('json'):
                    print("\n" + "-" * 60)
                    print("JSON Report:")
                    print("-" * 60)
                    print_json(result['json'])
                print("\n")
            
            else:
                # Process as transcription
                result = assistant.process_transcription(user_input)
                
                print("\n" + "=" * 60)
                print("Extracted Data:")
                print("=" * 60)
                print_json(result['json'])
                print(f"\nStatus: {result['message']}\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

