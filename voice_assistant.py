"""
Main Voice-to-Data Assistant class that orchestrates data extraction and analysis.
"""

import json
import os
from typing import Dict, Optional
from dotenv import load_dotenv
from data_extractor import DataExtractor
from csv_manager import CSVManager
from query_analyzer import QueryAnalyzer
from voice_to_text import VoiceToText

# Load environment variables from .env file
load_dotenv()


class VoiceToDataAssistant:
    """
    Main assistant class for processing voice transcriptions and answering queries.
    """
    
    def __init__(self, csv_file: str = "financial_records.csv", 
                 use_llm: bool = True, llm_backend: str = "groq",
                 groq_api_key: Optional[str] = None,
                 huggingface_api_key: Optional[str] = None):
        """
        Initialize the Voice-to-Data Assistant.
        
        Args:
            csv_file: Path to the CSV file for storing records
            use_llm: Whether to use LLM for enhanced summaries
            llm_backend: LLM backend to use ('groq', 'ollama', 'openai', 'huggingface', or 'simple')
            groq_api_key: Groq API key (optional, uses GROQ_API_KEY env var if not provided)
            huggingface_api_key: HuggingFace API key for voice-to-text (optional, uses HUGGINGFACE_API_KEY env var if not provided)
        """
        self.extractor = DataExtractor()
        self.csv_manager = CSVManager(csv_file)
        self.query_analyzer = QueryAnalyzer(
            self.csv_manager, 
            use_llm=use_llm, 
            llm_backend=llm_backend,
            llm_api_key=groq_api_key or os.getenv('GROQ_API_KEY') if llm_backend == "groq" else None
        )
        self.huggingface_api_key = huggingface_api_key or os.getenv('HUGGINGFACE_API_KEY')
        self.voice_to_text = None  # Lazy loading
    
    def process_voice_file(self, audio_file_path: str) -> Dict:
        """
        Process voice/audio file: convert to text, extract data, and save to CSV.
        
        Args:
            audio_file_path: Path to audio file (mp3, wav, m4a, etc.)
            
        Returns:
            Dictionary with 'json' key containing the extracted data
        """
        # Initialize voice-to-text if needed
        if self.voice_to_text is None:
            self.voice_to_text = VoiceToText(api_key=self.huggingface_api_key)
        
        # Convert voice to text
        transcribed_text = self.voice_to_text.transcribe_file(audio_file_path)
        
        # Process the transcription
        return self.process_transcription(transcribed_text)
    
    def process_voice_microphone(self, duration: int = 5) -> Dict:
        """
        Process voice from microphone: record, convert to text, extract data, and save to CSV.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Dictionary with 'json' key containing the extracted data
        """
        # Initialize voice-to-text if needed
        if self.voice_to_text is None:
            self.voice_to_text = VoiceToText(api_key=self.huggingface_api_key)
        
        # Convert voice to text
        transcribed_text = self.voice_to_text.transcribe_microphone(duration)
        
        # Process the transcription
        return self.process_transcription(transcribed_text)
    
    def process_transcription(self, transcribed_text: str) -> Dict:
        """
        Process transcribed voice text and extract structured data.
        
        Args:
            transcribed_text: Text transcribed from voice recording
            
        Returns:
            Dictionary with 'json' key containing the extracted data in the required format
        """
        # Extract structured data
        extracted_data = self.extractor.extract(transcribed_text)
        
        # Save to CSV
        success = self.csv_manager.save_record(extracted_data)
        
        if success:
            return {
                'json': extracted_data,
                'message': 'Record saved successfully',
                'transcribed_text': transcribed_text
            }
        else:
            return {
                'json': extracted_data,
                'message': 'Data extracted but failed to save to CSV',
                'transcribed_text': transcribed_text
            }
    
    def answer_query(self, query: str, csv_data: Optional[str] = None) -> Dict:
        """
        Answer a query by analyzing CSV data.
        
        Args:
            query: User's question/query
            csv_data: Optional CSV data as string (if provided, will be used instead of file)
            
        Returns:
            Dictionary with 'text' (human-readable) and optional 'json' (structured report)
        """
        # If CSV data is provided as string, we would need to parse it
        # For now, we'll use the CSV file directly
        # In a real implementation, you might want to support both
        
        return self.query_analyzer.analyze(query)
    
    def get_all_records(self) -> list:
        """
        Get all records from CSV.
        
        Returns:
            List of all records
        """
        return self.csv_manager.read_all_records()
    
    def export_json(self) -> str:
        """
        Export all records as JSON.
        
        Returns:
            JSON string of all records
        """
        records = self.get_all_records()
        return json.dumps(records, indent=2)

