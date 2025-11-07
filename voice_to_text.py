"""
Voice-to-Text module using HuggingFace Inference API.
Converts audio/voice input to text transcription via API (no local model loading).
"""

import os
import sys
import base64
from typing import Optional
import requests


class VoiceToText:
    """Converts voice/audio to text using HuggingFace Inference API or local Whisper model."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "openai/whisper-large-v3", 
                 use_local: bool = False, local_model: str = "base"):
        """
        Initialize voice-to-text client.
        
        Args:
            api_key: HuggingFace API key. If None and use_local=False, will try to get from environment variable HUGGINGFACE_API_KEY
            model: HuggingFace model identifier (default: openai/whisper-large-v3) - only used if use_local=False
            use_local: If True, use local Whisper model instead of API
            local_model: Local Whisper model size - "tiny", "base", "small", "medium", "large" (default: "base")
        """
        self.use_local = use_local
        self.local_model = local_model
        
        if use_local:
            # Initialize local Whisper model
            try:
                import whisper
                self.whisper_model = whisper.load_model(local_model)
                print(f"Local Whisper model '{local_model}' loaded successfully")
            except ImportError:
                raise ImportError(
                    "openai-whisper not installed. Install with: pip install openai-whisper"
                )
            except Exception as e:
                raise Exception(f"Failed to load local Whisper model: {e}")
        else:
            # Initialize HuggingFace API client
            self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
            if not self.api_key:
                print("Warning: HuggingFace API key not found. Falling back to local Whisper model.")
                try:
                    import whisper
                    self.use_local = True
                    self.local_model = "base"
                    self.whisper_model = whisper.load_model("base")
                    print("Local Whisper model 'base' loaded as fallback")
                except ImportError:
                    raise ValueError(
                        "HuggingFace API key required and local Whisper not available. "
                        "Either set HUGGINGFACE_API_KEY environment variable or install openai-whisper: pip install openai-whisper"
                    )
                except Exception as e:
                    raise ValueError(
                        f"HuggingFace API key required. Set HUGGINGFACE_API_KEY environment variable "
                        f"or pass api_key parameter. Local fallback failed: {e}"
                    )
            
            if not self.use_local:
                self.model = model
                # Updated to use new HuggingFace router endpoint
                self.api_url = f"https://router.huggingface.co/hf-inference/models/{model}"
                self.headers = {
                    "Authorization": f"Bearer {self.api_key}"
                }
                print(f"HuggingFace API client initialized with model: {model}")
    
    def transcribe_file(self, audio_file_path: str) -> str:
        """
        Transcribe audio file to text using HuggingFace Inference API or local Whisper model.
        
        Args:
            audio_file_path: Path to audio file (mp3, wav, m4a, mp4, mpeg, mpga, webm, flac)
            
        Returns:
            Transcribed text
        """
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        # Use local Whisper model if enabled
        if self.use_local:
            return self._transcribe_local(audio_file_path)
        
        # Otherwise use HuggingFace API
        try:
            print(f"Transcribing audio file via HuggingFace API: {audio_file_path}")
            
            # Read audio file
            with open(audio_file_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            
            # Determine content type based on file extension
            ext = os.path.splitext(audio_file_path)[1].lower()
            content_type_map = {
                '.wav': 'audio/wav',
                '.mp3': 'audio/mpeg',
                '.m4a': 'audio/m4a',
                '.mp4': 'audio/mp4',
                '.mpeg': 'audio/mpeg',
                '.mpga': 'audio/mpeg',
                '.webm': 'audio/webm',
                '.flac': 'audio/flac'
            }
            content_type = content_type_map.get(ext, 'audio/wav')
            
            # Add Content-Type to headers for the request
            request_headers = self.headers.copy()
            request_headers['Content-Type'] = content_type
            
            # Make API request to new HuggingFace router endpoint
            response = requests.post(
                self.api_url,
                headers=request_headers,
                data=audio_bytes,
                timeout=60  # HuggingFace can take time to load model on first request
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                print("Model is loading, waiting...")
                import time
                time.sleep(10)
                response = requests.post(
                    self.api_url,
                    headers=request_headers,
                    data=audio_bytes,
                    timeout=60
                )
            
            # Handle 410 (deprecated endpoint) error
            if response.status_code == 410:
                raise Exception(
                    "HuggingFace API endpoint has changed. The old endpoint is no longer supported. "
                    "Please update the code to use the new router endpoint."
                )
            
            if response.status_code != 200:
                raise Exception(f"API request failed: {response.status_code} - {response.text}")
            
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, dict):
                if 'text' in result:
                    transcribed_text = result['text'].strip()
                elif 'transcription' in result:
                    transcribed_text = result['transcription'].strip()
                else:
                    # Sometimes returns list
                    if isinstance(result.get('chunks'), list) and len(result['chunks']) > 0:
                        transcribed_text = result['chunks'][0].get('text', '').strip()
                    else:
                        transcribed_text = str(result).strip()
            else:
                transcribed_text = str(result).strip()
            
            print(f"Transcription complete: {transcribed_text[:100]}...")
            return transcribed_text
            
        except Exception as e:
            # If API fails and we have local model available, try local fallback
            if not self.use_local:
                error_msg = str(e)
                print(f"API transcription failed: {error_msg}")
                
                # Check if it's a network/SSL error or API error
                is_network_error = any(keyword in error_msg.lower() for keyword in [
                    'ssl', 'connection', 'timeout', 'max retries', 'eof', 
                    '410', '503', '500', '400'
                ])
                
                if is_network_error:
                    print("Network/API error detected. Attempting fallback to local Whisper model...")
                    try:
                        import whisper
                        if not hasattr(self, 'whisper_model'):
                            print("Loading local Whisper model 'base'...")
                            self.whisper_model = whisper.load_model("base")
                            self.use_local = True
                            self.local_model = "base"
                        return self._transcribe_local(audio_file_path)
                    except ImportError:
                        raise Exception(
                            f"API transcription failed due to network/SSL error: {error_msg}. "
                            f"Local Whisper not available. Install with: pip install openai-whisper"
                        )
                    except Exception as fallback_error:
                        raise Exception(
                            f"API transcription failed: {error_msg}. "
                            f"Local fallback also failed: {fallback_error}"
                        )
                else:
                    # For other errors, still try fallback but mention it's not a network issue
                    print("Attempting fallback to local Whisper model...")
                    try:
                        import whisper
                        if not hasattr(self, 'whisper_model'):
                            self.whisper_model = whisper.load_model("base")
                            self.use_local = True
                            self.local_model = "base"
                        return self._transcribe_local(audio_file_path)
                    except ImportError:
                        raise Exception(f"Error during transcription: {error_msg}. Local Whisper not available as fallback.")
                    except Exception as fallback_error:
                        raise Exception(f"Error during transcription: {error_msg}. Local fallback also failed: {fallback_error}")
            else:
                raise Exception(f"Error during transcription: {e}")
    
    def _transcribe_local(self, audio_file_path: str) -> str:
        """
        Transcribe audio file using local Whisper model.
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            print(f"Transcribing audio file using local Whisper model '{self.local_model}': {audio_file_path}")
            result = self.whisper_model.transcribe(audio_file_path)
            transcribed_text = result["text"].strip()
            print(f"Transcription complete: {transcribed_text[:100]}...")
            return transcribed_text
        except Exception as e:
            raise Exception(f"Error during local transcription: {e}")
    
    def transcribe_microphone(self, duration: int = 5) -> str:
        """
        Transcribe from microphone input using HuggingFace Inference API.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Transcribed text
        """
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            print("Error: sounddevice and soundfile libraries not installed.")
            print("Install with: pip install sounddevice soundfile")
            return ""
        
        try:
            sample_rate = 16000
            print(f"Recording for {duration} seconds...")
            print("Speak now...")
            
            # Record audio
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            print("Recording complete. Transcribing via HuggingFace API...")
            
            # Save to temporary file
            temp_file = "temp_recording.wav"
            sf.write(temp_file, audio_data, sample_rate)
            
            # Transcribe using API
            transcribed_text = self.transcribe_file(temp_file)
            
            # Clean up
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return transcribed_text
        except Exception as e:
            raise Exception(f"Error during microphone transcription: {e}")


# Alternative: Simple file-based transcription without microphone
def transcribe_audio_file(audio_path: str, api_key: Optional[str] = None) -> str:
    """
    Simple function to transcribe an audio file using HuggingFace API.
    
    Args:
        audio_path: Path to audio file
        api_key: HuggingFace API key (optional, uses HUGGINGFACE_API_KEY env var if not provided)
        
    Returns:
        Transcribed text
    """
    converter = VoiceToText(api_key=api_key)
    return converter.transcribe_file(audio_path)
