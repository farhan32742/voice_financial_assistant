# Voice-to-Data Assistant

A complete Python-based system that:
1. **Converts voice to text** using HuggingFace Inference API (no local models)
2. **Extracts structured financial information** from transcribed text
3. **Saves data to CSV** for persistent storage
4. **Generates intelligent summaries** using Groq API with Llama models (with strict CSV-only constraints to prevent hallucination)
5. **FastAPI REST API** for easy testing and integration

## Architecture

```
Voice Input → HuggingFace API → Text → Data Extraction → CSV Storage
                                                                    ↓
User Query → Query Analyzer → Groq API (Llama) → CSV-only Summary → Report
```

**All processing is API-based - no local model loading required!**

## Features

1. **Voice-to-Text**: 
   - HuggingFace Inference API (no local model loading)
   - Uses OpenAI Whisper Large V3 model via HuggingFace
   - Supports microphone recording
   - Supports audio file processing (mp3, wav, m4a, mp4, mpeg, mpga, webm, flac)
   - Requires HuggingFace API key (free tier available)

2. **Data Extraction**: Extracts structured information from transcribed voice text:
   - Transaction type (profit or loss)
   - Amount of money
   - Date (extracted or system date)
   - Additional details

3. **Data Storage**: Automatically saves extracted data to CSV format

4. **LLM-Powered Summaries**: 
   - Uses Groq API with Llama models (fast, free, no local loading)
   - **Strict CSV-only constraints** - never hallucinates data
   - Falls back to template-based summaries if API unavailable
   - Supports multiple LLM backends (Groq, Ollama, OpenAI, HuggingFace)

5. **Query Analysis**: Answers questions about stored financial data:
   - Monthly reports
   - Date-specific queries
   - Profit/loss summaries
   - Custom reports

## Installation

### Basic Installation (Text-only mode)

For basic functionality without voice or LLM:
- Python 3.7+ (uses only standard library)

### Full Installation (Voice + LLM)

Install dependencies:

```bash
pip install -r requirements.txt
```

This installs:
- `python-dotenv` - For loading .env file
- `requests` - For HuggingFace Inference API (voice-to-text)
- `groq` - For Groq API (Llama models for summaries)
- `fastapi` & `uvicorn` - For REST API server
- `sounddevice` & `soundfile` - For microphone recording (optional)

### API Keys Setup

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file and add your API keys:**
   ```bash
   # HuggingFace API Key (for voice-to-text)
   HUGGINGFACE_API_KEY=your_huggingface_api_key_here
   
   # Groq API Key (for LLM summaries)
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Get your API keys:**
   - **HuggingFace**: https://huggingface.co/settings/tokens (free tier available)
   - **Groq**: https://console.groq.com/ (free tier available)

**Note**: The system works without LLM (uses template-based summaries) and without voice input (manual text entry).

## Usage

### FastAPI REST API (Recommended for Testing)

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at:
- **API**: http://127.0.0.1:8000
- **Interactive Docs**: http://127.0.0.1:8000/docs
- **Alternative Docs**: http://127.0.0.1:8000/redoc

#### API Endpoints:

- `POST /transcribe` - Upload audio file and extract financial data
- `POST /transcribe-text` - Process text transcription directly
- `POST /query` - Query financial data and generate reports
- `GET /records` - Get all financial records
- `GET /records/{type}` - Get records by type (profit/loss)
- `GET /health` - Health check

#### Example API Usage:

```bash
# Transcribe audio file
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@recording.mp3"

# Process text
curl -X POST "http://127.0.0.1:8000/transcribe-text" \
  -F "text=I made a profit of $500 on March 15th"

# Query data
curl -X POST "http://127.0.0.1:8000/query" \
  -F "query=Show me all profit details for March"
```

### Command Line Interface

Run the main script:

```bash
python main.py
```

You'll be asked if you want to use LLM for enhanced summaries (optional).

### Processing Voice Input

**Option 1: Record from Microphone**
```
You: voice
Recording duration in seconds (default=5): 5
[Speak your transaction]
```

**Option 2: Process Audio File**
```
You: file:my_recording.mp3
```

**Option 3: Manual Text Entry**
Enter transcribed voice text directly. The assistant will extract and save the data:

```
You: I made a profit of $500 on March 15th selling old items
```

Output:
```json
{
  "type": "profit",
  "amount": 500.0,
  "date": "2024-03-15",
  "details": "selling old items"
}
```

### Asking Questions

Start your query with `?` or `query:`:

```
You: ? Show me all profit details for March
You: ? How much loss did I have on 12 August?
You: ? Generate a monthly profit/loss report
```

### Programmatic Usage

```python
from voice_assistant import VoiceToDataAssistant

# Initialize assistant (with LLM support)
assistant = VoiceToDataAssistant(
    use_llm=True, 
    llm_backend="groq",
    groq_api_key="your-groq-api-key",  # or set GROQ_API_KEY env var
    huggingface_api_key="your-huggingface-api-key"  # or set HUGGINGFACE_API_KEY env var
)

# Process voice file
result = assistant.process_voice_file("recording.mp3")
print(result['json'])

# Process microphone recording
result = assistant.process_voice_microphone(duration=5)
print(result['json'])

# Process transcription directly
transcribed_text = "I lost $200 yesterday on groceries"
result = assistant.process_transcription(transcribed_text)
print(result['json'])

# Answer queries (uses LLM if enabled)
query = "Show me all profit details for March"
response = assistant.answer_query(query)
print(response['text'])
if response.get('json'):
    print(response['json'])
```

## Data Format

All extracted data follows this JSON structure:

```json
{
  "type": "profit" | "loss",
  "amount": <number>,
  "date": "YYYY-MM-DD",
  "details": "<text description>"
}
```

## CSV Storage

Data is automatically saved to `financial_records.csv` with the following columns:
- `type`: profit or loss
- `amount`: numeric value
- `date`: YYYY-MM-DD format
- `details`: text description

## Supported Query Types

1. **Monthly Reports**: "Show me all profit details for March"
2. **Date Queries**: "How much loss did I have on 12 August?"
3. **Type Queries**: "Show all profits for January 2024"
4. **Summary Reports**: "Generate a monthly profit/loss report"

## Examples

### Example 1: Recording a Profit
```
Input: "I earned $1500 today from freelance work"
Output:
{
  "type": "profit",
  "amount": 1500.0,
  "date": "2024-12-19",
  "details": "from freelance work"
}
```

### Example 2: Recording a Loss
```
Input: "Spent $75 on 12/15/2024 for office supplies"
Output:
{
  "type": "loss",
  "amount": 75.0,
  "date": "2024-12-15",
  "details": "for office supplies"
}
```

### Example 3: Querying Data
```
Input: "? Show me all profit details for March"
Output:
All Profit Details for March 2024
==================================================

Total Profit: $2,500.00 (5 transactions)

Details:
--------------------------------------------------
  $500.00 on 2024-03-15 - selling old items
  $1,000.00 on 2024-03-20 - freelance project
  ...
```

## Anti-Hallucination Features

The system is designed to **never hallucinate data**:

1. **Strict CSV-Only Constraints**: LLM prompts explicitly instruct to only use CSV data
2. **Template Fallback**: If LLM fails or is unavailable, uses template-based summaries
3. **Data Validation**: All numbers and dates come directly from CSV records
4. **Explicit Instructions**: System prompts emphasize "DO NOT HALLUCINATE" and "USE ONLY CSV DATA"
5. **No External Data**: LLM never accesses external knowledge bases - only the provided CSV records

## Important Notes

- **No Data Hallucination**: The assistant only uses data from the CSV file you provide
- **Date Handling**: If no date is mentioned, the system date is used
- **Amount Detection**: Supports various formats ($500, 500 dollars, 500.00, etc.)
- **Type Detection**: Uses keyword matching to determine profit vs loss
- **LLM Optional**: System works perfectly without LLM (uses template summaries)
- **Voice Optional**: System works with manual text entry if voice input unavailable

## File Structure

```
voice_agent/
├── main.py              # CLI entry point
├── voice_assistant.py   # Main assistant class
├── data_extractor.py    # Text extraction logic
├── csv_manager.py       # CSV file operations
├── query_analyzer.py    # Query processing and reports
├── requirements.txt     # Dependencies (none required)
└── README.md           # This file
```

## License

This project is provided as-is for personal use.

