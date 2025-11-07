"""
FastAPI application for Voice-to-Data Assistant.
Provides REST API endpoints for testing and integration.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import json
import traceback
import tempfile
from typing import Optional

# Load environment variables
load_dotenv()

# Import our assistant
from voice_assistant import VoiceToDataAssistant

# Initialize FastAPI app
app = FastAPI(
    title="Voice-to-Data Assistant API",
    description="API for processing voice transcriptions and generating financial reports",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize assistant
assistant = None

def get_assistant():
    """Lazy initialization of assistant."""
    global assistant
    if assistant is None:
        assistant = VoiceToDataAssistant(
            use_llm=True,
            llm_backend="groq",
            groq_api_key=os.getenv('GROQ_API_KEY'),
            huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY')
        )
    return assistant


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Voice-to-Data Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Transcribe audio file to text and extract data",
            "POST /transcribe-text": "Process text transcription directly",
            "POST /query": "Query financial data and generate reports",
            "GET /records": "Get all financial records",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "voice-to-data-assistant"}


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file and extract financial data.
    
    - **file**: Audio file (mp3, wav, m4a, etc.)
    - Returns: Extracted financial data in JSON format
    """
    temp_path = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Save uploaded file temporarily with proper extension
        file_ext = os.path.splitext(file.filename)[1] or '.wav'
        temp_fd, temp_path = tempfile.mkstemp(suffix=file_ext, prefix='transcribe_')
        os.close(temp_fd)
        
        with open(temp_path, "wb") as f:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Empty file uploaded")
            f.write(content)
        
        # Process voice file
        assistant = get_assistant()
        result = assistant.process_voice_file(temp_path)
        
        return JSONResponse(content={
            "success": True,
            "transcribed_text": result.get('transcribed_text', ''),
            "extracted_data": result['json'],
            "message": result['message']
        })
        
    except HTTPException:
        raise
    except Exception as e:
        # Log full traceback to console for debugging
        error_traceback = traceback.format_exc()
        print(f"Error in /transcribe endpoint:\n{error_traceback}")
        # Return cleaner error message to client
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing audio file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Warning: Failed to remove temp file {temp_path}: {e}")


@app.post("/transcribe-text")
async def transcribe_text(text: str = Form(...)):
    """
    Process text transcription directly and extract financial data.
    
    - **text**: Transcribed text
    - Returns: Extracted financial data in JSON format
    """
    try:
        assistant = get_assistant()
        result = assistant.process_transcription(text)
        
        return JSONResponse(content={
            "success": True,
            "extracted_data": result['json'],
            "message": result['message']
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_data(query: str = Form(...)):
    """
    Query financial data and generate reports.
    
    - **query**: Natural language query (e.g., "Show me all profit details for March")
    - Returns: Human-readable summary and optional JSON report
    """
    try:
        assistant = get_assistant()
        result = assistant.answer_query(query)
        
        return JSONResponse(content={
            "success": True,
            "text": result['text'],
            "json_report": result.get('json')
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records")
async def get_records():
    """Get all financial records from CSV."""
    try:
        assistant = get_assistant()
        records = assistant.get_all_records()
        
        return JSONResponse(content={
            "success": True,
            "count": len(records),
            "records": records
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/records/{record_type}")
async def get_records_by_type(record_type: str):
    """
    Get records filtered by type.
    
    - **record_type**: 'profit' or 'loss'
    """
    try:
        assistant = get_assistant()
        all_records = assistant.get_all_records()
        filtered = [r for r in all_records if r['type'].lower() == record_type.lower()]
        
        return JSONResponse(content={
            "success": True,
            "type": record_type,
            "count": len(filtered),
            "records": filtered
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

