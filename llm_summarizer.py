"""
LLM-powered summarizer that generates reports from CSV data without hallucination.
Uses free/local LLM options with strict constraints to only use provided CSV data.
"""

import json
import os
from typing import List, Dict, Optional


class LLMSummarizer:
    """
    Generates summaries using LLM while strictly constraining to CSV data only.
    Supports multiple free LLM backends.
    """
    
    def __init__(self, backend: str = "groq", api_key: Optional[str] = None):
        """
        Initialize LLM summarizer.
        
        Args:
            backend: LLM backend to use - 'groq' (recommended), 'ollama' (local), 
                    'openai' (requires API key), 'huggingface' (free tier), 
                    or 'simple' (template-based, no LLM)
            api_key: API key for the selected backend (optional, uses env vars if not provided)
        """
        self.backend = backend
        self.api_key = api_key
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the selected LLM backend."""
        if self.backend == "groq":
            try:
                from groq import Groq
                api_key = self.api_key or os.getenv('GROQ_API_KEY')
                if not api_key:
                    raise ValueError(
                        "Groq API key required. Set GROQ_API_KEY environment variable "
                        "or pass api_key parameter."
                    )
                self.client = Groq(api_key=api_key)
                self.available = True
                print("Groq API client initialized successfully!")
            except ImportError:
                print("Warning: groq library not installed. Using simple template-based summaries.")
                print("Install with: pip install groq")
                self.backend = "simple"
                self.available = False
            except Exception as e:
                print(f"Warning: Groq setup failed: {e}")
                print("Using simple template-based summaries.")
                self.backend = "simple"
                self.available = False
        elif self.backend == "ollama":
            try:
                import ollama
                self.client = ollama
                self.available = True
            except ImportError:
                print("Warning: ollama not installed. Using simple template-based summaries.")
                print("Install with: pip install ollama")
                self.backend = "simple"
                self.available = False
        elif self.backend == "openai":
            try:
                from openai import OpenAI
                api_key = self.api_key or os.getenv('OPENAI_API_KEY')
                if not api_key:
                    raise ValueError("OpenAI API key required.")
                self.client = OpenAI(api_key=api_key)
                self.available = True
            except ImportError:
                print("Warning: openai not installed. Using simple template-based summaries.")
                self.backend = "simple"
                self.available = False
            except Exception as e:
                print(f"Warning: OpenAI setup failed: {e}")
                self.backend = "simple"
                self.available = False
        elif self.backend == "huggingface":
            try:
                from transformers import pipeline
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                self.available = True
            except Exception as e:
                print(f"Warning: HuggingFace setup failed: {e}")
                print("Using simple template-based summaries.")
                self.backend = "simple"
                self.available = False
        else:
            self.available = False
    
    def generate_summary(self, records: List[Dict], query: str, 
                        summary_type: str = "general") -> str:
        """
        Generate a summary from CSV records using LLM with strict constraints.
        
        Args:
            records: List of records from CSV
            query: Original user query
            summary_type: Type of summary ('general', 'monthly', 'date', 'type')
            
        Returns:
            Human-readable summary text
        """
        if not records:
            return "No records found in the CSV data."
        
        # Prepare context with strict instructions
        context = self._prepare_context(records, query)
        
        # Generate summary based on backend
        try:
            if self.backend == "groq" and self.available:
                return self._generate_with_groq(context, query, records)
            elif self.backend == "ollama" and self.available:
                return self._generate_with_ollama(context, query)
            elif self.backend == "openai" and self.available:
                return self._generate_with_openai(context, query)
            elif self.backend == "huggingface" and self.available:
                return self._generate_with_huggingface(context)
            else:
                return self._generate_template_summary(records, query, summary_type)
        except Exception as e:
            print(f"Error in LLM generation, falling back to template: {e}")
            return self._generate_template_summary(records, query, summary_type)
    
    def _prepare_context(self, records: List[Dict], query: str) -> str:
        """Prepare context string from CSV records."""
        context = "CSV DATA (USE ONLY THIS DATA, DO NOT HALLUCINATE):\n\n"
        
        for i, record in enumerate(records, 1):
            context += f"Record {i}:\n"
            context += f"  Type: {record['type']}\n"
            context += f"  Amount: ${float(record['amount']):.2f}\n"
            context += f"  Date: {record['date']}\n"
            context += f"  Details: {record['details']}\n\n"
        
        return context
    
    def _generate_with_groq(self, context: str, query: str, records: List[Dict]) -> str:
        """Generate summary using Groq API (fast, free Llama models)."""
        prompt = f"""You are a financial data assistant. Analyze the CSV data below and answer the user's query.

CRITICAL RULES:
1. ONLY use data from the CSV records provided below
2. DO NOT make up or hallucinate any data
3. If information is not in the CSV, say "No data available"
4. Be accurate and precise with numbers
5. Format the response in a clear, readable way

CSV DATA:
{context}

USER QUERY: {query}

Generate a clear, accurate summary based ONLY on the CSV data above:"""

        try:
            # Use Llama 3.1 8B or Llama 3.2 3B (fast and free)
            # Available models: llama-3.1-8b-instant, llama-3.1-70b-versatile, llama-3.2-3b-preview
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data assistant. Only use data provided in the CSV records. Never hallucinate."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama-3.1-8b-instant",  # Fast and free, can change to llama-3.1-70b-versatile for better quality
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=2000
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Error with Groq API: {e}")
            # Fallback to template
            return self._generate_template_summary(records, query, "general")
    
    def _generate_with_ollama(self, context: str, query: str) -> str:
        """Generate summary using Ollama (local, free)."""
        prompt = f"""You are a financial data assistant. Analyze the CSV data below and answer the user's query.

CRITICAL RULES:
1. ONLY use data from the CSV records provided below
2. DO NOT make up or hallucinate any data
3. If information is not in the CSV, say "No data available"
4. Be accurate and precise with numbers
5. Format the response in a clear, readable way

CSV DATA:
{context}

USER QUERY: {query}

Generate a clear, accurate summary based ONLY on the CSV data above:"""

        try:
            response = self.client.chat(
                model='llama3.2',  # or 'mistral', 'phi3', etc. - adjust based on installed models
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a financial data assistant. Only use data provided in the CSV records. Never hallucinate.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ]
            )
            # Handle different response formats
            if isinstance(response, dict):
                if 'message' in response:
                    return response['message'].get('content', '')
                elif 'content' in response:
                    return response['content']
            return str(response)
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return self._generate_template_summary(
                json.loads(context.replace("CSV DATA (USE ONLY THIS DATA, DO NOT HALLUCINATE):\n\n", "")),
                query, "general"
            )
    
    def _generate_with_openai(self, context: str, query: str) -> str:
        """Generate summary using OpenAI API (requires API key)."""
        prompt = f"""You are a financial data assistant. Analyze the CSV data below and answer the user's query.

CRITICAL RULES:
1. ONLY use data from the CSV records provided below
2. DO NOT make up or hallucinate any data
3. If information is not in the CSV, say "No data available"
4. Be accurate and precise with numbers

CSV DATA:
{context}

USER QUERY: {query}

Generate a clear, accurate summary based ONLY on the CSV data above:"""

        try:
            response = self.client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial data assistant. Only use data provided in CSV records. Never hallucinate."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3  # Lower temperature for more factual responses
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI: {e}")
            return "Error generating summary. Please check your API key."
    
    def _generate_with_huggingface(self, context: str) -> str:
        """Generate summary using HuggingFace transformers."""
        try:
            # Combine context for summarization
            full_text = context + "\n\nGenerate a summary of the financial data above."
            summary = self.summarizer(full_text, max_length=200, min_length=50, do_sample=False)
            return summary[0]['summary_text']
        except Exception as e:
            print(f"Error with HuggingFace: {e}")
            return "Error generating summary."
    
    def _generate_template_summary(self, records: List[Dict], query: str, 
                                  summary_type: str) -> str:
        """Generate summary using template-based approach (no LLM, always works)."""
        if not records:
            return "No records found."
        
        # Calculate statistics
        profits = [r for r in records if r['type'].lower() == 'profit']
        losses = [r for r in records if r['type'].lower() == 'loss']
        
        total_profit = sum(p['amount'] for p in profits)
        total_loss = sum(l['amount'] for l in losses)
        net = total_profit - total_loss
        
        summary = f"Financial Summary\n"
        summary += "=" * 50 + "\n\n"
        summary += f"Total Profit: ${total_profit:,.2f} ({len(profits)} transactions)\n"
        summary += f"Total Loss: ${total_loss:,.2f} ({len(losses)} transactions)\n"
        summary += f"Net: ${net:,.2f}\n\n"
        
        if summary_type == "monthly" or "month" in query.lower():
            summary += "Monthly Breakdown:\n"
            summary += "-" * 50 + "\n"
            # Group by month
            from collections import defaultdict
            monthly = defaultdict(lambda: {'profit': 0, 'loss': 0})
            for r in records:
                month = r['date'][:7]  # YYYY-MM
                if r['type'].lower() == 'profit':
                    monthly[month]['profit'] += float(r['amount'])
                else:
                    monthly[month]['loss'] += float(r['amount'])
            
            for month in sorted(monthly.keys()):
                summary += f"{month}: Profit ${monthly[month]['profit']:,.2f}, Loss ${monthly[month]['loss']:,.2f}\n"
        
        summary += "\nRecent Transactions:\n"
        summary += "-" * 50 + "\n"
        for r in sorted(records, key=lambda x: x['date'], reverse=True)[:10]:
            summary += f"{r['date']}: {r['type'].capitalize()} ${float(r['amount']):,.2f} - {r['details']}\n"
        
        return summary

