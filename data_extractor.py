"""
Data extraction module for parsing transcribed voice text and extracting
structured financial information (profit/loss transactions).
"""

import re
import json
from datetime import datetime
from typing import Dict, Optional


class DataExtractor:
    """Extracts structured data from transcribed voice text."""
    
    def __init__(self):
        # Patterns for detecting profit/loss keywords
        self.profit_patterns = [
            r'\b(profit|gained|earned|received|income|revenue|made|won)\b',
            r'\b(in|plus|positive)\b'
        ]
        self.loss_patterns = [
            r'\b(loss|lost|spent|expense|paid|cost|negative|minus)\b',
            r'\b(out|down)\b'
        ]
        
        # Pattern for extracting amounts
        self.amount_pattern = r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\b|\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD|rs|rupees?)'
        
        # Pattern for extracting dates
        self.date_patterns = [
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b',
            r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{2,4})\b',
            r'\b(today|yesterday|tomorrow)\b'
        ]
        
        self.month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
    
    def extract(self, text: str) -> Dict:
        """
        Extract structured data from transcribed text.
        
        Args:
            text: Transcribed voice text
            
        Returns:
            Dictionary with type, amount, date, and details
        """
        text_lower = text.lower().strip()
        
        # Extract type (profit or loss)
        transaction_type = self._extract_type(text_lower)
        
        # Extract amount (handles percentage-based calculations)
        amount = self._extract_amount(text)
        
        # Extract date
        date = self._extract_date(text_lower)
        
        # Extract details (everything else, cleaned up)
        details = self._extract_details(text, transaction_type, amount)
        
        return {
            "type": transaction_type,
            "amount": amount,
            "date": date,
            "details": details
        }
    
    def _extract_type(self, text: str) -> str:
        """Determine if transaction is profit or loss."""
        # Check for profit indicators
        for pattern in self.profit_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "profit"
        
        # Check for loss indicators
        for pattern in self.loss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return "loss"
        
        # Default to loss if unclear (conservative approach)
        # You might want to return "unknown" or ask for clarification
        return "loss"
    
    def _extract_amount(self, text: str) -> float:
        """Extract monetary amount from text, handling percentage-based profits and number words."""
        text_lower = text.lower()

        # Normalize common number words into digits to handle phrases like
        # "five thousand" and "five percent"
        normalized = self._normalize_number_words(text_lower)
        
        # Check for percentage-based profit/loss first
        # Pattern: "profit is X%" or "profit of X%" or "X% profit"
        percentage_pattern = r'(?:profit|loss).*?(?:is|of|at)\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*(?:profit|loss)'
        percentage_match = re.search(percentage_pattern, normalized, re.IGNORECASE)
        
        if percentage_match:
            # Extract percentage value
            percentage = float(percentage_match.group(1) if percentage_match.group(1) else percentage_match.group(2))
            
            # Find the base/investment amount to calculate from
            # Look for "invested X", "investment of X", "base X", etc.
            investment_patterns = [
                r'invested\s+(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'investment\s+(?:of|is)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'base\s+(?:of|is)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'capital\s+(?:of|is)\s+(\d+(?:,\d{3})*(?:\.\d{2})?)',
            ]
            
            base_amount = None
            for pattern in investment_patterns:
                inv_match = re.search(pattern, normalized, re.IGNORECASE)
                if inv_match:
                    base_str = inv_match.group(1).replace(',', '')
                    try:
                        base_amount = float(base_str)
                        break
                    except ValueError:
                        continue
            
            # If no explicit investment mentioned, try to find any amount in the text
            if base_amount is None:
                matches = re.findall(self.amount_pattern, normalized, re.IGNORECASE)
                if matches:
                    amount_str = matches[0][0] if matches[0][0] else matches[0][1]
                    amount_str = amount_str.replace(',', '')
                    try:
                        base_amount = float(amount_str)
                    except ValueError:
                        pass
            
            # Calculate percentage-based amount
            if base_amount is not None:
                calculated_amount = base_amount * (percentage / 100)
                return round(calculated_amount, 2)
        
        # If no percentage found, extract regular amount
        matches = re.findall(self.amount_pattern, normalized, re.IGNORECASE)
        
        if matches:
            # Take the first match
            amount_str = matches[0][0] if matches[0][0] else matches[0][1]
            # Remove commas and convert to float
            amount_str = amount_str.replace(',', '')
            try:
                return float(amount_str)
            except ValueError:
                pass
        
        return 0.0

    def _normalize_number_words(self, text: str) -> str:
        """Convert common number words in text to digits.
        Supports simple phrases like 'five', 'twenty five', 'one hundred',
        'five thousand', 'two lakh', 'one crore'.
        """
        # Basic dictionaries
        units = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
        }
        tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        scales = {
            'hundred': 100,
            'thousand': 1000,
            'lakh': 100000,
            'crore': 10000000
        }

        # Replace "percent" number words (e.g., 'five percent') first to aid regex
        def convert_number_phrase(tokens):
            current = 0
            total = 0
            i = 0
            while i < len(tokens):
                word = tokens[i]
                if word in units:
                    current += units[word]
                elif word in tens:
                    current += tens[word]
                elif word == 'and':
                    pass
                elif word in scales:
                    scale = scales[word]
                    if current == 0:
                        current = 1
                    current *= scale
                    total += current
                    current = 0
                else:
                    # stop at first non-number word
                    break
                i += 1
            return total + current, i

        def replace_sequences(text_in: str) -> str:
            tokens = re.findall(r"[a-zA-Z]+|\d+|%|\S", text_in)
            i = 0
            out = []
            while i < len(tokens):
                word = tokens[i].lower()
                if word in units or word in tens or word in scales or word == 'and':
                    number, consumed = convert_number_phrase([t.lower() for t in tokens[i:]])
                    if consumed > 0:
                        out.append(str(number))
                        i += consumed
                        continue
                out.append(tokens[i])
                i += 1
            return ''.join(x if re.fullmatch(r"\W", x or '') else (x if out and out[-1] == '' else x if x in ['%', ','] else ('' + x if not out else (' ' + x if re.fullmatch(r"[a-zA-Z]+", x) and re.fullmatch(r"[a-zA-Z]+|\d+", out[-1]) else x))) for x in out) if False else ' '.join(out).replace(' %', '%')

        return replace_sequences(text)
    
    def _extract_date(self, text: str) -> str:
        """Extract date from text, or return current date if not found."""
        text_lower = text.lower()
        
        # Check for relative dates first
        today = datetime.now()
        if re.search(r'\btoday\b', text_lower):
            return today.strftime("%Y-%m-%d")
        elif re.search(r'\byesterday\b', text_lower):
            yesterday = today.replace(day=today.day - 1)
            return yesterday.strftime("%Y-%m-%d")
        elif re.search(r'\btomorrow\b', text_lower):
            tomorrow = today.replace(day=today.day + 1)
            return tomorrow.strftime("%Y-%m-%d")
        
        # Check for numeric dates (MM/DD/YYYY or DD/MM/YYYY)
        match = re.search(r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b', text)
        if match:
            month, day, year = match.groups()
            year = int(year)
            if year < 100:
                year += 2000 if year < 50 else 1900
            
            try:
                # Try MM/DD/YYYY first
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                # Try DD/MM/YYYY
                try:
                    date_obj = datetime(int(year), int(day), int(month))
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        
        # Check for month name dates
        for pattern in self.date_patterns[1:3]:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[0].lower() in self.month_names:
                        month = self.month_names[groups[0].lower()]
                        day = int(groups[1])
                        year = int(groups[2])
                    elif groups[1].lower() in self.month_names:
                        month = self.month_names[groups[1].lower()]
                        day = int(groups[0])
                        year = int(groups[2])
                    else:
                        continue
                    
                    if year < 100:
                        year += 2000 if year < 50 else 1900
                    
                    try:
                        date_obj = datetime(year, month, day)
                        return date_obj.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
        
        # Default to current date
        return datetime.now().strftime("%Y-%m-%d")
    
    def _extract_details(self, text: str, transaction_type: str, amount: float) -> str:
        """Extract additional details from the text."""
        text_lower = text.lower()
        details = text
        
        # Check if this is a percentage-based transaction
        percentage_match = re.search(r'(?:profit|loss).*?(?:is|of|at)\s*(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*%\s*(?:profit|loss)', text_lower, re.IGNORECASE)
        investment_match = re.search(r'invested\s+(\d+(?:,\d{3})*(?:\.\d{2})?)', text_lower, re.IGNORECASE)
        
        # If percentage-based, create a more informative detail
        if percentage_match and investment_match:
            percentage = percentage_match.group(1) if percentage_match.group(1) else percentage_match.group(2)
            investment = investment_match.group(1).replace(',', '')
            details = f"Investment of {investment} with {percentage}% {transaction_type}"
        else:
            # Remove amount mentions
            details = re.sub(self.amount_pattern, '', details, flags=re.IGNORECASE)
            
            # Remove percentage mentions
            details = re.sub(r'\d+(?:\.\d+)?\s*%', '', details, flags=re.IGNORECASE)
            
            # Remove date mentions
            for pattern in self.date_patterns:
                details = re.sub(pattern, '', details, flags=re.IGNORECASE)
            
            # Remove common transaction type words but keep context
            details = re.sub(r'\b(profit|loss|gained|earned|received|spent|paid|lost)\s+(?:is|of|at)\b', '', details, flags=re.IGNORECASE)
            details = re.sub(r'\b(today|yesterday|tomorrow)\b', '', details, flags=re.IGNORECASE)
            details = re.sub(r'\b(and|is|nothing|nothing)\b', '', details, flags=re.IGNORECASE)
            
            # Clean up whitespace
            details = ' '.join(details.split())
            details = details.strip()
        
        # If details are empty or too short, provide a default
        if not details or len(details) < 3:
            details = f"{transaction_type.capitalize()} transaction"
        
        return details

