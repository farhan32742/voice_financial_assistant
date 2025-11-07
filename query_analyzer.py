"""
Query analyzer for processing questions and generating reports from CSV data.
"""

import re
from typing import List, Dict, Optional
from datetime import datetime
from csv_manager import CSVManager
from llm_summarizer import LLMSummarizer


class QueryAnalyzer:
    """Analyzes queries and generates reports from CSV data."""
    
    def __init__(self, csv_manager: CSVManager, use_llm: bool = True, 
                 llm_backend: str = "groq", llm_api_key: Optional[str] = None):
        """
        Initialize query analyzer.
        
        Args:
            csv_manager: CSVManager instance for data access
            use_llm: Whether to use LLM for enhanced summaries
            llm_backend: LLM backend to use ('groq', 'ollama', 'openai', 'huggingface', or 'simple')
            llm_api_key: API key for LLM backend (optional, uses env vars if not provided)
        """
        self.csv_manager = csv_manager
        self.use_llm = use_llm
        if use_llm:
            self.llm_summarizer = LLMSummarizer(backend=llm_backend, api_key=llm_api_key)
        else:
            self.llm_summarizer = None
        
        # Patterns for query types
        self.month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        self.date_pattern = r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|(\d{1,2})(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{2,4})\b'
        self.relative_date_pattern = r'\b(today|yesterday|tomorrow)\b'
        self.profit_pattern = r'\b(profit|profits|gained|earned|income|revenue|made|earn)\b'
        self.loss_pattern = r'\b(loss|losses|lost|spent|expense|expenses|spend)\b'
        self.amount_pattern = r'\b(how much|what|total|amount|sum|did|have)\b'
        
        self.month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze a query and return appropriate response.
        
        Args:
            query: User's question/query
            
        Returns:
            Dictionary with 'text' (human-readable) and optional 'json' (structured data)
        """
        query_lower = query.lower()
        
        # Determine query type and extract parameters
        query_type, params = self._parse_query(query_lower)
        
        if query_type == 'monthly_report':
            return self._generate_monthly_report(params.get('month'), params.get('year'))
        elif query_type == 'date_query':
            return self._generate_date_report(params.get('date'), params.get('type'))
        elif query_type == 'type_query':
            return self._generate_type_report(params.get('type'), params.get('month'), params.get('year'))
        elif query_type == 'summary':
            return self._generate_summary(params.get('type'))
        else:
            return {
                'text': "I couldn't understand your query. Please try asking about:\n"
                       "- Today's transactions (e.g., 'How much did I today?' or 'What did I spend today?')\n"
                       "- Specific dates (e.g., 'Show me all profit details for March')\n"
                       "- Amounts on dates (e.g., 'How much loss did I have on 12 August?')\n"
                       "- Monthly reports (e.g., 'Generate a monthly profit/loss report')",
                'json': None
            }
    
    def _parse_query(self, query: str) -> tuple:
        """Parse query to determine type and extract parameters."""
        params = {}
        query_type = 'unknown'
        
        # Check for relative dates (today, yesterday, tomorrow) first
        relative_date_match = re.search(self.relative_date_pattern, query, re.IGNORECASE)
        if relative_date_match:
            relative_date = relative_date_match.group(1).lower()
            today = datetime.now()
            if relative_date == 'today':
                date_str = today.strftime("%Y-%m-%d")
            elif relative_date == 'yesterday':
                from datetime import timedelta
                date_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            elif relative_date == 'tomorrow':
                from datetime import timedelta
                date_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                date_str = today.strftime("%Y-%m-%d")
            
            params['date'] = date_str
            # Check if asking for specific type
            if re.search(self.profit_pattern, query):
                params['type'] = 'profit'
            elif re.search(self.loss_pattern, query):
                params['type'] = 'loss'
            query_type = 'date_query'
        
        # Check for month mention
        month_match = re.search(self.month_pattern, query, re.IGNORECASE)
        if month_match and not relative_date_match:  # Don't override relative date
            month_name = month_match.group(1).lower()
            params['month'] = self.month_names[month_name]
            # Try to extract year
            year_match = re.search(r'\b(20\d{2})\b', query)
            if year_match:
                params['year'] = int(year_match.group(1))
            else:
                params['year'] = datetime.now().year
            
            # Check if asking for specific type
            if re.search(self.profit_pattern, query):
                params['type'] = 'profit'
                query_type = 'type_query'
            elif re.search(self.loss_pattern, query):
                params['type'] = 'loss'
                query_type = 'type_query'
            else:
                query_type = 'monthly_report'
        
        # Check for specific date (numeric format)
        date_match = re.search(self.date_pattern, query, re.IGNORECASE)
        if date_match and not relative_date_match:  # Don't override relative date
            date_str = self._extract_date_from_match(date_match, query)
            if date_str:
                params['date'] = date_str
                # Check if asking for specific type
                if re.search(self.profit_pattern, query):
                    params['type'] = 'profit'
                elif re.search(self.loss_pattern, query):
                    params['type'] = 'loss'
                query_type = 'date_query'
        
        # Check for amount queries (how much, what, total) - often with "today" or dates
        if re.search(self.amount_pattern, query) and not query_type == 'unknown':
            # Already handled by date_query or other types
            pass
        elif re.search(self.amount_pattern, query) and query_type == 'unknown':
            # Generic amount query - check if it's asking about today
            if re.search(r'\btoday\b', query, re.IGNORECASE):
                today = datetime.now()
                params['date'] = today.strftime("%Y-%m-%d")
                query_type = 'date_query'
            else:
                # Generic summary
                query_type = 'summary'
                params['type'] = None
        
        # Check for general summary/report request
        if re.search(r'\b(report|summary|overview|total)\b', query) and query_type == 'unknown':
            if re.search(self.profit_pattern, query):
                params['type'] = 'profit'
                query_type = 'summary'
            elif re.search(self.loss_pattern, query):
                params['type'] = 'loss'
                query_type = 'summary'
            elif not month_match and not date_match and not relative_date_match:
                query_type = 'summary'
                params['type'] = None  # All types
        
        return query_type, params
    
    def _extract_date_from_match(self, match, query: str) -> Optional[str]:
        """Extract date string from regex match."""
        groups = match.groups()
        
        # Try numeric date format (MM/DD/YYYY or DD/MM/YYYY)
        if groups[0] and groups[1] and groups[2]:
            month, day, year = groups[0], groups[1], groups[2]
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
        
        # Try month name format
        if groups[4] and groups[5]:
            month_name = groups[4].lower()
            day = int(groups[3])
            year = int(groups[5])
            if year < 100:
                year += 2000 if year < 50 else 1900
            
            if month_name in self.month_names:
                try:
                    date_obj = datetime(year, self.month_names[month_name], day)
                    return date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    pass
        
        return None
    
    def _generate_monthly_report(self, month: Optional[int] = None, year: Optional[int] = None) -> Dict:
        """Generate monthly profit/loss report."""
        month_name = None
        if month and year:
            records = self.csv_manager.get_records_by_month(year, month)
            month_name = [k for k, v in self.month_names.items() if v == month][0].capitalize()
            title = f"Monthly Report for {month_name} {year}"
        else:
            records = self.csv_manager.read_all_records()
            title = "Overall Monthly Report"
        
        if not records:
            return {
                'text': f"No records found for the specified period.",
                'json': None
            }
        
        # Separate by type
        profits = [r for r in records if r['type'].lower() == 'profit']
        losses = [r for r in records if r['type'].lower() == 'loss']
        
        total_profit = sum(p['amount'] for p in profits)
        total_loss = sum(l['amount'] for l in losses)
        net = total_profit - total_loss
        
        # Use LLM for enhanced summary if available, otherwise use template
        if self.llm_summarizer and self.use_llm:
            query = f"Generate a monthly report for {month_name} {year}" if month_name and year else "Generate a monthly report"
            text = self.llm_summarizer.generate_summary(records, query, "monthly")
        else:
            # Build text report (template-based)
            text = f"{title}\n"
            text += "=" * 50 + "\n\n"
            text += f"Total Profit: ${total_profit:,.2f} ({len(profits)} transactions)\n"
            text += f"Total Loss: ${total_loss:,.2f} ({len(losses)} transactions)\n"
            text += f"Net: ${net:,.2f}\n\n"
            
            if profits:
                text += "Profit Details:\n"
                text += "-" * 50 + "\n"
                for p in profits:
                    text += f"  ${p['amount']:,.2f} on {p['date']} - {p['details']}\n"
                text += "\n"
            
            if losses:
                text += "Loss Details:\n"
                text += "-" * 50 + "\n"
                for l in losses:
                    text += f"  ${l['amount']:,.2f} on {l['date']} - {l['details']}\n"
        
        # Build JSON report
        period_str = f"{month_name} {year}" if month_name and year else "All time"
        json_report = {
            'period': period_str,
            'summary': {
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net': net,
                'profit_count': len(profits),
                'loss_count': len(losses)
            },
            'profits': profits,
            'losses': losses
        }
        
        return {
            'text': text,
            'json': json_report
        }
    
    def _generate_date_report(self, date: str, transaction_type: Optional[str] = None) -> Dict:
        """Generate report for a specific date."""
        records = self.csv_manager.get_records_by_date(date)
        
        if transaction_type:
            records = [r for r in records if r['type'].lower() == transaction_type.lower()]
        
        if not records:
            type_str = f" {transaction_type}" if transaction_type else ""
            return {
                'text': f"No{type_str} records found for {date}.",
                'json': None
            }
        
        # Calculate totals
        total = sum(r['amount'] for r in records)
        profits = [r for r in records if r['type'].lower() == 'profit']
        losses = [r for r in records if r['type'].lower() == 'loss']
        total_profit = sum(p['amount'] for p in profits)
        total_loss = sum(l['amount'] for l in losses)
        
        # Build text report
        type_str = transaction_type.capitalize() if transaction_type else "All"
        text = f"{type_str} Records for {date}\n"
        text += "=" * 50 + "\n\n"
        
        if transaction_type:
            text += f"Total {transaction_type.capitalize()}: ${total:,.2f}\n\n"
            text += "Details:\n"
            text += "-" * 50 + "\n"
            for r in records:
                text += f"  ${r['amount']:,.2f} - {r['details']}\n"
        else:
            text += f"Total Profit: ${total_profit:,.2f} ({len(profits)} transactions)\n"
            text += f"Total Loss: ${total_loss:,.2f} ({len(losses)} transactions)\n"
            text += f"Net: ${total_profit - total_loss:,.2f}\n\n"
            text += "All Transactions:\n"
            text += "-" * 50 + "\n"
            for r in records:
                text += f"  {r['type'].capitalize()}: ${r['amount']:,.2f} - {r['details']}\n"
        
        # Build JSON report
        json_report = {
            'date': date,
            'type_filter': transaction_type,
            'summary': {
                'total': total if transaction_type else None,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net': total_profit - total_loss if not transaction_type else None,
                'count': len(records)
            },
            'records': records
        }
        
        return {
            'text': text,
            'json': json_report
        }
    
    def _generate_type_report(self, transaction_type: str, month: Optional[int] = None, 
                             year: Optional[int] = None) -> Dict:
        """Generate report for a specific type (profit or loss) in a month."""
        month_name = None
        if month and year:
            records = self.csv_manager.get_records_by_month(year, month)
            records = [r for r in records if r['type'].lower() == transaction_type.lower()]
            month_name = [k for k, v in self.month_names.items() if v == month][0].capitalize()
            title = f"All {transaction_type.capitalize()} Details for {month_name} {year}"
        else:
            records = self.csv_manager.get_records_by_type(transaction_type)
            title = f"All {transaction_type.capitalize()} Details"
        
        if not records:
            return {
                'text': f"No {transaction_type} records found for the specified period.",
                'json': None
            }
        
        total = sum(r['amount'] for r in records)
        
        # Build text report
        text = f"{title}\n"
        text += "=" * 50 + "\n\n"
        text += f"Total {transaction_type.capitalize()}: ${total:,.2f} ({len(records)} transactions)\n\n"
        text += "Details:\n"
        text += "-" * 50 + "\n"
        for r in records:
            text += f"  ${r['amount']:,.2f} on {r['date']} - {r['details']}\n"
        
        # Build JSON report
        period_str = f"{month_name} {year}" if month_name and year else "All time"
        json_report = {
            'type': transaction_type,
            'period': period_str,
            'summary': {
                'total': total,
                'count': len(records)
            },
            'records': records
        }
        
        return {
            'text': text,
            'json': json_report
        }
    
    def _generate_summary(self, transaction_type: Optional[str] = None) -> Dict:
        """Generate overall summary report."""
        if transaction_type:
            records = self.csv_manager.get_records_by_type(transaction_type)
        else:
            records = self.csv_manager.read_all_records()
        
        if not records:
            return {
                'text': "No records found.",
                'json': None
            }
        
        profits = [r for r in records if r['type'].lower() == 'profit']
        losses = [r for r in records if r['type'].lower() == 'loss']
        
        total_profit = sum(p['amount'] for p in profits)
        total_loss = sum(l['amount'] for l in losses)
        net = total_profit - total_loss
        
        # Build text report
        if transaction_type:
            total = total_profit if transaction_type == 'profit' else total_loss
            text = f"Overall {transaction_type.capitalize()} Summary\n"
            text += "=" * 50 + "\n\n"
            text += f"Total {transaction_type.capitalize()}: ${total:,.2f}\n"
            text += f"Number of transactions: {len(profits) if transaction_type == 'profit' else len(losses)}\n"
        else:
            text = "Overall Financial Summary\n"
            text += "=" * 50 + "\n\n"
            text += f"Total Profit: ${total_profit:,.2f} ({len(profits)} transactions)\n"
            text += f"Total Loss: ${total_loss:,.2f} ({len(losses)} transactions)\n"
            text += f"Net: ${net:,.2f}\n"
        
        # Build JSON report
        json_report = {
            'type_filter': transaction_type,
            'summary': {
                'total_profit': total_profit,
                'total_loss': total_loss,
                'net': net,
                'profit_count': len(profits),
                'loss_count': len(losses)
            },
            'total_records': len(records)
        }
        
        return {
            'text': text,
            'json': json_report
        }

