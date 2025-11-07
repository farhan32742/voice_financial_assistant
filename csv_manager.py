"""
CSV data manager for reading and writing financial records.
"""

import csv
import os
from typing import List, Dict, Optional
from datetime import datetime


class CSVManager:
    """Manages CSV file operations for financial records."""
    
    def __init__(self, csv_file: str = "financial_records.csv"):
        """
        Initialize CSV manager.
        
        Args:
            csv_file: Path to the CSV file
        """
        self.csv_file = csv_file
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist or is missing them."""
        expected_header = ['type', 'amount', 'date', 'details']

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=expected_header)
                writer.writeheader()
            return

        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            if not rows:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=expected_header)
                    writer.writeheader()
                return

            header = [column.strip().lower() for column in rows[0]]
            if header != expected_header:
                with open(self.csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(expected_header)
                    writer.writerows(rows)
        except Exception as exc:
            print(f"Warning: failed to ensure CSV header due to error: {exc}")

    def _ensure_header_before_io(self):
        """Ensure header integrity before any read/write operation."""
        try:
            self._ensure_file_exists()
        except Exception as exc:
            print(f"Warning: ensure header before IO failed: {exc}")
    
    def save_record(self, record: Dict, check_duplicates: bool = True) -> bool:
        """
        Save a single record to CSV.
        
        Args:
            record: Dictionary with type, amount, date, and details
            check_duplicates: If True, check for duplicates before saving (default: True)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Always ensure header correctness before writing
            self._ensure_header_before_io()
            # Check for duplicates if enabled
            if check_duplicates:
                existing_records = self.read_all_records()
                record_key = (str(record.get('type', '')).lower(), 
                            str(record.get('amount', '')), 
                            str(record.get('date', '')), 
                            str(record.get('details', '')))
                
                for existing in existing_records:
                    existing_key = (str(existing.get('type', '')).lower(),
                                  str(existing.get('amount', '')),
                                  str(existing.get('date', '')),
                                  str(existing.get('details', '')))
                    if record_key == existing_key:
                        print(f"Duplicate record detected and skipped: {record}")
                        return False  # Don't save duplicate
            
            # Save the record
            with open(self.csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['type', 'amount', 'date', 'details'])
                writer.writerow(record)
            return True
        except Exception as e:
            print(f"Error saving record: {e}")
            return False
    
    def read_all_records(self) -> List[Dict]:
        """
        Read all records from CSV.
        
        Returns:
            List of dictionaries containing all records
        """
        records = []
        # Ensure header is correct before reading
        self._ensure_header_before_io()
        if not os.path.exists(self.csv_file):
            return records
        
        try:
            with open(self.csv_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert amount to float
                    row['amount'] = float(row['amount'])
                    records.append(row)
        except Exception as e:
            print(f"Error reading records: {e}")
        
        return records
    
    def get_records_by_type(self, transaction_type: str) -> List[Dict]:
        """
        Get all records of a specific type (profit or loss).
        
        Args:
            transaction_type: 'profit' or 'loss'
            
        Returns:
            List of matching records
        """
        all_records = self.read_all_records()
        return [r for r in all_records if r['type'].lower() == transaction_type.lower()]
    
    def get_records_by_date_range(self, start_date: Optional[str] = None, 
                                   end_date: Optional[str] = None) -> List[Dict]:
        """
        Get records within a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive)
            end_date: End date in YYYY-MM-DD format (inclusive)
            
        Returns:
            List of matching records
        """
        all_records = self.read_all_records()
        
        if not start_date and not end_date:
            return all_records
        
        filtered = []
        for record in all_records:
            record_date = record['date']
            
            if start_date and record_date < start_date:
                continue
            if end_date and record_date > end_date:
                continue
            
            filtered.append(record)
        
        return filtered
    
    def get_records_by_month(self, year: int, month: int) -> List[Dict]:
        """
        Get all records for a specific month.
        
        Args:
            year: Year (e.g., 2024)
            month: Month (1-12)
            
        Returns:
            List of matching records
        """
        all_records = self.read_all_records()
        filtered = []
        
        for record in all_records:
            try:
                record_date = datetime.strptime(record['date'], '%Y-%m-%d')
                if record_date.year == year and record_date.month == month:
                    filtered.append(record)
            except ValueError:
                continue
        
        return filtered
    
    def get_records_by_date(self, date: str) -> List[Dict]:
        """
        Get all records for a specific date.
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            List of matching records
        """
        all_records = self.read_all_records()
        return [r for r in all_records if r['date'] == date]

