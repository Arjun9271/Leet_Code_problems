import pandas as pd
import camelot
import tabula
import pdfplumber
import PyPDF2
import fitz  # pymupdf
import cv2
import numpy as np
from typing import List, Dict, Any
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_tables_camelot(pdf_path: str, pages: str = 'all') -> List[pd.DataFrame]:
    """
    Extract tables using Camelot (best for bordered tables)
    """
    try:
        # Try lattice method first (for bordered tables)
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor='lattice')
        if len(tables) > 0:
            logger.info(f"Camelot lattice found {len(tables)} tables")
            return [table.df for table in tables]
        
        # Try stream method for borderless tables
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor='stream')
        logger.info(f"Camelot stream found {len(tables)} tables")
        return [table.df for table in tables]
    
    except Exception as e:
        logger.error(f"Camelot extraction failed: {e}")
        return []

def extract_tables_tabula(pdf_path: str, pages: str = 'all') -> List[pd.DataFrame]:
    """
    Extract tables using Tabula-py (good for various table types)
    """
    try:
        # Try multiple methods
        methods = ['lattice', 'stream']
        all_tables = []
        
        for method in methods:
            try:
                if pages == 'all':
                    tables = tabula.read_pdf(pdf_path, pages='all', 
                                           multiple_tables=True, 
                                           lattice=(method == 'lattice'))
                else:
                    tables = tabula.read_pdf(pdf_path, pages=pages, 
                                           multiple_tables=True, 
                                           lattice=(method == 'lattice'))
                
                if tables:
                    all_tables.extend(tables)
                    logger.info(f"Tabula {method} found {len(tables)} tables")
            except Exception as e:
                logger.warning(f"Tabula {method} failed: {e}")
        
        return all_tables
    
    except Exception as e:
        logger.error(f"Tabula extraction failed: {e}")
        return []

def extract_tables_pdfplumber(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extract tables using pdfplumber (excellent for borderless tables)
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables with different settings
                page_tables = page.extract_tables()
                
                if not page_tables:
                    # Try with custom settings for borderless tables
                    page_tables = page.extract_tables(
                        table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "edge_min_length": 3,
                            "min_words_vertical": 3,
                            "min_words_horizontal": 1,
                            "intersection_tolerance": 3,
                        }
                    )
                
                for table in page_tables:
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = clean_dataframe(df)
                        tables.append(df)
                        logger.info(f"PDFplumber found table on page {page_num}")
        
        return tables
    
    except Exception as e:
        logger.error(f"PDFplumber extraction failed: {e}")
        return []

def extract_text_based_tables(pdf_path: str) -> List[pd.DataFrame]:
    """
    Extract tables by analyzing text patterns (for complex borderless tables)
    """
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # Look for table-like patterns
                    potential_tables = find_table_patterns(text)
                    for table_text in potential_tables:
                        df = parse_text_table(table_text)
                        if not df.empty:
                            tables.append(df)
                            logger.info(f"Text-based table found on page {page_num}")
        
        return tables
    
    except Exception as e:
        logger.error(f"Text-based extraction failed: {e}")
        return []

def find_table_patterns(text: str) -> List[str]:
    """
    Find potential table patterns in text
    """
    lines = text.split('\n')
    potential_tables = []
    current_table_lines = []
    
    for line in lines:
        # Check if line looks like a table row (has multiple words/numbers separated by spaces)
        if is_potential_table_row(line):
            current_table_lines.append(line)
        else:
            if len(current_table_lines) >= 3:  # Minimum 3 rows to consider as table
                potential_tables.append('\n'.join(current_table_lines))
            current_table_lines = []
    
    # Don't forget the last table
    if len(current_table_lines) >= 3:
        potential_tables.append('\n'.join(current_table_lines))
    
    return potential_tables

def is_potential_table_row(line: str) -> bool:
    """
    Check if a line looks like a table row
    """
    if not line.strip():
        return False
    
    # Check for multiple words/numbers separated by whitespace
    parts = line.split()
    if len(parts) < 2:
        return False
    
    # Check for consistent spacing patterns
    spaces = re.findall(r'\s+', line)
    if len(spaces) >= 2:
        return True
    
    # Check for tab-separated values
    if '\t' in line and len(line.split('\t')) >= 2:
        return True
    
    return False

def parse_text_table(table_text: str) -> pd.DataFrame:
    """
    Parse text that looks like a table into a DataFrame
    """
    lines = table_text.split('\n')
    if len(lines) < 2:
        return pd.DataFrame()
    
    # Try different parsing strategies
    strategies = [
        lambda x: x.split('\t'),  # Tab-separated
        lambda x: re.split(r'\s{2,}', x),  # Multiple spaces
        lambda x: x.split(),  # Single spaces (less reliable)
    ]
    
    for strategy in strategies:
        try:
            rows = []
            max_cols = 0
            
            for line in lines:
                if line.strip():
                    row = strategy(line.strip())
                    rows.append(row)
                    max_cols = max(max_cols, len(row))
            
            # Normalize row lengths
            normalized_rows = []
            for row in rows:
                while len(row) < max_cols:
                    row.append('')
                normalized_rows.append(row[:max_cols])
            
            if len(normalized_rows) >= 2 and max_cols >= 2:
                df = pd.DataFrame(normalized_rows[1:], columns=normalized_rows[0])
                return clean_dataframe(df)
        
        except Exception:
            continue
    
    return pd.DataFrame()

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the extracted DataFrame
    """
    if df.empty:
        return df
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # Clean column names
    df.columns = [str(col).strip() if col is not None else f'Column_{i}' 
                  for i, col in enumerate(df.columns)]
    
    # Clean cell values
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace('', pd.NA)
    
    # Remove rows that are mostly empty
    df = df[df.count(axis=1) > len(df.columns) * 0.3]
    
    return df

def merge_similar_tables(tables: List[pd.DataFrame], similarity_threshold: float = 0.7) -> List[pd.DataFrame]:
    """
    Merge tables that appear to be parts of the same table
    """
    if len(tables) <= 1:
        return tables
    
    merged_tables = []
    used_indices = set()
    
    for i, table1 in enumerate(tables):
        if i in used_indices:
            continue
        
        current_merged = table1.copy()
        used_indices.add(i)
        
        for j, table2 in enumerate(tables[i+1:], i+1):
            if j in used_indices:
                continue
            
            if tables_are_similar(table1, table2, similarity_threshold):
                try:
                    current_merged = pd.concat([current_merged, table2], ignore_index=True)
                    used_indices.add(j)
                except Exception:
                    pass
        
        merged_tables.append(current_merged)
    
    return merged_tables

def tables_are_similar(table1: pd.DataFrame, table2: pd.DataFrame, threshold: float) -> bool:
    """
    Check if two tables are similar enough to be merged
    """
    if table1.empty or table2.empty:
        return False
    
    # Check column similarity
    cols1 = set(table1.columns)
    cols2 = set(table2.columns)
    
    if len(cols1) == 0 or len(cols2) == 0:
        return False
    
    intersection = cols1.intersection(cols2)
    union = cols1.union(cols2)
    
    similarity = len(intersection) / len(union) if len(union) > 0 else 0
    return similarity >= threshold

def extract_all_tables(pdf_path: str, pages: str = 'all') -> Dict[str, List[pd.DataFrame]]:
    """
    Main function to extract tables using all methods
    """
    logger.info(f"Starting table extraction from {pdf_path}")
    
    results = {
        'camelot': [],
        'tabula': [],
        'pdfplumber': [],
        'text_based': []
    }
    
    # Try each extraction method
    methods = [
        ('camelot', lambda: extract_tables_camelot(pdf_path, pages)),
        ('tabula', lambda: extract_tables_tabula(pdf_path, pages)),
        ('pdfplumber', lambda: extract_tables_pdfplumber(pdf_path)),
        ('text_based', lambda: extract_text_based_tables(pdf_path))
    ]
    
    for method_name, method_func in methods:
        try:
            tables = method_func()
            results[method_name] = tables
            logger.info(f"{method_name} extracted {len(tables)} tables")
        except Exception as e:
            logger.error(f"{method_name} failed: {e}")
            results[method_name] = []
    
    return results

def get_best_tables(results: Dict[str, List[pd.DataFrame]]) -> List[pd.DataFrame]:
    """
    Select the best tables from all extraction methods
    """
    all_tables = []
    
    # Prioritize methods based on reliability
    method_priority = ['pdfplumber', 'camelot', 'tabula', 'text_based']
    
    for method in method_priority:
        if method in results and results[method]:
            all_tables.extend(results[method])
    
    # Remove duplicates and merge similar tables
    if all_tables:
        all_tables = merge_similar_tables(all_tables)
    
    return all_tables

def save_tables_to_excel(tables: List[pd.DataFrame], output_path: str):
    """
    Save extracted tables to Excel file with multiple sheets
    """
    if not tables:
        logger.warning("No tables to save")
        return
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for i, table in enumerate(tables):
            sheet_name = f'Table_{i+1}'
            table.to_excel(writer, sheet_name=sheet_name, index=False)
            logger.info(f"Saved table {i+1} to sheet {sheet_name}")

def save_tables_to_csv(tables: List[pd.DataFrame], output_dir: str):
    """
    Save extracted tables to separate CSV files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, table in enumerate(tables):
        csv_path = os.path.join(output_dir, f'table_{i+1}.csv')
        table.to_csv(csv_path, index=False)
        logger.info(f"Saved table {i+1} to {csv_path}")

# Main execution function
def main():
    """
    Example usage of the table extraction functions
    """
    pdf_path = "sample.pdf"  # Replace with your PDF path
    
    # Extract tables using all methods
    results = extract_all_tables(pdf_path)
    
    # Get the best tables
    best_tables = get_best_tables(results)
    
    print(f"\nExtracted {len(best_tables)} tables:")
    for i, table in enumerate(best_tables):
        print(f"\nTable {i+1} shape: {table.shape}")
        print(f"Columns: {list(table.columns)}")
        print("First few rows:")
        print(table.head())
    
    # Save results
    if best_tables:
        save_tables_to_excel(best_tables, "extracted_tables.xlsx")
        save_tables_to_csv(best_tables, "extracted_tables_csv")

if __name__ == "__main__":
    main()
