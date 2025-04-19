import os
import fitz  # PyMuPDF
import re
import csv
import logging
from collections import defaultdict

# --- Configuration ---
# !!! MUST CHANGE: Set this to the root directory containing the 5000 folders !!!
ROOT_DATA_DIR = "/path/to/your/kaggle_dataset_root"
# !!! MUST CHANGE: Set this to where you want the output CSVs saved !!!
OUTPUT_DIR = "/path/to/your/output_csvs"
# The keyword required in the PDF filename
FILENAME_KEYWORD = "Paper"

# Define the target sections and potential variations in headings
# Order matters for the extraction logic (tries to find them sequentially)
TARGET_SECTIONS = [
    "Abstract",
    "Introduction", # Will also try to catch Lit Review here
    "Methods",      # Includes Methodology, Materials and Methods etc.
    "Results",      # Includes Findings
    "Discussion",
    "Conclusion"    # Includes Summary, Concluding Remarks
]

# Mapping variations to standard section names
HEADING_MAP = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "literature review": "Introduction",
    "background": "Introduction",
    "method": "Methods",
    "methods": "Methods",
    "methodology": "Methods",
    "materials and methods": "Methods",
    "experimental setup": "Methods",
    "experimental design": "Methods",
    "experimental": "Methods", # Be careful with broad terms
    "result": "Results",
    "results": "Results",
    "finding": "Results",
    "findings": "Results",
    "experimental results": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "summary": "Conclusion", # Can sometimes be ambiguous
    "concluding remarks": "Conclusion"
}

# Keywords that often signal the end of the main content
STOP_KEYWORDS = ["references", "bibliography", "acknowledgements", "appendix", "appendices", "supplementary material"]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def find_pdf_files(root_dir, keyword):
    """Walks through root_dir and finds PDF files containing the keyword."""
    pdf_files = []
    logging.info(f"Searching for PDF files containing '{keyword}' in '{root_dir}'...")
    if not os.path.isdir(root_dir):
        logging.error(f"Root directory '{root_dir}' not found.")
        return []

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # Case-insensitive check for keyword and extension
            if keyword.lower() in filename.lower() and filename.lower().endswith(".pdf"):
                full_path = os.path.join(dirpath, filename)
                pdf_files.append(full_path)

    logging.info(f"Found {len(pdf_files)} potential PDF files.")
    return pdf_files

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text += page.get_text("text") # Extract plain text
        doc.close()
        # Basic cleaning: replace multiple newlines/spaces
        full_text = re.sub(r'\s*\n\s*', '\n', full_text).strip()
        full_text = re.sub(r'[ \t]+', ' ', full_text)
        return full_text
    except Exception as e:
        logging.error(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
        return None

def identify_sections(full_text, filename):
    """
    Identifies text blocks corresponding to predefined sections using regex.
    This is heuristic and may not be perfect.
    """
    extracted_data = {section: "" for section in TARGET_SECTIONS}
    if not full_text:
        return extracted_data

    # --- Build Regex for Section Headings ---
    # This regex tries to find lines that likely start a section.
    # It looks for optional numbering (e.g., 1., II.), the keyword,
    # and ensures it's likely a heading (checking line start, allowing some whitespace).
    # It captures the keyword itself for mapping.
    section_keywords_pattern = "|".join(re.escape(k) for k in HEADING_MAP.keys())
    stop_keywords_pattern = "|".join(re.escape(k) for k in STOP_KEYWORDS)

    # Pattern: (Optional Numbering) (Section Keyword) (Rest of Line Break)
    # Using MULTILINE and IGNORECASE
    # We capture the keyword (group 1) to map it later
    heading_pattern = re.compile(
        r"^\s*(?:[\dIVX]+\.?\s*)*\s*(" + section_keywords_pattern + r")\b.*$",
        re.IGNORECASE | re.MULTILINE
    )
    stop_pattern = re.compile(
        r"^\s*(?:[\dIVX]+\.?\s*)*\s*(" + stop_keywords_pattern + r")\b.*$",
        re.IGNORECASE | re.MULTILINE
    )

    # Find all potential section heading matches
    matches = list(heading_pattern.finditer(full_text))
    stop_matches = list(stop_pattern.finditer(full_text))

    if not matches:
        logging.warning(f"No section headings found in {filename}. Skipping section extraction.")
        # Try a simple Abstract grab as a fallback if needed
        abstract_match = re.search(r"^\s*Abstract\b(.*?)^\s*(?:1\.|I\.|\bIntroduction)\b", full_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if abstract_match:
             extracted_data["Abstract"] = abstract_match.group(1).strip()
             logging.info(f"Found Abstract using fallback for {filename}")
        return extracted_data

    # Determine the end point of the last relevant section
    # This is the start of the first "stop" keyword (like References)
    # or the end of the document if no stop keyword is found *after* the last section match.
    last_section_end_pos = len(full_text)
    if stop_matches:
       # Find the first stop match that occurs *after* the last main section match
       last_match_pos = matches[-1].start()
       for stop_match in sorted(stop_matches, key=lambda m: m.start()):
            if stop_match.start() > last_match_pos:
                last_section_end_pos = stop_match.start()
                break

    # Iterate through matches to delineate sections
    for i, current_match in enumerate(matches):
        section_keyword_matched = current_match.group(1).lower() # The keyword found
        standard_section_name = HEADING_MAP.get(section_keyword_matched)

        if not standard_section_name: # Should not happen based on regex, but safety check
            continue

        # Define start and end points for the section's text
        start_pos = current_match.end() # Text starts *after* the heading line

        # End position is the start of the *next* section heading or the overall end point
        if i + 1 < len(matches):
            end_pos = matches[i+1].start()
        else:
            # If this is the last matched section, end at the calculated 'last_section_end_pos'
            end_pos = last_section_end_pos

        # Ensure end_pos doesn't precede start_pos (can happen with overlapping regex maybe)
        if end_pos < start_pos:
            end_pos = start_pos # Assign empty string effectively

        section_text = full_text[start_pos:end_pos].strip()

        # Append text (in case a section like "Methods" appears twice)
        # Use += if you expect sections to be split; use = if each heading restarts the section
        if extracted_data[standard_section_name]: # If already has text, add newline
             extracted_data[standard_section_name] += "\n\n" + section_text
        else:
             extracted_data[standard_section_name] = section_text

        # Special handling for Abstract: often short and might be captured poorly.
        # If Abstract is empty but found, try a more specific regex up to the next heading.
        if standard_section_name == "Abstract" and not section_text:
            try:
                next_heading_start = matches[i+1].start() if i+1 < len(matches) else len(full_text)
                abstract_chunk = full_text[current_match.start():next_heading_start]
                # More refined search within this chunk
                abs_match = re.search(r"^\s*Abstract\b(.*?)(?:^\s*(?:[\dIVX]+\.?\s*)*\s*(?:Introduction|Method|Result|Discussion|Conclusion)\b)", abstract_chunk, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                if abs_match:
                    extracted_data["Abstract"] = abs_match.group(1).strip()
            except Exception: # Ignore errors in this fallback
                 pass

    return extracted_data


def write_csvs(all_sections_data, output_dir):
    """Writes the extracted section data to separate CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    for section_name in TARGET_SECTIONS:
        csv_filename = os.path.join(output_dir, f"{section_name.replace(' ', '_').lower()}.csv")
        data_to_write = all_sections_data[section_name]

        if not data_to_write:
            logging.warning(f"No data found for section '{section_name}'. Skipping CSV creation.")
            continue

        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                # Assuming data_to_write is a list of dicts [{'filename': fn, 'text': txt}, ...]
                fieldnames = ['filename', 'text']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(data_to_write)
            logging.info(f"Successfully wrote {len(data_to_write)} entries to {csv_filename}")
        except IOError as e:
            logging.error(f"Could not write CSV file {csv_filename}: {e}")
        except Exception as e:
             logging.error(f"An unexpected error occurred while writing {csv_filename}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting PDF processing script.")

    # 1. Find relevant PDF files
    pdf_files_to_process = find_pdf_files(ROOT_DATA_DIR, FILENAME_KEYWORD)

    if not pdf_files_to_process:
        logging.warning("No PDF files found matching criteria. Exiting.")
        exit()

    # 2. Initialize data storage
    # Stores data as: {'Abstract': [{'filename': 'paper1.pdf', 'text': '...'}], 'Introduction': [...]}
    all_extracted_data = defaultdict(list)
    processed_count = 0
    error_count = 0

    # 3. Process each PDF
    for pdf_path in pdf_files_to_process:
        filename = os.path.basename(pdf_path)
        logging.info(f"Processing {filename}...")

        # Extract text
        full_text = extract_text_from_pdf(pdf_path)

        if full_text:
            # Identify sections
            sections = identify_sections(full_text, filename)

            # Add extracted sections to the main data structure
            found_any_section = False
            for section_name, text in sections.items():
                if text: # Only add if text was actually extracted
                    all_extracted_data[section_name].append({
                        'filename': filename,
                        'text': text
                    })
                    found_any_section = True
            if found_any_section:
                 processed_count += 1
            else:
                 logging.warning(f"No target sections extracted from {filename}.")
                 # Optionally keep track of files with no extracted sections
        else:
            error_count += 1
            # Error already logged in extract_text_from_pdf

    logging.info(f"Finished processing files. Successfully processed: {processed_count}, Errors: {error_count}")

    # 4. Write data to CSVs
    write_csvs(all_extracted_data, OUTPUT_DIR)

    logging.info("Script finished.")