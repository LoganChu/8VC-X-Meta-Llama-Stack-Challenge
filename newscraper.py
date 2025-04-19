import subprocess
import os
import glob
import csv
import re
import time
import fitz  # PyMuPDF

# --- Configuration ---
SEARCH_TERM = "research"
SCHOLAR_PAGES = 2  # Number of Google Scholar pages to fetch results from (e.g., 2 pages = ~20 results)
DOWNLOAD_FOLDER = "pypaperbot_downloads" # MUST match PyPaperBot's --dwn-dir
CSV_FOLDER = "output_csvs_pypaperbot"
DELETE_PDF_AFTER_PROCESSING = True # Set to False to keep downloaded PDFs

# --- Section Keywords (Heuristics - NEEDS ADJUSTMENT & WILL BE IMPERFECT) ---
# Using regex for flexibility (case-insensitive, potential variations)
SECTIONS = {
    "abstract": r"Abstract",
    "introduction": r"(Introduction|Background|Literature Review)",
    "methods": r"(Method|Methodology|Materials and Methods|Experimental Setup|Experimental)",
    "results": r"(Results|Findings)",
    "discussion": r"Discussion",
    "conclusion": r"(Conclusion|Summary|Concluding Remarks)",
}

# Keywords that typically mark the *start* of the *next* section
# Used to find the end boundary of the current section. Order matters!
# Includes common section starters and paper terminators.
END_MARKERS_ORDERED = [
    SECTIONS["introduction"],
    SECTIONS["methods"],
    SECTIONS["results"],
    SECTIONS["discussion"],
    SECTIONS["conclusion"],
    r"(References|Bibliography|Acknowledgements|Acknowledgments|Appendix|Supplementary Material)" # Common endings
]

CSV_FILES = {
    section: os.path.join(CSV_FOLDER, f"{section}_data.csv") for section in SECTIONS
}

# --- Helper Functions (Similar to previous script) ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    print(f"  Extracting text from: {os.path.basename(pdf_path)}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            # Attempt to remove headers/footers heuristically (e.g., short lines, page numbers)
            lines = page_text.split('\n')
            cleaned_lines = [line for line in lines if len(line.strip()) > 10 and not re.match(r"^\s*\d+\s*$", line.strip())]
            text += "\n".join(cleaned_lines) + "\n"

        doc.close()
        # Basic cleaning - replace multiple newlines/spaces
        text = re.sub(r'\s*\n\s*', '\n', text).strip()
        text = re.sub(r'[ \t]+', ' ', text)
        print(f"    -> Text extracted successfully ({len(text):,} chars).")
        return text
    except Exception as e:
        print(f"    -> Error extracting text from PDF {os.path.basename(pdf_path)}: {e}")
        return None

def find_section_text(full_text, section_name, start_regex, end_regex_list):
    """
    Heuristic function to find text for a specific section.
    Finds start_regex, then looks for the *earliest* match from end_regex_list.
    Returns the text between start and end match. THIS IS VERY IMPERFECT.
    """
    # Search for the start marker - allow it to be at the beginning of a line maybe preceded by whitespace
    start_match = re.search(r"^\s*" + start_regex + r"\s*$", full_text, re.IGNORECASE | re.MULTILINE)

    # Fallback: search anywhere if not found at line start (less precise)
    if not start_match:
         start_match = re.search(start_regex, full_text, re.IGNORECASE | re.MULTILINE)

    if not start_match:
        # print(f"    -> Section '{section_name}' start marker not found.")
        return "" # Section start not found

    start_pos = start_match.end() # Position *after* the start marker
    end_pos = len(full_text) # Default to end of document if no end marker found

    # Find the earliest occurrence of any *subsequent* section's start marker *after* the current section's start marker
    first_end_marker_pos = len(full_text)
    found_end_marker = None

    for end_regex in end_regex_list:
        # Search only in the text *after* the start marker
        # Look for end markers potentially at start of line too
        for end_match in re.finditer(r"^\s*" + end_regex + r"\s*$", full_text[start_pos:], re.IGNORECASE | re.MULTILINE):
             current_end_pos = start_pos + end_match.start() # Absolute position of the *start* of the end marker
             if current_end_pos < first_end_marker_pos:
                 first_end_marker_pos = current_end_pos
                 found_end_marker = end_regex
             # Optimization: If we find an end marker very close, it's likely correct
             # if (current_end_pos - start_pos) > 50: break

        # Fallback search anywhere (less precise) if no start-of-line match found yet or first end marker is still at the end
        if first_end_marker_pos == len(full_text) or not found_end_marker:
             for end_match in re.finditer(end_regex, full_text[start_pos:], re.IGNORECASE | re.MULTILINE):
                 current_end_pos = start_pos + end_match.start()
                 if current_end_pos < first_end_marker_pos:
                      first_end_marker_pos = current_end_pos
                      found_end_marker = end_regex
                 # if (current_end_pos - start_pos) > 50: break


    end_pos = first_end_marker_pos
    section_content = full_text[start_pos:end_pos].strip()

    # Basic sanity checks
    if not section_content:
        return ""
    # Avoid grabbing just noise if section is super short
    if len(section_content) < 50:
        # print(f"    -> Section '{section_name}' too short, likely incorrect.")
        return ""
    # Avoid overly long sections if end marker was missed (adjust limit as needed)
    # if len(section_content) > 75000:
    #     print(f"    -> Warning: Section '{section_name}' seems very long, might be misidentified.")
        # return "" # Or truncate: return section_content[:75000] + "... [TRUNCATED]"


    # print(f"    -> Found section '{section_name}' (approx. {len(section_content):,} chars). End marker triggered: {found_end_marker}")
    return section_content


def append_to_csv(filepath, data_row):
    """Appends a row to a CSV file."""
    try:
        # Ensure the directory exists right before writing
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        is_new_file = not os.path.exists(filepath)
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if is_new_file:
                writer.writerow(["Text"]) # Write header only if file is new
            writer.writerow([data_row])
    except Exception as e:
         print(f"    -> Error writing to CSV {filepath}: {e}")


# --- Main Script ---

if __name__ == "__main__":
    start_time = time.time()
    # --- Step 1: Create directories ---
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(CSV_FOLDER, exist_ok=True)
    print(f"Download directory: {os.path.abspath(DOWNLOAD_FOLDER)}")
    print(f"CSV output directory: {os.path.abspath(CSV_FOLDER)}")

    # --- Step 2: Initialize CSV files ---
    print("Initializing CSV files...")
    for section, filepath in CSV_FILES.items():
         try:
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Text"]) # Write header, clearing existing file
         except Exception as e:
             print(f"  Error initializing CSV {filepath}: {e}")
             # Decide if you want to stop or continue
             # exit() # Or just print warning and continue

    # --- Step 3: Run PyPaperBot ---
    print("\n--- Running PyPaperBot to download papers ---")
    # Construct the command. Use restrict=1 to prioritize PDFs.
    pypaperbot_cmd = [
        "python", "-m", "PyPaperBot",
        "--query", SEARCH_TERM,
        "--scholar-pages", str(SCHOLAR_PAGES),
        "--dwn-dir", DOWNLOAD_FOLDER,
        "--restrict=1" # Crucial: Attempt to download only PDFs
        # Add other PyPaperBot arguments if needed:
        # "--min-year=2020",
        # "--proxy=http://user:pass@host:port",
        # "--scihub-mirror=https://sci-hub.se" # Example mirror
    ]
    print(f"Executing command: {' '.join(pypaperbot_cmd)}")

    try:
        # Run PyPaperBot as a subprocess
        result = subprocess.run(pypaperbot_cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("--- PyPaperBot Output ---")
        print(result.stdout)
        if result.stderr:
            print("--- PyPaperBot Errors/Warnings ---")
            print(result.stderr)
        print("--- PyPaperBot finished ---")

    except FileNotFoundError:
        print("\nError: 'python' command not found.")
        print("Please ensure Python is installed and in your system's PATH.")
        print("You also need to install PyPaperBot: pip install PyPaperBot")
        exit()
    except subprocess.CalledProcessError as e:
        print(f"\nError: PyPaperBot exited with status {e.returncode}")
        print("--- PyPaperBot Output (Error) ---")
        print(e.stdout)
        print("--- PyPaperBot Stderr (Error) ---")
        print(e.stderr)
        print("PyPaperBot failed to execute correctly. Please check its output above.")
        # Decide whether to proceed with potentially incomplete downloads or exit
        # exit()
    except Exception as e:
        print(f"\nAn unexpected error occurred while running PyPaperBot: {e}")
        # exit()

    # --- Step 4: Process Downloaded PDFs ---
    print("\n--- Processing downloaded PDFs ---")
    pdf_files = glob.glob(os.path.join(DOWNLOAD_FOLDER, '*.pdf'))
    print(f"Found {len(pdf_files)} PDF files in {DOWNLOAD_FOLDER}")

    processed_count = 0
    section_counts = {section: 0 for section in SECTIONS}

    if not pdf_files:
        print("No PDF files found to process. Did PyPaperBot download anything?")
    else:
        for pdf_path in pdf_files:
            print(f"\nProcessing: {os.path.basename(pdf_path)}")
            processed_count += 1
            full_text = extract_text_from_pdf(pdf_path)

            if full_text:
                # Attempt to extract sections
                extracted_data = {}
                section_keys = list(SECTIONS.keys())

                for i, section_name in enumerate(section_keys):
                    start_regex = SECTIONS[section_name]
                    # Define potential end markers: the start of all subsequent sections + generic terminators
                    current_end_markers = END_MARKERS_ORDERED[i:]

                    section_text = find_section_text(full_text, section_name, start_regex, current_end_markers)
                    if section_text:
                        extracted_data[section_name] = section_text
                        section_counts[section_name] += 1
                        append_to_csv(CSV_FILES[section_name], section_text)
                    # else:
                    #     print(f"    -> Section '{section_name}' not found or empty.")

                if not extracted_data:
                     print("    -> No sections could be reliably extracted for this PDF.")

            else:
                print(f"  Skipping section analysis due to text extraction failure.")

            # Optional: Delete PDF after processing
            if DELETE_PDF_AFTER_PROCESSING and full_text: # Only delete if processed ok
                try:
                    os.remove(pdf_path)
                    print(f"  -> Removed processed PDF: {os.path.basename(pdf_path)}")
                except OSError as e:
                    print(f"  -> Error removing PDF {os.path.basename(pdf_path)}: {e}")

    # --- Step 5: Final Report ---
    end_time = time.time()
    print("\n--- Processing Complete ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"Attempted to process {processed_count} PDF files found in {DOWNLOAD_FOLDER}.")
    print("Sections found and added to CSVs:")
    for section, count in section_counts.items():
        print(f"  - {section.capitalize()}: {count}")
    print(f"CSV files are located in: {os.path.abspath(CSV_FOLDER)}")
    if DELETE_PDF_AFTER_PROCESSING and processed_count > 0:
        print("Processed PDF files have been deleted.")
    elif processed_count > 0:
         print(f"Original PDF files remain in: {os.path.abspath(DOWNLOAD_FOLDER)}")