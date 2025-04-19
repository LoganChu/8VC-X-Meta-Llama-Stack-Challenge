import time
import os
import csv
import re
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import requests
import fitz  # PyMuPDF

# --- Configuration ---
SEARCH_TERM = "research"
START_PAGE = 0  # Google Scholar page numbering starts at 0 (0, 10, 20...)
MAX_PAGES_TO_SCRAPE = 2 # How many pages of results to process
DOWNLOAD_FOLDER = "downloaded_pdfs"
CSV_FOLDER = "output_csvs"
BASE_URL = "https://scholar.google.com/scholar?start={start}&q={query}&hl=en&as_sdt=0,5"

# --- Section Keywords (Heuristics - NEEDS ADJUSTMENT & WILL BE IMPERFECT) ---
# Using regex for flexibility (case-insensitive, potential variations)
SECTIONS = {
    "abstract": r"Abstract",
    "introduction": r"(Introduction|Background|Literature Review)",
    "methods": r"(Method|Methodology|Materials and Methods|Experimental Setup)",
    "results": r"(Results|Findings)",
    "discussion": r"Discussion",
    "conclusion": r"(Conclusion|Summary|Concluding Remarks)",
}

# Keywords that typically mark the *start* of the *next* section
# Used to find the end boundary of the current section
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

# --- Helper Functions ---

def setup_driver():
    """Initializes the Selenium WebDriver."""
    print("Setting up WebDriver...")
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") # Run in background
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36") # Mimic browser
    options.add_experimental_option('excludeSwitches', ['enable-logging']) # Suppress USB/Bluetooth errors
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.implicitly_wait(5) # Implicit wait for elements
    print("WebDriver setup complete.")
    return driver

def safe_filename(title):
    """Creates a safe filename from an article title."""
    # Remove invalid characters
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    # Replace spaces with underscores
    title = title.replace(" ", "_")
    # Truncate if too long
    return title[:100]

def download_pdf(pdf_url, filename):
    """Downloads a PDF from a URL, handling potential redirects and errors."""
    print(f"Attempting to download PDF from: {pdf_url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        # Use stream=True to handle large files and get headers first
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30, allow_redirects=True)
        response.raise_for_status() # Raise exception for bad status codes

        # Check if the content is actually PDF
        content_type = response.headers.get('Content-Type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"Warning: URL did not return PDF content type. Got: {content_type}. Skipping download.")
            # Sometimes content-disposition might have filename.pdf
            content_disposition = response.headers.get('Content-Disposition', '')
            if '.pdf' not in content_disposition.lower():
               return None # Definitely not a PDF or download link

        # Get effective URL after potential redirects
        effective_url = response.url
        print(f"  Effective URL: {effective_url}")

        # Check if it's likely an HTML page instead of direct PDF link
        if not effective_url.lower().endswith('.pdf') and 'application/pdf' not in content_type:
             print("  Warning: Effective URL doesn't end with .pdf and content type isn't PDF. Likely a landing page. Skipping.")
             return None

        filepath = os.path.join(DOWNLOAD_FOLDER, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded: {filename}")
        return filepath
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading PDF: {e}")
        return None
    except Exception as e:
        print(f"  An unexpected error occurred during download: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyMuPDF."""
    print(f"Extracting text from: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        doc.close()
        print(f"  Text extracted successfully ({len(text)} chars).")
        # Basic cleaning - replace multiple newlines/spaces
        text = re.sub(r'\s*\n\s*', '\n', text).strip()
        text = re.sub(r'[ \t]+', ' ', text)
        return text
    except Exception as e:
        print(f"  Error extracting text from PDF {pdf_path}: {e}")
        return None

def find_section_text(full_text, section_name, start_regex, end_regex_list):
    """
    Heuristic function to find text for a specific section.
    Finds start_regex, then looks for the *earliest* match from end_regex_list.
    Returns the text between start and end match.
    THIS IS VERY IMPERFECT.
    """
    start_match = re.search(start_regex, full_text, re.IGNORECASE | re.MULTILINE)
    if not start_match:
        # print(f"  Section '{section_name}' start marker not found.")
        return "" # Section start not found

    start_pos = start_match.end() # Position *after* the start marker
    end_pos = len(full_text) # Default to end of document if no end marker found

    # Find the earliest occurrence of any end marker *after* the start marker
    first_end_marker_pos = len(full_text)
    found_end_marker = False

    for end_regex in end_regex_list:
        # Search only in the text *after* the start marker
        end_match = re.search(end_regex, full_text[start_pos:], re.IGNORECASE | re.MULTILINE)
        if end_match:
            # Calculate the absolute position in the original text
            current_end_pos = start_pos + end_match.start()
            if current_end_pos < first_end_marker_pos:
                first_end_marker_pos = current_end_pos
                found_end_marker = True

    end_pos = first_end_marker_pos

    section_content = full_text[start_pos:end_pos].strip()

    # Basic sanity check: avoid excessively long sections if an end marker was missed
    # Or very short sections that might just be the heading itself
    if len(section_content) > 50000: # Arbitrary limit, adjust as needed
        # print(f"  Warning: Section '{section_name}' seems very long, might be truncated.")
        # return section_content[:50000] + "..." # Or return empty if likely wrong
         return ""
    if len(section_content) < 50: # Arbitrary limit, adjust as needed
        # print(f"  Warning: Section '{section_name}' seems very short, might be incorrect.")
        return ""

    # print(f"  Found section '{section_name}' (approx. {len(section_content)} chars).")
    return section_content


def append_to_csv(filepath, data_row):
    """Appends a row to a CSV file."""
    is_new_file = not os.path.exists(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if is_new_file:
            writer.writerow(["Text"]) # Write header only if file is new
        writer.writerow([data_row])

# --- Main Script ---

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    os.makedirs(CSV_FOLDER, exist_ok=True)

    # Initialize CSV files (or clear existing ones by writing header)
    for section, filepath in CSV_FILES.items():
         with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Text"]) # Write header

    driver = setup_driver()
    wait = WebDriverWait(driver, 10)
    processed_articles = 0
    pdfs_downloaded = 0

    try:
        for page in range(START_PAGE, MAX_PAGES_TO_SCRAPE):
            page_start_index = page * 10
            url = BASE_URL.format(start=page_start_index, query=SEARCH_TERM)
            print(f"\n--- Scraping Page {page + 1} ({url}) ---")
            driver.get(url)

            # Basic check for CAPTCHA - very rudimentary
            if "CAPTCHA" in driver.page_source or "robots.txt" in driver.title:
                print("!!! CAPTCHA or block detected. Stopping script. !!!")
                print("Please solve the CAPTCHA manually in the browser or wait and try again later.")
                input("Press Enter after solving CAPTCHA (if possible) or to exit...")
                # Try reloading or just break
                break # Stop scraping if blocked

            # Wait for results to load
            try:
                wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.gs_r.gs_or.gs_scl")))
                print("Results loaded.")
            except TimeoutException:
                print("Timed out waiting for page results. Skipping page or stopping.")
                if "Our systems have detected unusual traffic" in driver.page_source:
                     print("!!! Google block detected !!!")
                     break
                continue # Try next page

            # Find all result blocks
            results = driver.find_elements(By.CSS_SELECTOR, "div.gs_r.gs_or.gs_scl")
            print(f"Found {len(results)} results on page.")
            pdfs_found_on_page = 0

            for result in results:
                processed_articles += 1
                pdf_link_element = None
                pdf_url = None
                title = "unknown_article"
                title_element = None

                try:
                    title_element = result.find_element(By.CSS_SELECTOR, "h3.gs_rt a")
                    title = title_element.text
                except NoSuchElementException:
                    try:
                        # Handle titles that are not links (e.g., citations)
                        title_element = result.find_element(By.CSS_SELECTOR, "h3.gs_rt")
                        title = title_element.text
                    except NoSuchElementException:
                        print("  Could not find title for a result.")
                        continue # Skip this result if no title found

                # Try to find the [PDF] link specifically
                try:
                    pdf_link_element = result.find_element(By.CSS_SELECTOR, "div.gs_or_ggsm a[href$='.pdf']")
                    pdf_url = pdf_link_element.get_attribute('href')
                    print(f"  Found direct PDF link for: '{title}'")
                except NoSuchElementException:
                    # Look for links marked [PDF] even if they don't end in .pdf
                    try:
                        pdf_markers = result.find_elements(By.XPATH, ".//span[contains(text(), '[PDF]')]")
                        if pdf_markers:
                             # Find the associated link, often the parent div's link
                             parent_div = pdf_markers[0].find_element(By.XPATH, "./ancestor::div[contains(@class, 'gs_ggsd')]")
                             pdf_link_element = parent_div.find_element(By.TAG_NAME, 'a')
                             pdf_url = pdf_link_element.get_attribute('href')
                             print(f"  Found [PDF] marker link for: '{title}'")
                        else:
                            # print(f"  No direct PDF link found for: '{title}'")
                            pass # Continue, maybe process non-PDF links later if desired
                    except NoSuchElementException:
                       # print(f"  No direct PDF link found for: '{title}'")
                       pass # Continue


                if pdf_url:
                    pdfs_found_on_page += 1
                    # Generate a safe filename
                    base_filename = safe_filename(title)
                    pdf_filename = f"{base_filename}_{processed_articles}.pdf"

                    # Download the PDF
                    pdf_filepath = download_pdf(pdf_url, pdf_filename)

                    if pdf_filepath:
                        pdfs_downloaded += 1
                        # Extract text
                        full_text = extract_text_from_pdf(pdf_filepath)

                        if full_text:
                            # Attempt to extract sections
                            extracted_data = {}
                            section_keys = list(SECTIONS.keys())

                            for i, section_name in enumerate(section_keys):
                                start_regex = SECTIONS[section_name]

                                # Define potential end markers: the start of all subsequent sections + generic terminators
                                current_end_markers = END_MARKERS_ORDERED[i:] # Take relevant end markers based on order

                                section_text = find_section_text(full_text, section_name, start_regex, current_end_markers)
                                extracted_data[section_name] = section_text

                            # Append extracted sections to respective CSVs
                            print(f"  Appending extracted sections to CSVs for: {pdf_filename}")
                            for section_name, text_content in extracted_data.items():
                                if text_content: # Only append if something was found
                                    append_to_csv(CSV_FILES[section_name], text_content)
                                # else:
                                    # print(f"    - Section '{section_name}' not found or empty.")


                            # Optional: Delete PDF after processing to save space
                            try:
                                os.remove(pdf_filepath)
                                # print(f"  Removed processed PDF: {pdf_filepath}")
                            except OSError as e:
                                print(f"  Error removing PDF {pdf_filepath}: {e}")
                        else:
                             print(f"  Could not extract text from {pdf_filename}, skipping section analysis.")
                    else:
                        print(f"  Failed to download PDF for: '{title}'")

                # Add a small random delay between processing articles
                time.sleep(random.uniform(1, 3)) # Be polite

            if pdfs_found_on_page == 0:
                 print(f"No PDF links found on page {page + 1}.")
                 # Continue to the next page as requested

            # Check if there is a "Next" button/link
            try:
                # Google Scholar's next button is often just a link with 'Next' text
                # It might be within a specific navigation element, e.g., #gs_n
                next_button = driver.find_element(By.CSS_SELECTOR, '#gs_n a:last-child') # Often the last link in nav
                # Or more reliably by link text if the selector fails
                # next_button = driver.find_element(By.LINK_TEXT, 'Next')

                if "Next" in next_button.text: # Make sure it's actually the next button
                    print("Navigating to the next page...")
                    time.sleep(random.uniform(3, 7)) # Longer delay before next page load
                    next_button.click()
                else:
                    print("Could not find 'Next' button text. Reached the end?")
                    break # Exit loop if no clear next button
            except NoSuchElementException:
                print("No 'Next' button found. Assuming end of results.")
                break # Exit loop

    except Exception as e:
        print(f"\nAn unexpected error occurred during scraping: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
    finally:
        print("\n--- Scraping Finished ---")
        print(f"Total articles checked (approx): {processed_articles}")
        print(f"Total PDFs downloaded and processed: {pdfs_downloaded}")
        if driver:
            print("Closing WebDriver.")
            driver.quit()