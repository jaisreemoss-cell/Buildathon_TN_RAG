from langchain_community.document_loaders import PlaywrightURLLoader

URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"
OUTPUT_FILENAME = "scraped_schemes.txt"

# The Tamil Nadu block sits between these two section headings on the page.
START_MARKER = "Tamil Nadu"
END_MARKER = "Kerala"


def extract_tamil_nadu_section(text: str) -> str:
    """Return only the text between the 'Tamil Nadu' and 'Kerala' headings."""
    start = text.find(START_MARKER)
    if start == -1:
        raise ValueError(f"Start marker '{START_MARKER}' not found; page layout may have changed.")

    end = text.find(END_MARKER, start)
    if end == -1:
        raise ValueError(f"End marker '{END_MARKER}' not found; page layout may have changed.")

    section = text[start:end]

    # Drop blank lines and leftover navigation text such as "Top" links.
    lines = [line.strip() for line in section.split("\n")]
    lines = [line for line in lines if line and line.lower() != "top"]
    return "\n".join(lines)


def scrape_and_filter_tnau():
    print("Initializing Playwright browser...")
    loader = PlaywrightURLLoader(urls=[URL], headless=True)

    try:
        print("Fetching webpage content...")
        documents = loader.load()

        if not documents:
            print("Warning: No documents returned.")
            return

        raw_text = documents[0].page_content
        print("Raw data fetched successfully. Extracting Tamil Nadu section...")

        tn_text = extract_tamil_nadu_section(raw_text)

        with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
            f.write(tn_text)

        print(f"Success! Tamil Nadu content saved to '{OUTPUT_FILENAME}'.")
        print(f"Characters kept: {len(tn_text)} of {len(raw_text)}")

        print("\n--- Preview of Filtered Content ---")
        print(tn_text[:400] + "...")
        print("\n--- Last lines (should end before the Kerala section) ---")
        print(tn_text[-300:])
        print("-----------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    scrape_and_filter_tnau()