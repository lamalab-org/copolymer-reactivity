from selenium import webdriver
from bs4 import BeautifulSoup
import json
from collections import Counter


def fetch_journals(output_journal_file):
    """Fetch journal names from the website and save to a JSON file."""
    driver = webdriver.Chrome()
    url = "https://chemsearch.kovsky.net/supported_journals.php"
    driver.get(url)

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Extract journal names
    journals = []
    for row in soup.find_all("tr"):
        cols = row.find_all("td")
        if cols:  # Check if <td> tags exist in the row
            journal_name = cols[0].get_text(strip=True)
            journals.append(journal_name)

    # Save to JSON
    with open(output_journal_file, "w") as f:
        json.dump(journals, f, indent=2)
    print(f"Journals saved to {output_journal_file}.")


def calculate_score(entry, journal_list, keywords):
    """Calculate the score for a given paper."""
    score = 0

    # Check if 'Journal' key exists and if journal is in the list
    if "Journal" in entry:
        journal_in_list = any(
            journal.lower() in entry["Journal"].lower() for journal in journal_list
        )
        if journal_in_list:
            score += 40

    # Check for weighted keywords in title and abstract
    title_abstract = entry.get("Title", "") + " " + entry.get("Abstract", "")
    for word, weight in keywords.items():
        if word.lower() in title_abstract.lower():
            score += weight

    entry["Score"] = score
    return entry


def process_papers(input_file, journal_file, keywords, output_file):
    """Process papers, calculate scores, and save results to a JSON file."""
    # Load journals
    with open(journal_file, "r") as f:
        journal_list = json.load(f)

    # Load papers
    with open(input_file, "r") as f:
        data = json.load(f)

    # Process entries and calculate scores
    scored_data = [calculate_score(entry, journal_list, keywords) for entry in data]

    # Sort by score and get top 50
    top_50_papers = sorted(scored_data, key=lambda x: x["Score"], reverse=True)[:50]

    # Print top 50 papers with title and score
    print("\nTop 50 papers by score:")
    for paper in top_50_papers:
        print(f"Title: {paper['Title']}, Score: {paper['Score']}")

    # Print score distribution
    score_counts = Counter(entry["Score"] for entry in scored_data)
    print("\nScore distribution:")
    for score, count in sorted(score_counts.items(), reverse=True):
        print(f"Score {score}: {count} papers")

    print("Total number of papers:", len(scored_data))

    # Save the updated data to a new JSON file
    with open(output_file, "w") as f:
        json.dump(scored_data, f, indent=2)
    print(f"Scored papers saved to {output_file}.")


def main():
    # Input and output files
    journal_file = "../../data_extraction/obtain_data/output/journals.json"
    input_file = "../../data_extraction/obtain_data/output/collected_doi_metadata.json"
    output_file = "../../data_extraction/obtain_data/output/scored_doi.json"

    # Keywords and their weights
    keywords = {
        "copolymerization": 10,
        "polymerization": 5,
        "monomers": 5,
        "copolymers": 5,
        "ratios": 20,
        "reactivity ratios": 40,
    }

    # Step 1: Fetch journals and save to a JSON file
    fetch_journals(journal_file)

    # Step 2: Process papers and calculate scores
    process_papers(input_file, journal_file, keywords, output_file)


if __name__ == "__main__":
    main()
