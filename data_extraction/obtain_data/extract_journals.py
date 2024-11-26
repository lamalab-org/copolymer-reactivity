from selenium import webdriver
from bs4 import BeautifulSoup
import json

# Set up Selenium
driver = webdriver.Chrome()
url = "https://chemsearch.kovsky.net/supported_journals.php"
driver.get(url)

# Parse the HTML with BeautifulSoup
soup = BeautifulSoup(driver.page_source, 'html.parser')
driver.quit()

# Extract journal names from the first <td> in each <tr>
journals = []
for row in soup.find_all('tr'):
    cols = row.find_all('td')
    if cols:  # Check if <td> tags exist in the row
        journal_name = cols[0].get_text(strip=True)
        journals.append(journal_name)

# Save to JSON as a list
with open("journals.json", "w") as f:
    json.dump(journals, f, indent=2)
