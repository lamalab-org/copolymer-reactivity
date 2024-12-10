from PreExtractionFilter.pre_extraction_filter import main as pre_extraction_filter
from copolextractor.PDF_download import main as pdf_download
from PreDownloadFilter.pre_download_filter import main as pre_download_filter
from copolextractor.crossref_search import main as crossref_search


def obtain_data(crossref_keyword, keywords, score_limit, number_of_selected_papers):
    # crossref search for relevant paper
    crossref_search(crossref_keyword)

    # metadata filter with keywords and embeddings
    pre_download_filter(keywords, score_limit, number_of_selected_papers)

    # PDF download with Scidownl
    pdf_download()

    # PDF quality RF-filter
    pre_extraction_filter()


def main():
    # Define Crossref Keywords
    crossref_keyword = "'copolymerization' AND 'reactivity ratio'"

    # Keywords and weights for pre download scoring
    keywords = {
        "copolymerization": 10,
        "polymerization": 5,
        "monomers": 5,
        "copolymers": 5,
        "ratios": 20,
        "reactivity ratios": 40,
    }

    score_limit = 65  # Minimum score for embedding generation
    number_of_selected_papers = 200  # Number of nearest papers to select

    # run obtain data pipline
    obtain_data(crossref_keyword, keywords, score_limit, number_of_selected_papers)


if __name__ == "__main__":
    main()
