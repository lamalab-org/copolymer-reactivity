from copolextractor.preextractionfilter import main as pre_extraction_filter
from copolextractor.PDF_download import main as pdf_download
from copolextractor.predownloadfilter import main as pre_download_filter
from copolextractor.crossref_search import main as crossref_search
from copolextractor.extraction_with_GPT_PDF import main as extractor


def obtain_data(
    crossref_keyword,
    keywords,
    score_limit,
    number_of_selected_papers,
    input_folder_images,
    output_folder,
    paper_list_path,
    pdf_folder,
    extracted_data_file,
    seed_rf,
    threshold,
    pdf_input_folder,
    output_folder_images,
    output_folder_score,
    training_file,
    scoring_file,
    output_file
):
    # crossref search for relevant paper
    crossref_search(crossref_keyword)

    # metadata filter with keywords and embeddings
    pre_download_filter(keywords, score_limit, number_of_selected_papers)

    # PDF download with Scidownl
    pdf_download()

    # PDF quality XGBoost-filter
    pre_extraction_filter(seed_rf, threshold, pdf_input_folder, output_folder_images, output_folder_score, training_file, scoring_file, output_file)

    # Extraction
    extractor(
        input_folder_images,
        output_folder,
        paper_list_path,
        pdf_folder,
        extracted_data_file,
    )


def main():
    # Define Crossref Keywords
    crossref_keyword = "'copolymerization' AND 'reactivity ratio'"  # Note that this prompt was created in the
    # wrong way and collect all paper with reactivity and/or copolymerization in title and abstract

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

    # XGBoost filter
    seed_rf = 22
    threshold = 0.7  # threshold to define precision limit for the filter
    pdf_input_folder = "../obtain_data/output/PDF"
    output_folder_images = "./output/processed_images"
    output_folder_score = "./output/model_output_score"

    training_file = (
        "../../data_extraction/data_extraction_GPT-4o/output/copol_paper_list.json"
    )
    scoring_file = "../../../data_extraction/obtain_data/output/selected_200_papers.json"
    output_file = "../../../data_extraction/data_extraction_GPT-4o/output/paper_list.json"

    # Input and output folders for data extraction
    input_folder_images = "./processed_images"
    output_folder = "./model_output_GPT4-o"
    paper_list_path = (
        "../../data_extraction/data_extraction_GPT-4o/output/paper_list.json"
    )
    pdf_folder = "../obtain_data/output/PDF"
    extracted_data_file = (
        "../../data_extraction/comparison_of_models/extracted_data.json"
    )

    # run obtain data pipline
    obtain_data(
        crossref_keyword,
        keywords,
        score_limit,
        number_of_selected_papers,
        input_folder_images,
        output_folder,
        paper_list_path,
        pdf_folder,
        extracted_data_file,
        seed_rf,
        threshold,
        pdf_input_folder,
        output_folder_images,
        output_folder_score,
        training_file,
        scoring_file,
        output_file,
    )


if __name__ == "__main__":
    main()
