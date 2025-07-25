from copolextractor.preextractionfilter.pre_extraction_filter import main as pre_extraction_filter
from copolextractor.PDF_download import main as pdf_download
from copolextractor.predownloadfilter.pre_download_filter import main as pre_download_filter
from copolextractor.crossref_search import main as crossref_search
from copolextractor.extraction_with_GPT_PDF import main as extractor
from copolextractor.data_into_csv import main as save_data


def obtain_data(
    crossref_keyword,
    output_file_crossref_search,
    crossref_metadata_output_file,
    keywords_filter,
    output_file_pre_download_filter,
    score_limit,
    number_of_selected_papers,
    input_folder_images,
    output_folder_data_extraction,
    output_file_data_extraction,
    seed_xgboost_model,
    threshold_xgboost_model,
    pdf_folder,
    output_folder_images,
    output_folder_LLM_score,
    training_file_xgboost_model,
    output_file_xgboost_filter,
    key_embedding_filter,
    values_embedding_filter,
    scoring_file_embedding_filter,
existing_doi_csv
):
    # crossref search for relevant paper
    #crossref_search(crossref_keyword, output_file_crossref_search, crossref_metadata_output_file)

    # metadata filter with keywords and embeddings
    #pre_download_filter(keywords_filter, score_limit, number_of_selected_papers, crossref_metadata_output_file, output_file_pre_download_filter, key_embedding_filter, values_embedding_filter, scoring_file_embedding_filter, existing_doi_csv)

    # PDF download with Scidownl
    #pdf_download(output_file_pre_download_filter, pdf_folder)

    # PDF quality XGBoost-filter
    #pre_extraction_filter(seed_xgboost_model, threshold_xgboost_model, pdf_folder, output_folder_images, output_folder_LLM_score, training_file_xgboost_model, output_file_pre_download_filter, output_file_xgboost_filter)

    # Extraction
    extractor(
        input_folder_images,
        output_folder_data_extraction,
        output_file_xgboost_filter,
        pdf_folder,
        output_file_data_extraction,
    )

    # Save data in database
    save_data(output_folder_data_extraction)


def main():
    # Crossref search
    crossref_keyword = "'copolymerization' AND 'reactivity ratio'"  # Note that this prompt was created in the
    # wrong way and collect all paper with reactivity and/or copolymerization in title and abstract
    output_file_crossref_search = (
        "./obtain_data/output/crossref_search.json"
    )
    crossref_metadata_output_file = (
        "./obtain_data/output/collected_doi_metadata.json"
    )

    existing_doi_csv = 'extracted_reactions.csv'
    # Keywords and weights for pre download scoring
    keywords_filter = {
        "copolymerization": 10,
        "polymerization": 5,
        "monomers": 5,
        "copolymers": 5,
        "ratios": 20,
        "reactivity ratios": 40,
    }

    # Embedding filter
    score_limit = 65  # Minimum score for embedding generation
    number_of_selected_papers = 2000  # Number of nearest papers to select
    output_file_pre_download_filter = "./output/selected_papers.json"
    key_embedding_filter = "polymerization_type"
    values_embedding_filter = [
        'free radical', 'Free radical', 'Free Radical',
        'atom transfer radical polymerization',
        'atom-transfer radical polymerization',
        'nickel-mediated radical', 'bulk',
        'Radical', 'radical',
        'controlled radical',
        'controlled/living radical',
        'conventional radical polymerization',
        'reversible deactivation radical polymerization',
        'reversible addition-fragmentation chain transfer polymerization',
        'reversible addition-fragmentation chain transfer',
        'Homogeneous Radical',
        'Radiation-induced', 'radiation-induced',
        'Radiation-Initiated',
        'photo-induced polymerization',
        'photopolymerization',
        'thermal polymerization',
        'thermal',
        'group transfer polymerization',
        'Emulsion',
        'Homogeneous Radical',
        'semicontinuous emulsion',
        'emulsion'
    ]
    scoring_file_embedding_filter = "extracted_reactions.csv"

    # PDF download
    pdf_folder = "./output/PDF"  # the PDFs get downloaded form the SciHub corpus.
    # People should be aware of copyright law before using this

    # XGBoost filter
    seed_xgboost_model = 22
    threshold_xgboost_model = 0.7  # threshold to define precision limit for the filter
    # LLM scoring PDF as parameters for XGBoost model
    output_folder_images = "./output/processed_images"
    output_folder_LLM_score = "./output/model_output_score"

    training_file_xgboost_model = (
        "output/copol_database/copol_paper_list.json"
    )
    output_file_xgboost_filter = "output/paper_list.json"

    # Data extraction of filtered paper
    input_folder_images = "./processed_images"
    output_folder_data_extraction = "./model_output_GPT4-o"
    output_file_data_extraction = (
        "./comparison_of_models/extracted_data.json"
    )

    # run obtain data pipline
    obtain_data(
        crossref_keyword,
        output_file_crossref_search,
        crossref_metadata_output_file,
        keywords_filter,
        output_file_pre_download_filter,
        score_limit,
        number_of_selected_papers,
        input_folder_images,
        output_folder_data_extraction,
        output_file_data_extraction,
        seed_xgboost_model,
        threshold_xgboost_model,
        pdf_folder,
        output_folder_images,
        output_folder_LLM_score,
        training_file_xgboost_model,
        output_file_xgboost_filter,
        key_embedding_filter,
        values_embedding_filter,
        scoring_file_embedding_filter,
        existing_doi_csv
    )


if __name__ == "__main__":
    main()
