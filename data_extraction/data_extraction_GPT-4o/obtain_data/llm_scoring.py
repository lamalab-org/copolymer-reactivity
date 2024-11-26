import json
from openai import OpenAI


def call_openai(prompt, model="gpt-4o", temperature=0.0, **kwargs):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a scientific assistant, scoring papers based on their relevance for copolymerization "
                    "reactivity ratios. Return the output in json format."
                )
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
        **kwargs,
    )
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    message_content = completion.choices[0].message.content
    print(message_content)
    return message_content, input_tokens, output_tokens


def update_paper_scores(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        # Filter for scores over 70
        papers_to_score = [paper for paper in data if paper.get("Score", 0) == 0 and paper.get("Title") is not None]

    # Interate over all paper wit score over 70
    for paper in papers_to_score:
        print("Processing paper ", paper.get('Title'), "with abstract ", paper.get('Abstract'))
        prompt = (f"""Please score the paper titled '{paper.get('Title')}' with abstract: {paper.get('Abstract')}. 
Score the paper based on the relevance for copolymerization reactivity. 
The goal is to find the most relevant paper that includes copolymerization reactivity ratios called 'r-values'. 
The more reactivity ratios are in the paper, the more relevant the paper is. 
If more than 10 reactions with reactivity ratios could be in the paper score it a 10.
If the paper investigates solvent or temperature effects in copolymerizations score it a 9.
If the chance is high that no reactivity ratios are in the paper, score the paper a '0'. Score the papers from 0-10. 
 Return the answer in the following schema:
    "score": your score as INTEGER""")

        try:
            llm_response, _, _ = call_openai(prompt)

            # Parse the LLM response as JSON to extract the score
            response_data = json.loads(llm_response)
            paper["LLM_Score"] = response_data.get("score", 0)

            with open("updated_paper_scores.json", "w") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error scoring paper with DOI {paper['DOI']}: {e}")
            paper["LLM_Score"] = None

    with open("updated_paper_scores.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Updated data with LLM scores saved to 'updated_paper_scores.json'")


update_paper_scores("./scored_doi.json")
