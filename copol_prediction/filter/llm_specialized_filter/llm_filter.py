import hashlib  # NEW
import json
import os
import re
from time import sleep

import pandas as pd
import requests
from crossref.restful import Works
from openai import OpenAI

works = Works()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------- Local DB helpers ----------------
def load_local_db(json_path):
    if not json_path:
        return []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"⚠️ Could not load local DB: {e}")
        return []


def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()) if isinstance(s, str) else ""


def normalize_title(s: str) -> str:
    s = norm_space(s).lower()
    s = s.strip(" .,:;-/\\()[]{}\"'|")
    return s


def normalize_doi(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip()
    lowered = s.lower()
    prefixes = ("https://doi.org/", "http://doi.org/", "doi:", "doi ")
    for p in prefixes:
        if lowered.startswith(p):
            s = s[len(p) :]
            break
    return s.strip().lower()


def normalize_pdf_name(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    name = os.path.basename(s)
    name = re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name)  # drop extension
    name = name.replace("_", " ").replace("-", " ")
    name = norm_space(name).lower()
    return name


def build_local_index(rows):
    doi_index = {}
    title_index = {}
    for row in rows:
        doi = normalize_doi(row.get("DOI", ""))
        if doi and doi not in doi_index:
            doi_index[doi] = row
        title = normalize_title(row.get("Title", ""))
        if title and title not in title_index:
            title_index[title] = row
    return {"doi": doi_index, "title": title_index}


def lookup_local(original_source, fallback_title, local_index):
    """
    Returns (title, abstract, classification_or_none)
    """
    if not local_index:
        return "", "", None

    doi_idx = local_index.get("doi", {})
    title_idx = local_index.get("title", {})

    # 1) DOI from original_source
    norm_doi = normalize_doi(original_source)
    if norm_doi and norm_doi in doi_idx:
        row = doi_idx[norm_doi]
        return row.get("Title", "") or "", row.get("Abstract", "") or "", row.get("Classification")

    # 2) Try PDF_name
    pdf_norm = normalize_pdf_name(fallback_title)
    if pdf_norm:
        # 2a) Try if PDF encodes a DOI without slashes/spaces
        pdf_as_doi_try = pdf_norm.replace(" ", "").replace("\\", "").replace("/", "")
        for d, row in doi_idx.items():
            d_slashless = d.replace("/", "")
            if pdf_as_doi_try == d_slashless:
                return (
                    row.get("Title", "") or "",
                    row.get("Abstract", "") or "",
                    row.get("Classification"),
                )
        # 2b) Title match
        if pdf_norm in title_idx:
            row = title_idx[pdf_norm]
            return (
                row.get("Title", "") or "",
                row.get("Abstract", "") or "",
                row.get("Classification"),
            )

    # 3) fallback_title as real title
    norm_fb_title = normalize_title(fallback_title)
    if norm_fb_title and norm_fb_title in title_idx:
        row = title_idx[norm_fb_title]
        return row.get("Title", "") or "", row.get("Abstract", "") or "", row.get("Classification")

    return "", "", None


# ---------------- Cache helpers (NEW) ----------------
def load_cache(cache_path):
    if not cache_path:
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"⚠️ Could not load cache: {e}")
        return {}


def save_cache(cache_path, cache):
    if not cache_path:
        return
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")


def compute_cache_key(original_source, title, abstract):
    """
    Prefer DOI-based key; else hash of (normalized title, abstract).
    """
    doi = normalize_doi(original_source)
    if doi:
        return f"doi::{doi}"
    t = normalize_title(title or "")
    a = (abstract or "").strip()
    h = hashlib.sha1((t + "||" + a).encode("utf-8")).hexdigest()
    return f"ta::{h}"


# ---------------- Crossref (with local pre-check) ----------------
def get_title_abstract(original_source, fallback_title="", local_index=None):
    """
    Returns (title, abstract) after:
    0) Local DB
    1) Crossref DOI
    2) Crossref title search
    """

    def is_valid(s):
        return isinstance(s, str) and s.strip() != ""

    # 0) Local DB
    if local_index:
        t, a, _ = lookup_local(original_source, fallback_title, local_index)
        if t:
            return t, a

    # Normalize DOI for Crossref
    source = original_source.strip() if is_valid(original_source) else ""
    if source.startswith("https://doi.org/"):
        source = source.replace("https://doi.org/", "")
    elif source.startswith("http://doi.org/"):
        source = source.replace("http://doi.org/", "")

    # 1) DOI lookup
    if source.startswith("10."):
        try:
            url = f"https://api.crossref.org/works/{source}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json().get("message", {})
                title = data.get("title", [""])[0]
                abstract = data.get("abstract", "")
                return title, abstract
            else:
                print(f"Crossref returned status {response.status_code} for DOI {source}")
        except Exception as e:
            print(f"Crossref DOI error for '{source}': {e}")

    # 2) Fallback → Crossref title search
    title_query = fallback_title.strip() if is_valid(fallback_title) else ""
    if title_query:
        try:
            result = next(works.query(title=title_query), None)
            if result:
                title = result.get("title", [""])[0]
                abstract = result.get("abstract", "")
                return title, abstract
            else:
                print(f"Crossref fallback title search failed for: {title_query}")
        except Exception as e:
            print(f"Crossref title query error for '{title_query}': {e}")

    return "", ""


def classify_paper(title, abstract):
    """
    LLM classification based on title and abstract.
    """
    if not title:
        return "error"

    system_prompt = (
        "You are an expert in polymer chemistry. "
        "Your task is to classify whether a research paper on copolymerization is 'normal', 'specialized', or 'unclear'.\n\n"
        "A 'normal' paper investigates common variables such as monomer types, temperatures, solvents, and "
        "polymerization types such as free radical, cationic, anionic or also other controlled radical polymerizations "
        "(ATRP, RAFT, NMP, ...).\n"
        "In a 'specialized' paper the reactivity is also influenced by other factors such as by ligands, catalysts or "
        "other additives.\n"
        "If the classification is ambiguous or not clear from the given information, return 'unclear'. If you're not "
        "100 % also return 'unclear'\n\n"
        "Only respond with one word: normal, specialized, or unclear."
    )

    user_prompt = (
        "Classify the given paper into 'normal', 'specialized' and if the decision would be ambiguous 'unclear'. "
        "An example for an 'specialized' paper would be Title: 'Study of the state of titanium ions and the composition "
        "of the active component in titanium‐magnesium catalysts for ethylene polymerization' or Title: 'Effects of "
        "complexing agents in radical copolymerization'. An example of a 'normal' paper would be Title: 'Determination "
        "of copolymerization parameters of methyl methacrylate with dodecyl methacrylate by means of FTIR spectroscopy' "
        "or Title: 'Preparation and thermal properties of polyquinazolone containing 4‐substituted phenyl groups on the "
        "quinazolone ring'.\n\n "
        f"The Article has the following Title: {title}\n\nand Abstract: {abstract}"
    )

    print(f"User prompt: {user_prompt}")
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        content = response.choices[0].message.content.strip().lower()
        print(f"Classification: {content}")
        if content not in ["normal", "specialized", "unclear"]:
            return "unclear"
        return content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return "error"


# ---------------- Main loop with incremental save & caching ----------------
def classify_csv(
    input_path,
    output_path,
    local_json_path=None,
    cache_path="classification_cache.json",
    save_every=1,
    resume_from_output=True,
):
    """
    - Loads local DB once (optional).
    - Loads cache JSON (optional).
    - Writes progress to CSV every `save_every` rows (default: every row).
    - Reuses:
        * 'Classification' from local DB (if Title/Abstract come from there)
        * Cache entry keyed by DOI or (title,abstract)
        * Previous output file (if resume_from_output=True)
    """
    # Local DB
    local_rows = load_local_db(local_json_path) if local_json_path else []
    local_index = build_local_index(local_rows) if local_rows else None
    if local_index:
        print(f"✅ Local DB loaded: {len(local_rows)} records indexed.")
    else:
        print("ℹ️ No local DB or empty DB; using Crossref only (plus cache).")

    # Cache
    cache = load_cache(cache_path) if cache_path else {}
    if cache:
        print(f"✅ Cache loaded: {len(cache)} entries.")

    # Load input
    df = pd.read_csv(input_path)

    # Resume from existing output if desired
    if resume_from_output and os.path.exists(output_path):
        try:
            prev = pd.read_csv(output_path)
            if "llm_specialized_filter" in prev.columns:
                # align by index/row order; if you have a stable key column, you can merge by key instead
                to_copy = prev.get("llm_specialized_filter")
                if to_copy is not None and len(to_copy) == len(df):
                    df["llm_specialized_filter"] = to_copy
                    print("↩️ Resumed classifications from existing output where available.")
        except Exception as e:
            print(f"⚠️ Could not resume from existing output: {e}")

    # Ensure output column exists
    if "llm_specialized_filter" not in df.columns:
        df["llm_specialized_filter"] = ""

    processed = 0
    for idx, row in df.iterrows():
        # Skip if already classified (resumed)
        if str(df.at[idx, "llm_specialized_filter"]).strip() in [
            "normal",
            "specialized",
            "unclear",
            "error",
        ]:
            continue

        source = str(row.get("original_source", ""))
        fallback_title = str(row.get("PDF_name", ""))

        print(f"[{idx+1}/{len(df)}] Working on: {source or '(no source)'}")

        # Resolve title/abstract (local DB first, then Crossref)
        local_t, local_a, local_cls = ("", "", None)
        if local_index:
            local_t, local_a, local_cls = lookup_local(source, fallback_title, local_index)

        if local_t:  # local hit
            title, abstract = local_t, local_a
            if local_cls and local_cls in ["normal", "specialized", "unclear", "error"]:
                # ✅ Use stored classification from your DB
                classification = local_cls
                print("→ Using local DB classification.")
            else:
                # Check cache with DOI or title/abstract
                key = compute_cache_key(source, title, abstract)
                if key in cache:
                    classification = cache[key]["classification"]
                    print("→ Using cached classification.")
                else:
                    classification = classify_paper(title, abstract)
        else:
            # no local hit → Crossref
            title, abstract = get_title_abstract(source, fallback_title, local_index=None)
            if not title and fallback_title:
                print(f"→ Falling back to PDF_name as title: {fallback_title}")
                title = fallback_title
                abstract = ""

            # If we still lack a title, mark error
            if not title:
                classification = "error"
            else:
                # Try cache first
                key = compute_cache_key(source, title, abstract)
                if key in cache:
                    classification = cache[key]["classification"]
                    print("→ Using cached classification.")
                else:
                    classification = classify_paper(title, abstract)

        # Set and persist
        df.at[idx, "llm_specialized_filter"] = classification

        # Update cache when we have a real title (avoid caching pure empty)
        if title:
            key = compute_cache_key(source, title, abstract)
            cache[key] = {
                "title": title,
                "abstract": abstract,
                "classification": classification,
                "source": source,
            }
            save_cache(cache_path, cache)  # save after every row

        processed += 1
        # Incremental CSV save
        if processed % save_every == 0:
            df.to_csv(output_path, index=False)
            print(f"💾 Progress saved → {output_path}")

        # be nice to APIs
        sleep(1)

    # Final save
    df.to_csv(output_path, index=False)
    print(f"\n✅ All done. Output saved to {output_path}")
    if cache_path:
        save_cache(cache_path, cache)
        print(f"✅ Cache saved to {cache_path}")


if __name__ == "__main__":
    input_csv = "../paper_dataset/processed_data.csv"
    output_csv = "classified_output.csv"
    local_json = "../../data_extraction/obtain_data/output/collected_doi_metadata.json"
    cache_json = "classification_cache.json"  # persistent cache file
    classify_csv(
        input_csv,
        output_csv,
        local_json_path=local_json,
        cache_path=cache_json,
        save_every=1,  # save after each row
        resume_from_output=True,  # resume if output already exists
    )
