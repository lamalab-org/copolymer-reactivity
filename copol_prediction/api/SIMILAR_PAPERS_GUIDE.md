# Similar Papers Feature Guide

## 📚 Overview

The API now automatically finds and returns the **10 most similar papers** from the dataset for each prediction request. This helps users understand which experimental conditions and papers are most relevant to their query.

## 🎯 How It Works

### Similarity Calculation

The similarity is calculated using a **weighted multi-component approach**:

| Component | Weight | Description |
|-----------|--------|-------------|
| **Monomers** | 40% | Tanimoto similarity of Morgan fingerprints |
| **Solvent** | 25% | Tanimoto similarity of solvent structures |
| **Temperature** | 15% | Exponential decay based on temperature difference |
| **Method** | 10% | Euclidean distance in 2D embedding space |
| **Polytype** | 10% | Euclidean distance in 2D embedding space |

### Monomer Matching

- Checks **both combinations**: (M1, M2) and (M2, M1)
- Uses the **higher similarity** of the two
- Handles flipped monomer pairs automatically

### Paper Deduplication

- Groups reactions by DOI/Paper
- Returns only the **best match** per paper
- Ensures 10 unique papers (not 10 reactions)

## 📊 API Response Format

### Example Request

```json
POST /preprocess_all
{
  "monomer1_smiles": "C=CC1=CC=CC=C1",
  "monomer2_smiles": "C=C(C)C(=O)OCCO",
  "solvent_smiles": "CCO",
  "method": "solvent",
  "polytype": "free radical",
  "temperature": 60.0
}
```

### Example Response

```json
{
  "features": {...},
  "success": true,
  "similar_papers": [
    {
      "rank": 1,
      "doi": "https://doi.org/10.1016/0014-3057(84)90010-7",
      "paper_name": "Copolymerisation of 2-Hydroxyethyl...",
      "similarity_score": 0.817,
      "match_quality": "Good match",
      "details": {
        "monomer_similarity": 0.542,
        "solvent_similarity": 1.000,
        "temperature_similarity": 1.000,
        "method_similarity": 1.000,
        "polytype_similarity": 1.000
      },
      "reaction_info": {
        "monomer1": "2-Hydroxyethyl methacrylate",
        "monomer2": "Acryloxymethylpentamethyldisiloxane",
        "solvent": "ethanol",
        "temperature": 60.0,
        "r_product": 0.473
      }
    },
    ...9 more papers...
  ]
}
```

## 🏆 Match Quality Labels

| Score Range | Label | Meaning |
|-------------|-------|---------|
| ≥ 0.90 | **Excellent match** | Near-identical experimental conditions |
| 0.75 - 0.89 | **Good match** | Very similar conditions, highly relevant |
| 0.60 - 0.74 | **Moderate match** | Reasonably similar, good reference |
| 0.45 - 0.59 | **Weak match** | Some similarities, use with caution |
| < 0.45 | **Poor match** | Few similarities, limited relevance |

## 💻 Usage Examples

### Python

```python
import requests

API_URL = "http://localhost:8000"

response = requests.post(
    f"{API_URL}/preprocess_all",
    json={
        "monomer1_smiles": "C=CC1=CC=CC=C1",
        "monomer2_smiles": "C=C(C)C(=O)OCCO",
        "solvent_smiles": "CCO",
        "method": "solvent",
        "polytype": "free radical",
        "temperature": 60.0
    }
)

result = response.json()

if result['success'] and result.get('similar_papers'):
    print(f"Found {len(result['similar_papers'])} similar papers:")
    
    for paper in result['similar_papers']:
        print(f"\n{paper['rank']}. {paper['match_quality']}")
        print(f"   Similarity: {paper['similarity_score']:.3f}")
        print(f"   DOI: {paper['doi']}")
        print(f"   Monomers: {paper['reaction_info']['monomer1']} + "
              f"{paper['reaction_info']['monomer2']}")
```

### JavaScript

```javascript
const response = await fetch(url + '/preprocess_all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        monomer1_smiles: 'C=CC1=CC=CC=C1',
        monomer2_smiles: 'C=C(C)C(=O)OCCO',
        solvent_smiles: 'CCO',
        method: 'solvent',
        polytype: 'free radical',
        temperature: 60.0
    })
});

const result = await response.json();

if (result.success && result.similar_papers) {
    console.log(`Found ${result.similar_papers.length} similar papers`);
    
    result.similar_papers.forEach(paper => {
        console.log(`${paper.rank}. ${paper.match_quality} (${paper.similarity_score})`);
        console.log(`   DOI: ${paper.doi}`);
    });
}
```

## 🔍 Interpreting Results

### High Similarity (>0.75)
- **Very relevant** experimental reference
- Similar monomers, conditions, and methods
- Good baseline for expected results

### Medium Similarity (0.60-0.75)
- **Relevant** but not identical conditions
- May provide useful insights
- Consider differences in experimental setup

### Low Similarity (<0.60)
- **Limited relevance**
- Use as general reference only
- Expect different results

## 📈 Use Cases

### 1. Literature Search
Find related papers without manual searching

### 2. Method Validation
Compare your conditions with published work

### 3. Result Prediction
See what r-products were obtained under similar conditions

### 4. Experimental Design
Learn from similar experimental setups

### 5. Paper Discovery
Discover relevant papers you might have missed

## ⚙️ Technical Details

### Tanimoto Similarity
- Uses **Morgan fingerprints** (radius=2, 2048 bits)
- Range: 0.0 (completely different) to 1.0 (identical)
- Industry standard for molecular similarity

### Temperature Similarity
- Exponential decay with max_diff=50°C
- `similarity = exp(-|T1 - T2| / 50)`
- Emphasizes small temperature differences

### Embedding Similarity
- 2D PCA embeddings for method and polytype
- Euclidean distance normalized by 10
- `similarity = exp(-distance / 10)`

### Weighting Rationale
- **Monomers (40%)**: Most important - defines the reaction
- **Solvent (25%)**: Significant impact on kinetics
- **Temperature (15%)**: Affects reactivity ratios
- **Method/Polytype (20% total)**: Important but less critical

## 🎨 Customization

To adjust similarity weights, modify `paper_similarity.py`:

```python
weights = {
    'monomer': 0.40,    # Adjust these values
    'solvent': 0.25,
    'temperature': 0.15,
    'method': 0.10,
    'polytype': 0.10
}
```

Make sure they sum to 1.0!

## 📝 Notes

- **Dataset size**: Searches through 7622 reactions
- **Performance**: ~1-2 seconds per request
- **Caching**: Results are not cached (recalculated each time)
- **Fallback**: If similarity calculation fails, returns `null`

## 🐛 Troubleshooting

**No similar_papers in response:**
- Check that dataset is loaded (`/health` endpoint)
- Verify paper_similarity.py is in container
- Check Docker logs for errors

**All similarities are low:**
- Your query might be unique/novel
- Try more common monomers or conditions
- Check that SMILES strings are valid

**Performance issues:**
- Expected with large datasets
- Consider caching for repeated queries
- Use Redis for production deployments

## 🚀 Future Enhancements

Possible improvements:
- [ ] Cache similarity calculations
- [ ] Add more weighting options
- [ ] Include polymer properties in similarity
- [ ] Add minimum similarity threshold parameter
- [ ] Return confidence intervals from similar papers

