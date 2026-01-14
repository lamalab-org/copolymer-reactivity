"""
Paper Similarity Finder for Copolymerization Predictions

This module finds the most similar papers from the dataset based on:
- Monomer similarity (Tanimoto similarity of fingerprints)
- Solvent similarity (Tanimoto similarity)
- Temperature proximity
- Method/Polytype embedding distance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def calculate_tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> float:
    """
    Calculate Tanimoto similarity between two molecules using Morgan fingerprints.
    
    Args:
        smiles1: SMILES string of first molecule
        smiles2: SMILES string of second molecule
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        
    Returns:
        Tanimoto similarity score (0-1)
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def temperature_similarity(temp1: float, temp2: float, max_diff: float = 50.0) -> float:
    """
    Calculate temperature similarity (1.0 = same, 0.0 = very different).
    
    Args:
        temp1: First temperature
        temp2: Second temperature
        max_diff: Maximum temperature difference for normalization (default 50°C)
        
    Returns:
        Similarity score (0-1)
    """
    diff = abs(temp1 - temp2)
    # Exponential decay: similar temps get high score, different temps get low score
    return np.exp(-diff / max_diff)


def embedding_similarity(emb1: Tuple[float, float], emb2: Tuple[float, float]) -> float:
    """
    Calculate similarity between 2D embeddings using Euclidean distance.
    
    Args:
        emb1: First embedding (pca_1, pca_2)
        emb2: Second embedding (pca_1, pca_2)
        
    Returns:
        Similarity score (0-1), where 1 = identical, 0 = very different
    """
    dist = np.sqrt((emb1[0] - emb2[0])**2 + (emb1[1] - emb2[1])**2)
    # Convert distance to similarity using exponential decay
    # Typical embedding distances are 0-20, so we normalize by 10
    return np.exp(-dist / 10.0)


def calculate_overall_similarity(
    query_mon1: str,
    query_mon2: str,
    query_solvent: str,
    query_temp: float,
    query_method_emb: Tuple[float, float],
    query_polytype_emb: Tuple[float, float],
    row: pd.Series,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall similarity between query and a dataset row.
    
    Args:
        query_mon1: Query monomer 1 SMILES
        query_mon2: Query monomer 2 SMILES
        query_solvent: Query solvent SMILES
        query_temp: Query temperature
        query_method_emb: Query method embedding (pca_1, pca_2)
        query_polytype_emb: Query polytype embedding (pca_1, pca_2)
        row: DataFrame row from dataset
        weights: Optional weights for different similarity components
        
    Returns:
        Tuple of (overall_similarity, component_scores)
    """
    if weights is None:
        weights = {
            'monomer': 0.40,  # 40% weight on monomers (most important)
            'solvent': 0.25,  # 25% weight on solvent
            'temperature': 0.15,  # 15% weight on temperature
            'method': 0.10,  # 10% weight on method
            'polytype': 0.10  # 10% weight on polytype
        }
    
    # Calculate monomer similarity (both combinations: 1-1,2-2 and 1-2,2-1)
    mon_sim_direct = (
        calculate_tanimoto_similarity(query_mon1, row['monomer1_smiles']) +
        calculate_tanimoto_similarity(query_mon2, row['monomer2_smiles'])
    ) / 2.0
    
    mon_sim_flipped = (
        calculate_tanimoto_similarity(query_mon1, row['monomer2_smiles']) +
        calculate_tanimoto_similarity(query_mon2, row['monomer1_smiles'])
    ) / 2.0
    
    mon_similarity = max(mon_sim_direct, mon_sim_flipped)
    
    # Calculate solvent similarity
    solvent_similarity = calculate_tanimoto_similarity(query_solvent, row['solvent_smiles'])
    
    # Calculate temperature similarity
    temp_similarity = temperature_similarity(query_temp, row['temperature'])
    
    # Calculate method embedding similarity
    row_method_emb = (row['method_emb_1'], row['method_emb_2'])
    method_similarity = embedding_similarity(query_method_emb, row_method_emb)
    
    # Calculate polytype embedding similarity
    row_polytype_emb = (row['polytype_emb_1'], row['polytype_emb_2'])
    polytype_similarity = embedding_similarity(query_polytype_emb, row_polytype_emb)
    
    # Calculate weighted overall similarity
    overall = (
        weights['monomer'] * mon_similarity +
        weights['solvent'] * solvent_similarity +
        weights['temperature'] * temp_similarity +
        weights['method'] * method_similarity +
        weights['polytype'] * polytype_similarity
    )
    
    components = {
        'monomer_similarity': mon_similarity,
        'solvent_similarity': solvent_similarity,
        'temperature_similarity': temp_similarity,
        'method_similarity': method_similarity,
        'polytype_similarity': polytype_similarity
    }
    
    return overall, components


def find_similar_papers(
    dataset: pd.DataFrame,
    monomer1_smiles: str,
    monomer2_smiles: str,
    solvent_smiles: str,
    temperature: float,
    method_emb: Tuple[float, float],
    polytype_emb: Tuple[float, float],
    top_n: int = 10
) -> List[Dict]:
    """
    Find the most similar papers from the dataset.
    
    Args:
        dataset: DataFrame with paper data
        monomer1_smiles: Query monomer 1 SMILES
        monomer2_smiles: Query monomer 2 SMILES
        solvent_smiles: Query solvent SMILES
        temperature: Query temperature
        method_emb: Query method embedding (pca_1, pca_2)
        polytype_emb: Query polytype embedding (pca_1, pca_2)
        top_n: Number of similar papers to return
        
    Returns:
        List of similar papers with similarity scores
    """
    # Calculate similarity for each row
    similarities = []
    
    for idx, row in dataset.iterrows():
        overall_sim, components = calculate_overall_similarity(
            monomer1_smiles,
            monomer2_smiles,
            solvent_smiles,
            temperature,
            method_emb,
            polytype_emb,
            row
        )
        
        similarities.append({
            'index': idx,
            'doi': row.get('original_source', 'Unknown'),
            'paper_name': row.get('PDF_name', 'Unknown'),
            'overall_similarity': overall_sim,
            **components,
            'monomer1': row['monomer1_name'],
            'monomer2': row['monomer2_name'],
            'solvent': row.get('solvent', 'Unknown'),
            'temperature': row['temperature'],
            'r_product': row.get('r1r2', None)
        })
    
    # Sort by overall similarity
    similarities.sort(key=lambda x: x['overall_similarity'], reverse=True)
    
    # Group by paper (DOI) and keep only the best match per paper
    seen_papers = set()
    unique_papers = []
    
    for sim in similarities:
        doi = sim['doi']
        if doi not in seen_papers and doi != 'Unknown':
            seen_papers.add(doi)
            unique_papers.append(sim)
            
            if len(unique_papers) >= top_n:
                break
    
    return unique_papers


def format_similarity_output(similar_papers: List[Dict]) -> List[Dict]:
    """
    Format similar papers for API output.
    
    Args:
        similar_papers: List of similar papers with scores
        
    Returns:
        Formatted list for API response
    """
    formatted = []
    
    for i, paper in enumerate(similar_papers, 1):
        # Create match quality label
        overall = paper['overall_similarity']
        if overall >= 0.9:
            match_quality = "Excellent match"
        elif overall >= 0.75:
            match_quality = "Good match"
        elif overall >= 0.60:
            match_quality = "Moderate match"
        elif overall >= 0.45:
            match_quality = "Weak match"
        else:
            match_quality = "Poor match"
        
        formatted.append({
            'rank': i,
            'doi': paper['doi'],
            'paper_name': paper['paper_name'],
            'similarity_score': round(paper['overall_similarity'], 3),
            'match_quality': match_quality,
            'details': {
                'monomer_similarity': round(paper['monomer_similarity'], 3),
                'solvent_similarity': round(paper['solvent_similarity'], 3),
                'temperature_similarity': round(paper['temperature_similarity'], 3),
                'method_similarity': round(paper['method_similarity'], 3),
                'polytype_similarity': round(paper['polytype_similarity'], 3)
            },
            'reaction_info': {
                'monomer1': paper['monomer1'],
                'monomer2': paper['monomer2'],
                'solvent': paper['solvent'],
                'temperature': paper['temperature'],
                'r_product': paper['r_product']
            }
        })
    
    return formatted

