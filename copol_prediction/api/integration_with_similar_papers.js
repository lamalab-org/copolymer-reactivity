// Example: Integration with Similar Papers Feature
// This shows how to use the new similar papers feature in your web application

const Molecule = OCL.Molecule;
const url = 'https://polygalaceous-guadalupe-gonangial.ngrok-free.dev';
// const url = 'http://localhost:8000';  // For local testing

const prefs = API.getData('preferences').resurrect();
const mf1 = API.getData('molfile1').resurrect();
const mf2 = API.getData('molfile2').resurrect();
const mfS = API.getData('molfileSolvent').resurrect();

const m1 = Molecule.fromMolfile(mf1).toSmiles();
const m2 = Molecule.fromMolfile(mf2).toSmiles();
const s = Molecule.fromMolfile(mfS).toSmiles();

fetch(url + '/preprocess_all', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        monomer1_smiles: m1,
        monomer2_smiles: m2,
        solvent_smiles: s,
        method: prefs.polymerisationMethod || 'solvent',
        polytype: prefs.polymerisationType || 'free radical',
        temperature: Number(prefs.temperature) || 60.0
    })
})
.then(function(r) { return r.json(); })
.then(function(preprocessed) {
    if (!preprocessed.success) {
        throw new Error(preprocessed.error || 'Preprocessing failed');
    }
    
    // Store similar papers if available
    if (preprocessed.similar_papers) {
        API.createData('similar_papers', preprocessed.similar_papers);
        console.log('Found ' + preprocessed.similar_papers.length + ' similar papers');
        
        // Show summary of best matches
        var bestMatches = preprocessed.similar_papers.slice(0, 3);
        console.log('Top 3 matches:');
        bestMatches.forEach(function(paper) {
            console.log('- ' + paper.match_quality + ' (Score: ' + paper.similarity_score.toFixed(3) + ')');
            console.log('  ' + paper.doi);
        });
    }
    
    // Make prediction with the preprocessed features
    return fetch(url + '/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: preprocessed.features })
    });
})
.then(function(r) { return r.json(); })
.then(function(result) {
    // Store prediction result
    API.createData('result', result);
    
    // Optionally: Display similar papers in UI
    var similarPapers = API.getData('similar_papers');
    if (similarPapers && similarPapers.length > 0) {
        // Format similar papers for display
        var papersHTML = '<h3>📚 Similar Papers from Dataset</h3><ul>';
        similarPapers.forEach(function(paper) {
            papersHTML += '<li>';
            papersHTML += '<strong>' + paper.rank + '. ' + paper.match_quality + '</strong> ';
            papersHTML += '(Similarity: ' + (paper.similarity_score * 100).toFixed(1) + '%)';
            papersHTML += '<br>DOI: <a href="' + paper.doi + '" target="_blank">' + paper.doi + '</a>';
            papersHTML += '<br>Monomers: ' + paper.reaction_info.monomer1 + ' + ' + paper.reaction_info.monomer2;
            papersHTML += '<br>Conditions: ' + paper.reaction_info.solvent + ', ' + paper.reaction_info.temperature + '°C';
            if (paper.reaction_info.r_product !== null) {
                papersHTML += '<br>r-product: ' + paper.reaction_info.r_product.toFixed(3);
            }
            papersHTML += '</li>';
        });
        papersHTML += '</ul>';
        
        API.createData('similar_papers_html', papersHTML);
    }
})
.catch(function(err) {
    API.createData('result', { error: err.message, success: false });
    console.error('Error:', err.message);
});


// Alternative: Display similar papers in a formatted table
function displaySimilarPapersTable(papers) {
    var table = '<table border="1" style="width:100%; border-collapse: collapse;">';
    table += '<thead><tr>';
    table += '<th>Rank</th>';
    table += '<th>Match Quality</th>';
    table += '<th>Score</th>';
    table += '<th>DOI</th>';
    table += '<th>Monomers</th>';
    table += '<th>Conditions</th>';
    table += '<th>r-product</th>';
    table += '</tr></thead><tbody>';
    
    papers.forEach(function(paper) {
        table += '<tr>';
        table += '<td>' + paper.rank + '</td>';
        table += '<td>' + paper.match_quality + '</td>';
        table += '<td>' + (paper.similarity_score * 100).toFixed(1) + '%</td>';
        table += '<td><a href="' + paper.doi + '" target="_blank">Link</a></td>';
        table += '<td>' + paper.reaction_info.monomer1 + ' + ' + paper.reaction_info.monomer2 + '</td>';
        table += '<td>' + paper.reaction_info.solvent + ', ' + paper.reaction_info.temperature + '°C</td>';
        table += '<td>' + (paper.reaction_info.r_product !== null ? 
                           paper.reaction_info.r_product.toFixed(3) : 'N/A') + '</td>';
        table += '</tr>';
    });
    
    table += '</tbody></table>';
    
    return table;
}


// Alternative: Get just the DOIs for further processing
function extractDOIs(papers) {
    return papers.map(function(paper) {
        return paper.doi;
    });
}


// Alternative: Filter papers by match quality
function filterByMatchQuality(papers, minQuality) {
    var qualityLevels = {
        'Excellent match': 5,
        'Good match': 4,
        'Moderate match': 3,
        'Weak match': 2,
        'Poor match': 1
    };
    
    var minLevel = qualityLevels[minQuality] || 0;
    
    return papers.filter(function(paper) {
        var level = qualityLevels[paper.match_quality] || 0;
        return level >= minLevel;
    });
}


// Alternative: Group papers by similarity score ranges
function groupBySimilarity(papers) {
    var groups = {
        excellent: [],
        good: [],
        moderate: [],
        weak: [],
        poor: []
    };
    
    papers.forEach(function(paper) {
        var score = paper.similarity_score;
        if (score >= 0.9) groups.excellent.push(paper);
        else if (score >= 0.75) groups.good.push(paper);
        else if (score >= 0.60) groups.moderate.push(paper);
        else if (score >= 0.45) groups.weak.push(paper);
        else groups.poor.push(paper);
    });
    
    return groups;
}

