// Example: How to use the DOI check endpoint

const url = 'http://localhost:8000';  // oder deine ngrok URL

// Beispiel 1: DOI Check mit vollständiger URL
fetch(url + '/check_doi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        doi: 'https://doi.org/10.1016/0014-3057(84)90010-7'
    })
})
.then(function(r) { return r.json(); })
.then(function(result) {
    if (result.exists) {
        console.log('✓ DOI gefunden im Datensatz!');
        console.log('DOI:', result.doi);
        console.log('Normalisierte DOI:', result.normalized_doi);
    } else {
        console.log('✗ DOI nicht im Datensatz');
        console.log('DOI:', result.doi);
    }
})
.catch(function(err) {
    console.error('Fehler:', err.message);
});

// Beispiel 2: DOI Check nur mit DOI-Nummer (ohne URL)
fetch(url + '/check_doi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        doi: '10.1016/0014-3057(84)90010-7'
    })
})
.then(function(r) { return r.json(); })
.then(function(result) {
    console.log('Ergebnis:', result.exists ? 'JA' : 'NEIN');
})
.catch(function(err) {
    console.error('Fehler:', err.message);
});

// Beispiel 3: Integration in dein bestehendes Setup
// Wenn du die API.getData() Funktionen verwenden willst:
/*
const doi = API.getData('doi_input').resurrect();

fetch(url + '/check_doi', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ doi: doi })
})
.then(function(r) { return r.json(); })
.then(function(result) {
    // Speichere das Ergebnis für die weitere Verwendung
    API.createData('doi_check_result', result);
    
    // Zeige eine Nachricht basierend auf dem Ergebnis
    if (result.exists) {
        API.createData('message', { 
            text: 'Diese DOI ist bereits im Datensatz vorhanden!',
            type: 'success'
        });
    } else {
        API.createData('message', { 
            text: 'Diese DOI ist neu und noch nicht im Datensatz.',
            type: 'info'
        });
    }
})
.catch(function(err) {
    API.createData('doi_check_result', { 
        error: err.message, 
        exists: false 
    });
});
*/


