/**
 * TruthScan AI - Client Side Hardening & UI Control Pipeline
 */

// Cross-Site Scripting (XSS) Mitigation Encoder
function escapeHTML(str) {
    if (!str) return '';
    return str.replace(/[&<>'"]/g, 
        tag => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            "'": '&#39;',
            '"': '&quot;'
        }[tag] || tag)
    );
}

function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
}

function switchTab(type) {
    const isText = type === 'text';
    document.getElementById('textInputDiv').style.display = isText ? 'block' : 'none';
    document.getElementById('urlInputDiv').style.display = isText ? 'none' : 'block';
    
    document.getElementById('tabText').style.background = isText ? 'var(--accent)' : 'var(--accent-soft)';
    document.getElementById('tabText').style.color = isText ? '#fff' : 'var(--accent)';
    document.getElementById('tabUrl').style.background = isText ? 'var(--accent-soft)' : 'var(--accent)';
    document.getElementById('tabUrl').style.color = isText ? 'var(--accent)' : '#fff';
}

async function fetchFromUrl() {
    const urlVal = document.getElementById('urlInput').value.trim();
    const errorMsg = document.getElementById('errorMsg');
    if(!urlVal) {
        errorMsg.textContent = "Please input a valid target link.";
        errorMsg.classList.add('visible');
        return;
    }
    errorMsg.classList.remove('visible');
    
    try {
        const res = await fetch('/fetch-url', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            },
            body: JSON.stringify({ url: urlVal })
        });
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || "Extraction failed.");
        
        document.getElementById('fetchedText').value = data.text;
    } catch(err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.add('visible');
    }
}

async function analyzeNews() {
    const activeTab = document.getElementById('textInputDiv').style.display !== 'none' ? 'text' : 'url';
    const textVal = document.getElementById('newsInput').value.trim();
    const urlVal = document.getElementById('urlInput').value.trim();
    const modelVal = document.getElementById('modelSelect').value;
    
    const payload = activeTab === 'text' ? { text: textVal, model: modelVal } : { url: urlVal, model: modelVal };
    
    const errorMsg = document.getElementById('errorMsg');
    const overlay = document.getElementById('loadingOverlay');
    const card = document.getElementById('resultCard');
    
    errorMsg.classList.remove('visible');
    card.style.display = 'none';
    overlay.classList.add('visible');
    
    try {
        const res = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRF-Token': getCsrfToken()
            },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if(!res.ok) throw new Error(data.error || "Analysis pipeline error.");
        
        // Update UX parameters cleanly with XSS protection
        const banner = document.getElementById('verdictBanner');
        banner.className = 'verdict-banner ' + data.ml_label.toLowerCase();
        
        document.getElementById('verdictIcon').textContent = data.ml_label === 'REAL' ? '✅' : '🚨';
        document.getElementById('verdictLabel').textContent = data.ml_label;
        document.getElementById('confidenceNum').textContent = data.ml_confidence + '%';
        
        document.getElementById('confBarFill').style.width = data.ml_confidence + '%';
        document.getElementById('confBarPct').textContent = data.ml_confidence + '%';
        
        document.getElementById('credBarTruth').style.width = data.truth_percent + '%';
        document.getElementById('credBarFake').style.width = data.fake_percent + '%';
        document.getElementById('credTruthPct').textContent = data.truth_percent + '%';
        document.getElementById('credFakePct').textContent = data.fake_percent + '%';
        
        // Parsing layout details directly out of lines safely
        let summary = "No distinct analysis compiled.";
        let rec = "Verify with local trusted platforms.";
        let flags = [];
        
        if (data.llm_analysis) {
            const lines = data.llm_analysis.split('\n');
            let captureFlags = false;
            lines.forEach(l => {
                if(l.toUpperCase().startsWith('SUMMARY:')) summary = l.split(':')[1].trim();
                if(l.toUpperCase().startsWith('RECOMMENDATION:')) rec = l.split(':')[1].trim();
                if(l.toUpperCase().startsWith('RED FLAGS:')) { captureFlags = true; return; }
                if(captureFlags && l.trim().startsWith('-')) {
                    flags.push(l.trim().substring(1).trim());
                } else if(captureFlags && l.trim() && !l.trim().startsWith('-')) {
                    captureFlags = false;
                }
            });
        }
        
        document.getElementById('summaryText').textContent = summary;
        document.getElementById('recommendationText').textContent = rec;
        
        const listContainer = document.getElementById('redFlagsList');
        listContainer.innerHTML = '';
        if(flags.length === 0) flags = ["No critical systemic abnormalities flagged."];
        flags.forEach(f => {
            const li = document.createElement('li');
            li.textContent = "• " + f;
            listContainer.appendChild(li);
        });
        
        document.getElementById('mlBadge').textContent = "🤖 Reference Model: " + data.model_used;
        card.style.display = 'block';
        loadHistory();
    } catch(err) {
        errorMsg.textContent = err.message;
        errorMsg.classList.add('visible');
    } finally {
        overlay.classList.remove('visible');
    }
}

async function loadHistory() {
    try {
        const res = await fetch('/history');
        const data = await res.json();
        const tbody = document.getElementById('historyTableBody');
        tbody.innerHTML = '';
        
        if(!data || data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;padding:10px;">No verification metrics cached.</td></tr>';
            return;
        }
        
        data.forEach(row => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid rgba(56,139,253,0.1)';
            
            // Generate table data securely using programmatic elements
            const tdHead = document.createElement('td');
            tdHead.style.padding = '10px';
            tdHead.textContent = row.headline ? row.headline.substring(0, 50) + '...' : 'Unknown';
            
            const tdPred = document.createElement('td');
            tdPred.style.padding = '10px';
            tdPred.style.fontWeight = 'bold';
            tdPred.style.color = row.prediction === 'REAL' ? 'var(--real-color)' : 'var(--fake-color)';
            tdPred.textContent = row.prediction;
            
            const tdConf = document.createElement('td');
            tdConf.style.padding = '10px';
            tdConf.textContent = row.confidence + '%';
            
            const tdDate = document.createElement('td');
            tdDate.style.padding = '10px';
            tdDate.textContent = row.date || '—';
            
            tr.appendChild(tdHead);
            tr.appendChild(tdPred);
            tr.appendChild(tdConf);
            tr.appendChild(tdDate);
            tbody.appendChild(tr);
        });
    } catch(e) {}
}

async function clearHistory() {
    if(!confirm("Purge complete verification table history logs?")) return;
    try {
        await fetch('/clear-history', {
            method: 'POST',
            headers: { 'X-CSRF-Token': getCsrfToken() }
        });
        loadHistory();
    } catch(e) {}
}

function copyResult() {
    const summary = document.getElementById('summaryText').textContent;
    const verdict = document.getElementById('verdictLabel').textContent;
    const text = `[TruthScan AI Verdict: ${verdict}]\nSummary: ${summary}`;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyBtn');
        btn.textContent = "✅ Copied!";
        setTimeout(() => { btn.textContent = "📋 Copy Result"; }, 2000);
    });
}

window.addEventListener('DOMContentLoaded', loadHistory);
