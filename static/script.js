// ── Charts ──────────────────────────────────────────────
let donutChartInstance = null;

// Fallback if Chart.js fails to load
if (typeof Chart === 'undefined') {
  console.warn('Chart.js not available, charts will be skipped');
}

function initDonutChart(fakeCount, realCount) {
  if (typeof Chart === 'undefined') {
    console.warn('Cannot initialize chart: Chart.js not loaded');
    return;
  }

  const ctx = document.getElementById('donutChart');
  if (!ctx) return;

  if (donutChartInstance) {
    donutChartInstance.data.datasets[0].data = [fakeCount, realCount];
    donutChartInstance.update();
    return;
  }

  try {
    donutChartInstance = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Fake News', 'Real News'],
        datasets: [{ 
          data: [fakeCount, realCount], 
          backgroundColor: ['#5b8def', '#1c4ed8'], 
          borderWidth: 0, 
          hoverOffset: 6 
        }]
      },
      options: {
        responsive: true, 
        maintainAspectRatio: false, 
        cutout: '65%',
        plugins: {
          legend: { display: false },
          tooltip: { 
            callbacks: { 
              label: tooltipItem => ` ${tooltipItem.label}: ${tooltipItem.raw.toLocaleString()}` 
            } 
          }
        }
      }
    });
  } catch (e) {
    console.error('Failed to initialize chart:', e);
  }
}

// Update live stats from prediction history
async function updateLiveStats() {
  try {
    const response = await fetch('/history');
    const data = await response.json();
    const fakeCount = data.filter(r => r.prediction === 'FAKE').length;
    const realCount = data.filter(r => r.prediction === 'REAL').length;
    document.getElementById('fakeCount').textContent = fakeCount;
    document.getElementById('realCount').textContent = realCount;
    initDonutChart(fakeCount || 0, realCount || 0);
  } catch (err) {
    console.error('Could not load live stats:', err);
    initDonutChart(1, 1);
  }
}
updateLiveStats();

// ── Analyzer ─────────────────────────────────────────────

// Tab switching
function switchTab(tab) {
  const textDiv = document.getElementById('textInputDiv');
  const urlDiv = document.getElementById('urlInputDiv');
  const tabText = document.getElementById('tabText');
  const tabUrl = document.getElementById('tabUrl');

  if (tab === 'text') {
    textDiv.style.display = 'block';
    urlDiv.style.display = 'none';
    tabText.style.background = 'var(--accent)';
    tabText.style.color = '#fff';
    tabText.style.border = 'none';
    tabUrl.style.background = 'var(--accent-soft)';
    tabUrl.style.color = 'var(--accent)';
    tabUrl.style.border = '1px solid var(--glass-border)';
  } else {
    textDiv.style.display = 'none';
    urlDiv.style.display = 'block';
    tabUrl.style.background = 'var(--accent)';
    tabUrl.style.color = '#fff';
    tabUrl.style.border = 'none';
    tabText.style.background = 'var(--accent-soft)';
    tabText.style.color = 'var(--accent)';
    tabText.style.border = '1px solid var(--glass-border)';
  }
}

// Fetch article from URL
async function fetchFromUrl() {
  const url = document.getElementById('urlInput').value.trim();
  const fetchedText = document.getElementById('fetchedText');
  const errorMsg = document.getElementById('errorMsg');

  errorMsg.classList.remove('visible');

  if (!url) {
    errorMsg.textContent = '⚠️ Please enter a URL first.';
    errorMsg.classList.add('visible');
    return false;
  }

  fetchedText.placeholder = '⏳ Fetching article...';
  fetchedText.value = '';

  try {
    const response = await fetch('/fetch-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await response.json();

    if (data.error) {
      errorMsg.textContent = '⚠️ ' + data.error;
      errorMsg.classList.add('visible');
      fetchedText.placeholder = 'Fetched article will appear here automatically...';
      return false;
    }

    fetchedText.value = data.text;
    fetchedText.placeholder = 'Fetched article will appear here automatically...';
    return true;
  } catch (err) {
    errorMsg.textContent = '⚠️ Could not fetch article. Try copying the text manually.';
    errorMsg.classList.add('visible');
    fetchedText.placeholder = 'Fetched article will appear here automatically...';
    return false;
  }
}

async function analyzeNews() {
  const isUrlTab = document.getElementById('urlInputDiv').style.display !== 'none';
  const btn = document.getElementById('analyzeBtn');
  const errorMsg = document.getElementById('errorMsg');
  const resultCard = document.getElementById('resultCard');
  const loadingOverlay = document.getElementById('loadingOverlay');

  errorMsg.classList.remove('visible');
  resultCard.classList.remove('visible');

  let text;
  if (isUrlTab) {
    text = document.getElementById('fetchedText').value.trim();

    if (!text) {
      const url = document.getElementById('urlInput').value.trim();
      if (!url) {
        errorMsg.textContent = '⚠️ Please enter a URL or paste an article first.';
        errorMsg.classList.add('visible');
        return;
      }
      document.getElementById('btnText').textContent = 'Fetching article...';
      btn.disabled = true;
      const fetchedOk = await fetchFromUrl();
      btn.disabled = false;
      document.getElementById('btnText').textContent = '🔍 Analyze News';
      if (!fetchedOk) return;
      text = document.getElementById('fetchedText').value.trim();
    }
  } else {
    text = document.getElementById('newsInput').value.trim();
  }

  const model = document.getElementById('modelSelect').value;

  if (!text) { 
    errorMsg.textContent = '⚠️ Please paste a news article first.'; 
    errorMsg.classList.add('visible'); 
    return; 
  }
  if (text.split(' ').length < 5) { 
    errorMsg.textContent = '⚠️ Too short — paste at least a sentence.'; 
    errorMsg.classList.add('visible'); 
    return; 
  }

  btn.disabled = true;
  document.getElementById('btnText').textContent = 'Analyzing...';
  document.getElementById('btnSpinner').style.display = 'block';
  loadingOverlay.classList.add('visible');

  try {
    const response = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, model })
    });
    const data = await response.json();
    if (data.error) { 
      errorMsg.textContent = '⚠️ ' + data.error; 
      errorMsg.classList.add('visible'); 
      return; 
    }
    renderResult(data);
  } catch (err) {
    errorMsg.textContent = '⚠️ Something went wrong. Make sure the server is running.';
    errorMsg.classList.add('visible');
  } finally {
    btn.disabled = false;
    document.getElementById('btnText').textContent = '🔍 Analyze News';
    document.getElementById('btnSpinner').style.display = 'none';
    loadingOverlay.classList.remove('visible');
  }
}

let lastResultData = null;

function renderResult(data) {
  lastResultData = data;
  const isReal = data.ml_label === 'REAL';
  document.getElementById('verdictBanner').className = 'verdict-banner ' + (isReal ? 'real' : 'fake');
  document.getElementById('confBarWrap').className = 'conf-bar-wrap ' + (isReal ? 'real' : 'fake');
  document.getElementById('verdictIcon').textContent = isReal ? '✓' : '✗';
  document.getElementById('verdictLabel').textContent = (isReal ? '✅ REAL' : '❌ FAKE') + ' NEWS';
  document.getElementById('verdictSub').textContent = isReal ? 'This article appears credible.' : 'This article shows signs of misinformation.';
  document.getElementById('confidenceNum').textContent = data.ml_confidence + '%';
  document.getElementById('confBarPct').textContent = data.ml_confidence + '%';
  setTimeout(() => { document.getElementById('confBarFill').style.width = data.ml_confidence + '%'; }, 100);

  let truthPct = data.truth_percent;
  let fakePct = data.fake_percent;
  if (truthPct === undefined || truthPct === null) {
    const subScores = [data.claim_score, data.language_score, data.source_score].filter(s => s !== undefined && s !== null);
    if (subScores.length) {
      const avg = subScores.reduce((a, b) => a + b, 0) / subScores.length;
      truthPct = Math.round((avg / 10) * 1000) / 10;
    } else {
      truthPct = isReal ? 85 : 15;
    }
    fakePct = Math.round((100 - truthPct) * 10) / 10;
  }
  document.getElementById('credBarTruth').style.width = '0%';
  document.getElementById('credBarFake').style.width = '0%';
  setTimeout(() => {
    document.getElementById('credBarTruth').style.width = truthPct + '%';
    document.getElementById('credBarFake').style.width = fakePct + '%';
  }, 100);
  document.getElementById('credTruthPct').textContent = truthPct + '%';
  document.getElementById('credFakePct').textContent = fakePct + '%';
  document.getElementById('credibilityBlock').style.display = 'block';

  const llm = data.llm_analysis || '';
  document.getElementById('summaryText').textContent = extractSection(llm, 'SUMMARY') || 'Analysis complete.';
  document.getElementById('recommendationText').textContent = extractSection(llm, 'RECOMMENDATION') || 'Verify with credible sources.';

  const flagList = document.getElementById('redFlagsList');
  flagList.innerHTML = '';
  const redFlags = extractSection(llm, 'RED FLAGS') || '';
  redFlags.split('\n').filter(l => l.trim()).slice(0, 4).forEach(flag => {
    const li = document.createElement('li');
    li.textContent = flag.replace(/^[-•*\d.]+\s*/, '').trim();
    flagList.appendChild(li);
  });
  if (!flagList.children.length) {
    const li = document.createElement('li');
    li.textContent = isReal ? 'Credible source indicators found' : 'Suspicious language detected';
    flagList.appendChild(li);
  }

  if (data.claim_score || data.language_score || data.source_score) {
    const scoreColor = (s) => s >= 7 ? 'var(--real-color)' : s >= 4 ? '#fbbf24' : 'var(--fake-color)';
    document.getElementById('claimScore').textContent = data.claim_score ?? '—';
    document.getElementById('claimScore').style.color = data.claim_score ? scoreColor(data.claim_score) : 'var(--text-muted)';
    document.getElementById('languageScore').textContent = data.language_score ?? '—';
    document.getElementById('languageScore').style.color = data.language_score ? scoreColor(data.language_score) : 'var(--text-muted)';
    document.getElementById('sourceScore').textContent = data.source_score ?? '—';
    document.getElementById('sourceScore').style.color = data.source_score ? scoreColor(data.source_score) : 'var(--text-muted)';
    document.getElementById('scoreBreakdown').style.display = 'block';
  }

  if (data.self_critique) {
    // Backend sometimes bundles a "RED FLAGS:" section inside self_critique text —
    // red flags already have their own dedicated box above, so strip any duplicate here.
    const cleanCritique = data.self_critique.split(/RED FLAGS\s*:/i)[0].trim();
    document.getElementById('selfCritiqueText').textContent = cleanCritique || data.self_critique;
    document.getElementById('selfCritiqueBlock').style.display = 'block';
  }

  if (data.key_source && data.key_source.toLowerCase() !== 'none') {
    document.getElementById('keySourceText').textContent = '📰 ' + data.key_source;
    document.getElementById('keySourceBlock').style.display = 'block';
  }

  document.getElementById('mlBadge').textContent = '🤖 Model: ' + data.model_used + ' · ' + data.ml_confidence + '% confidence';
  document.getElementById('resultCard').classList.add('visible');
  document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });

  loadHistory();
  updateLiveStats();
}

function extractSection(text, section) {
  const regex = new RegExp('(?:###\\s*|\\*\\*)?' + section + ':?\\*?\\*?\\s*([\\s\\S]*?)(?=\\n(?:###\\s*|\\*\\*)?[A-Z ]+:?|$)', 'i');
  const match = text.match(regex);
  return match ? match[1].trim() : '';
}

// Load real time history
async function loadHistory() {
  try {
    const response = await fetch('/history');
    const data = await response.json();
    const tbody = document.getElementById('historyTableBody');

    if (!data || data.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="6" style="text-align:center;color:var(--text-muted);padding:20px">
            No predictions yet. Analyze some news! 😊
          </td>
        </tr>`;
      return;
    }

    tbody.innerHTML = data.map(row => `
      <tr>
        <td style="color:var(--text-muted)">${row.id}</td>
        <td>${row.headline.substring(0, 60)}...</td>
        <td><span class="badge ${row.prediction === 'REAL' ? 'badge-real' : 'badge-fake'}">${row.prediction}</span></td>
        <td style="color:${row.prediction === 'REAL' ? 'var(--real-color)' : 'var(--fake-color)'};font-weight:600">${row.confidence}%</td>
        <td style="color:var(--text-muted);font-size:11px">${row.model_used}</td>
        <td style="color:var(--text-muted)">${row.date}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error('History load failed:', err);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const newsInput = document.getElementById('newsInput');
  if (newsInput) {
    newsInput.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') analyzeNews();
    });
  }
  loadHistory();
});

// Copy result to clipboard
function copyResult() {
  if (!lastResultData) return;
  const d = lastResultData;
  const summary = extractSection(d.llm_analysis, 'SUMMARY') || '';
  const recommendation = extractSection(d.llm_analysis, 'RECOMMENDATION') || '';
  const scores = (d.claim_score || d.language_score || d.source_score)
    ? `\nScore Breakdown:\n  Claim Accuracy: ${d.claim_score ?? '—'}/10\n  Language Quality: ${d.language_score ?? '—'}/10\n  Source Credibility: ${d.source_score ?? '—'}/10`
    : '';
  const credibility = (d.truth_percent !== undefined && d.truth_percent !== null)
    ? `\nCredibility Breakdown: ${d.truth_percent}% truthful · ${d.fake_percent}% fabricated/misleading`
    : '';
  const critique = d.self_critique ? `\nAI Self-Critique: ${d.self_critique}` : '';
  const source = (d.key_source && d.key_source.toLowerCase() !== 'none') ? `\nKey Grounding Source: ${d.key_source}` : '';
  const text = `TruthScan AI — Analysis Result
Verdict: ${d.ml_label}
Confidence: ${d.ml_confidence}%
Model Used: ${d.model_used}${scores}${credibility}${critique}${source}

Summary: ${summary}

Recommendation: ${recommendation}

Analyzed by TruthScan AI — Built by Tilal Ahmed`;

  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    const original = btn.textContent;
    btn.textContent = '✅ Copied!';
    setTimeout(() => { btn.textContent = original; }, 2000);
  }).catch(() => {
    alert('Could not copy automatically. Please select and copy manually.');
  });
}

// Clear history
async function clearHistory() {
  if (!confirm('Are you sure you want to delete all prediction history? This cannot be undone.')) return;
  try {
    const response = await fetch('/clear-history', { method: 'POST' });
    const data = await response.json();
    if (data.success) {
      loadHistory();
      updateLiveStats();
    }
  } catch (err) {
    console.error('Clear history failed:', err);
  }
}

// Export history as CSV
async function exportHistoryCSV() {
  try {
    const response = await fetch('/history');
    const data = await response.json();
    if (!data || data.length === 0) {
      alert('No history to export yet.');
      return;
    }
    let csv = 'ID,Headline,Prediction,Confidence,Model,Date\n';
    data.forEach(row => {
      const headline = row.headline.replace(/"/g, '""').replace(/\n/g, ' ');
      csv += `${row.id},"${headline}",${row.prediction},${row.confidence},${row.model_used},"${row.date}"\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', 'truthscan_history.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  } catch (err) {
    console.error('Export failed:', err);
    alert('Could not export history.');
  }
}
