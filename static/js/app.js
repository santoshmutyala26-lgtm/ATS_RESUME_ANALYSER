/* ═══════════════════════════════════════════
   ResumeIQ — Frontend Application JS
   Handles: upload, API calls, rendering, charts, interactions
═══════════════════════════════════════════ */

'use strict';

// ─── State ───────────────────────────────────
let analysisData = null;
let scoreRingChart = null;
let densityChartInstance = null;
let radarChartInstance = null;

// ─── DOM refs ─────────────────────────────────
const uploadScreen   = document.getElementById('uploadScreen');
const loadingScreen  = document.getElementById('loadingScreen');
const resultsScreen  = document.getElementById('resultsScreen');
const analyzeBtn     = document.getElementById('analyzeBtn');
const resumeInput    = document.getElementById('resumeInput');
const jobDescTA      = document.getElementById('jobDesc');
const dropZone       = document.getElementById('dropZone');
const dropText       = document.getElementById('dropText');
const fileChosen     = document.getElementById('fileChosen');
const fileName       = document.getElementById('fileName');
const jdCount        = document.getElementById('jdCount');
const errorToast     = document.getElementById('errorToast');
const errorMsg       = document.getElementById('errorMsg');
const backBtn        = document.getElementById('backBtn');
const optimizeBtn    = document.getElementById('optimizeBtn');

// ─── Upload interactions ───────────────────────
resumeInput.addEventListener('change', () => {
  if (resumeInput.files[0]) {
    fileName.textContent = resumeInput.files[0].name;
    dropText.textContent = resumeInput.files[0].name;
    fileChosen.classList.remove('hidden');
  }
  checkReady();
});

jobDescTA.addEventListener('input', () => {
  jdCount.textContent = jobDescTA.value.length.toLocaleString();
  checkReady();
});

// Drag & drop
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type === 'application/pdf') {
    const dt = new DataTransfer();
    dt.items.add(file);
    resumeInput.files = dt.files;
    fileName.textContent = file.name;
    dropText.textContent = file.name;
    fileChosen.classList.remove('hidden');
    checkReady();
  } else {
    showError('Please drop a PDF file.');
  }
});

function checkReady() {
  const hasFile = resumeInput.files && resumeInput.files.length > 0;
  const hasJD   = jobDescTA.value.trim().length >= 50;
  analyzeBtn.disabled = !(hasFile && hasJD);
}

// ─── Wake up server on load (for Render free tier) ───────
const statusEl = document.getElementById('connectionStatus');
fetch('/').then(() => {
  if (statusEl) statusEl.textContent = 'AI Server Ready';
}).catch(() => {
  if (statusEl) statusEl.textContent = 'Server starting... (waking up)';
});

// ─── Main analyze ──────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  console.log('Analyze button clicked'); // Debug log
  if (analyzeBtn.disabled) return;

  const formData = new FormData();
  formData.append('resume', resumeInput.files[0]);
  formData.append('job_description', jobDescTA.value.trim());

  showLoading();
  let retries = 3;

  while (retries > 0) {
    try {
      const resp = await fetch('/analyze', { method: 'POST', body: formData });
      
      // Handle non-JSON responses (like Render 502/504 timeouts)
      const contentType = resp.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
         throw new Error("Network error during cold-start. Retrying...");
      }

      const data = await resp.json();

      if (!resp.ok) {
        throw new Error(data.error || 'Analysis failed.');
      }

      analysisData = data;
      hideLoading();
      renderResults(data);
      return;

    } catch (err) {
      console.error('Fetch error:', err);
      // Catch network errors OR the custom "Network error" we just threw
      if (err.message.includes('fetch') || err.message.includes('Network') || err.message.includes('token')) {
        retries--;
        if (retries > 0) {
          const title = document.getElementById('loadingTitle');
          const sub = document.getElementById('loadingSub');
          if (title) title.textContent = 'Waking up AI server...';
          if (sub) sub.textContent = 'This takes ~50 seconds on the first try. Please wait.';
          await new Promise(r => setTimeout(r, 5000));
          continue;
        }
      }
      hideLoading();
      showError(err.message || 'Server error. Please try again in 30 seconds.');
      showScreen(uploadScreen);
      return;
    }
  }
});

// ─── Back / optimize buttons ───────────────────
backBtn.addEventListener('click', () => {
  showScreen(uploadScreen);
  analysisData = null;
});

optimizeBtn.addEventListener('click', () => {
  showScreen(uploadScreen);
  analysisData = null;
});

// ─── Screen management ─────────────────────────
function showScreen(screen) {
  [uploadScreen, loadingScreen, resultsScreen].forEach(s => s.classList.add('hidden'));
  screen.classList.remove('hidden');
}

function showLoading() {
  const title = document.getElementById('loadingTitle');
  const sub = document.getElementById('loadingSub');
  if (title) title.textContent = 'Analyzing Resume...';
  if (sub) sub.textContent = 'Our AI is extracting skills and scoring your profile against the JD.';
  showScreen(loadingScreen);
  animateLoadingSteps();
}

function hideLoading() {
  loadingScreen.classList.add('hidden');
}

function animateLoadingSteps() {
  const steps = ['step1','step2','step3','step4','step5'];
  const pct = [0, 20, 45, 68, 88, 100];
  let i = 0;

  steps.forEach(id => {
    document.getElementById(id).classList.remove('active','done');
  });
  document.getElementById('loadingPct').textContent = '0%';

  const interval = setInterval(() => {
    if (i >= steps.length) { clearInterval(interval); return; }
    if (i > 0) {
      document.getElementById(steps[i-1]).classList.remove('active');
      document.getElementById(steps[i-1]).classList.add('done');
    }
    document.getElementById(steps[i]).classList.add('active');
    document.getElementById('loadingPct').textContent = pct[i+1] + '%';
    i++;
  }, 700);
}

// ─── RENDER RESULTS ───────────────────────────
function renderResults(data) {
  showScreen(resultsScreen);
  document.getElementById('analysisTimestamp').textContent =
    'Analyzed ' + new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});

  renderScoreRing(data);
  renderBreakdown(data);
  renderSkillTabs(data);
  renderSections(data);
  renderDensityChart(data);
  renderRadarChart(data);
  renderMiniScores(data);
  renderResumePreview(data);
  renderRewrites(data);
  renderSkillGaps(data);
  renderOptimizeMode(data);
  renderKeywordLog(data);  // keyword log panel
}

// ─── Score Ring ────────────────────────────────
function renderScoreRing(data) {
  const score = data.ats_score;
  const color = data.score_color;

  // Animate number
  animateNumber('scoreNumber', 0, score, 1200);
  document.getElementById('scoreVerdict').textContent = data.score_label;
  document.getElementById('scoreVerdict').style.color = color;
  document.getElementById('scoreLabel').textContent = scoreDescription(score);

  if (scoreRingChart) scoreRingChart.destroy();

  const ctx = document.getElementById('scoreRing').getContext('2d');
  scoreRingChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [score, 100 - score],
        backgroundColor: [color, 'rgba(255,255,255,0.04)'],
        borderWidth: 0,
        circumference: 280,
        rotation: 220,
      }]
    },
    options: {
      responsive: false, cutout: '72%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      animation: { duration: 1200, easing: 'easeOutQuart' }
    }
  });
}

function scoreDescription(score) {
  if (score >= 80) return 'Strong match — ready to submit';
  if (score >= 65) return 'Good match — minor improvements recommended';
  if (score >= 50) return 'Moderate match — address missing skills';
  if (score >= 35) return 'Weak match — significant gaps detected';
  return 'Poor match — major rework needed';
}

// ─── Score Breakdown ───────────────────────────
function renderBreakdown(data) {
  const bd = data.score_breakdown;
  const wt = data.weights;
  const items = [
    { name: 'Keyword Match',       key: 'keyword_match',        color: '#6c63ff' },
    { name: 'Semantic Similarity', key: 'semantic_similarity',  color: '#22c55e' },
    { name: 'Skill Coverage',      key: 'skill_coverage',       color: '#f59e0b' },
    { name: 'Experience Relevance',key: 'experience_relevance', color: '#3b82f6' },
  ];

  const list = document.getElementById('breakdownList');
  list.innerHTML = items.map(item => {
    const score = bd[item.key] || 0;
    const weight = wt[item.key] || 0;
    return `
      <div class="breakdown-item">
        <div class="breakdown-row">
          <span class="breakdown-name">${item.name}</span>
          <span class="breakdown-score" style="color:${item.color}">${score}</span>
        </div>
        <div class="breakdown-bar-track">
          <div class="breakdown-bar-fill" style="width:0%;background:${item.color}"
            data-target="${score}"></div>
        </div>
        <div class="breakdown-weight">Weight: ${weight}% of total</div>
      </div>
    `;
  }).join('');

  // Animate bars after DOM insert
  setTimeout(() => {
    list.querySelectorAll('.breakdown-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  }, 50);
}

// ─── Skill Tabs ────────────────────────────────
function renderSkillTabs(data) {
  // Missing
  const missingPanel = document.getElementById('tab-missing');
  if (data.missing_skills && data.missing_skills.length > 0) {
    missingPanel.innerHTML = `<div class="skill-list">${
      data.missing_skills.slice(0, 20).map(s => `
        <div class="skill-tag missing">
          <span class="skill-tag-name">${s.name}</span>
          <span class="skill-impact">+${s.impact_score || 5}%</span>
        </div>
      `).join('')
    }</div>`;
  } else {
    missingPanel.innerHTML = '<p style="font-size:13px;color:#5a5a7a;text-align:center;padding:20px 0">No critical missing skills detected 🎉</p>';
  }

  // Matched
  const matchedPanel = document.getElementById('tab-matched');
  if (data.matched_skills && data.matched_skills.length > 0) {
    matchedPanel.innerHTML = `<div class="skill-list">${
      data.matched_skills.slice(0, 20).map(s => `
        <div class="skill-tag matched">
          <span class="skill-tag-name">${s.name}</span>
          <span class="skill-tag-cat">${s.category || ''}</span>
        </div>
      `).join('')
    }</div>`;
  } else {
    matchedPanel.innerHTML = '<p style="font-size:13px;color:#5a5a7a;text-align:center;padding:20px 0">No matched skills found</p>';
  }

  // Clusters
  const clustersPanel = document.getElementById('tab-clusters');
  const clusters = data.skill_clusters || {};
  if (Object.keys(clusters).length > 0) {
    const labels = {
      tools: '🔧 Tools', techniques: '⚙️ Techniques', soft_skills: '💬 Soft Skills',
      domain_knowledge: '🧠 Domain', languages_frameworks: '💻 Languages & Frameworks', general: '📦 Other'
    };
    clustersPanel.innerHTML = Object.entries(clusters).map(([cat, skills]) => `
      <div class="cluster-group">
        <div class="cluster-name">${labels[cat] || cat}</div>
        <div class="cluster-tags">${
          skills.map(s => `<span class="cluster-tag">${s}</span>`).join('')
        }</div>
      </div>
    `).join('');
  } else {
    clustersPanel.innerHTML = '<p style="font-size:13px;color:#5a5a7a;padding:10px 0">No clusters detected</p>';
  }

  // Tab switching
  document.querySelectorAll('.skill-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.skill-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      tab.classList.add('active');
      document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
  });
}

// ─── Keyword Log ───────────────────────────────
function renderKeywordLog(data) {
  const log = data.keyword_match_log || [];
  const panel = document.getElementById('tab-kwlog');
  if (!panel) return;

  if (log.length === 0) {
    panel.innerHTML = '<p style="font-size:13px;color:#5a5a7a;padding:16px 0">No keyword matches recorded.</p>';
    return;
  }

  // Group by match_type
  const exact   = log.filter(k => k.match_type === 'exact');
  const fuzzy   = log.filter(k => k.match_type === 'fuzzy');
  const partial = log.filter(k => k.match_type === 'partial');

  function badge(type) {
    const map = {
      exact:   { label: 'Exact',   color: '#22c55e' },
      fuzzy:   { label: 'Fuzzy',   color: '#f59e0b' },
      partial: { label: 'Partial', color: '#f97316' },
    };
    const b = map[type] || { label: type, color: '#9494b8' };
    return `<span style="font-size:10px;padding:1px 6px;border-radius:9px;background:${b.color}22;color:${b.color};font-weight:600;letter-spacing:.4px">${b.label}</span>`;
  }

  function importanceBadge(imp) {
    const map = {
      required: { label: 'Required', color: '#6c63ff' },
      moderate: { label: 'Moderate', color: '#3b82f6' },
      optional: { label: 'Optional', color: '#5a5a7a' },
    };
    const b = map[imp] || { label: imp, color: '#5a5a7a' };
    return `<span style="font-size:10px;padding:1px 6px;border-radius:9px;background:${b.color}22;color:${b.color}">${b.label}</span>`;
  }

  function renderGroup(items, heading) {
    if (!items.length) return '';
    return `
      <div style="margin-bottom:14px">
        <div style="font-size:11px;font-weight:700;color:#9494b8;letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px">
          ${heading} &mdash; ${items.length} keyword${items.length > 1 ? 's' : ''}
        </div>
        ${items.map(k => `
          <div class="kwlog-row">
            <div class="kwlog-name">${k.keyword}</div>
            <div class="kwlog-badges">
              ${badge(k.match_type)}
              ${importanceBadge(k.jd_importance)}
            </div>
            <div class="kwlog-note">${k.explanation}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  panel.innerHTML = `
    <div class="kwlog-summary">
      <span class="kwlog-stat"><span style="color:#22c55e;font-weight:700">${exact.length}</span> exact</span>
      <span class="kwlog-sep">/</span>
      <span class="kwlog-stat"><span style="color:#f59e0b;font-weight:700">${fuzzy.length}</span> fuzzy</span>
      <span class="kwlog-sep">/</span>
      <span class="kwlog-stat"><span style="color:#f97316;font-weight:700">${partial.length}</span> partial</span>
      <span class="kwlog-sep">/</span>
      <span class="kwlog-stat"><span style="color:#9494b8;font-weight:700">${log.length}</span> total</span>
    </div>
    <div class="kwlog-list">
      ${renderGroup(exact,   '✓ Exact Matches')}
      ${renderGroup(fuzzy,   '~ Fuzzy Matches')}
      ${renderGroup(partial, '≈ Partial Matches')}
    </div>
  `;
}

// ─── Sections ──────────────────────────────────
function renderSections(data) {
  const sections = data.sections;
  if (!sections) return;
  const el = document.getElementById('sectionHealth');
  const sectionData = sections.sections || {};

  el.innerHTML = Object.entries(sectionData).map(([name, info]) => {
    const present = info.present;
    const quality = info.quality || 'missing';
    const statusClass = present ? `present ${quality}` : 'missing';
    const statusText = present ? quality : 'missing';
    return `
      <div class="section-row">
        <span class="section-name">${name}</span>
        <span class="section-status ${statusClass}">${statusText}</span>
      </div>
    `;
  }).join('');
}

// ─── Keyword Density Chart ─────────────────────
function renderDensityChart(data) {
  const density = data.keyword_density || {};
  const entries = Object.entries(density).slice(0, 10);
  if (!entries.length) return;

  const labels = entries.map(([kw]) => kw.length > 12 ? kw.slice(0,12)+'…' : kw);
  const values = entries.map(([, v]) => v.count || 0);
  const colors = entries.map(([, v]) => {
    if (v.status === 'optimal') return '#22c55e';
    if (v.status === 'too_low') return '#f59e0b';
    return '#ef4444';
  });

  if (densityChartInstance) densityChartInstance.destroy();
  const ctx = document.getElementById('densityChart').getContext('2d');
  densityChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Occurrences',
        data: values,
        backgroundColor: colors,
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: {
          label: (ctx) => {
            const entry = entries[ctx.dataIndex];
            return ` ${entry[1].count} times (${entry[1].frequency_pct}%) — ${entry[1].status}`;
          }
        }
      }},
      scales: {
        x: { ticks: { color: '#9494b8', font: { size: 10 } }, grid: { display: false } },
        y: { ticks: { color: '#9494b8', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

// ─── Radar Chart ───────────────────────────────
function renderRadarChart(data) {
  const bd = data.score_breakdown;
  if (radarChartInstance) radarChartInstance.destroy();

  const ctx = document.getElementById('radarChart').getContext('2d');
  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Keywords', 'Semantics', 'Skills', 'Experience'],
      datasets: [{
        label: 'Your Resume',
        data: [
          bd.keyword_match, bd.semantic_similarity, bd.skill_coverage, bd.experience_relevance
        ],
        backgroundColor: 'rgba(108,99,255,0.15)',
        borderColor: '#6c63ff',
        borderWidth: 2,
        pointBackgroundColor: '#6c63ff',
        pointRadius: 4,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { stepSize: 25, color: '#9494b8', font: { size: 10 }, backdropColor: 'transparent' },
          grid: { color: 'rgba(255,255,255,0.06)' },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
          pointLabels: { color: '#9494b8', font: { size: 11 } }
        }
      }
    }
  });
}

// ─── Mini score cards ──────────────────────────
function renderMiniScores(data) {
  const bd = data.score_breakdown;
  const items = [
    { label: 'Keywords',    key: 'keyword_match',        color: '#6c63ff' },
    { label: 'Semantics',   key: 'semantic_similarity',  color: '#22c55e' },
    { label: 'Skills',      key: 'skill_coverage',       color: '#f59e0b' },
    { label: 'Experience',  key: 'experience_relevance', color: '#3b82f6' },
  ];

  document.getElementById('miniScoresGrid').innerHTML = items.map(item => {
    const score = bd[item.key] || 0;
    return `
      <div class="mini-score-card">
        <div class="mini-score-label">${item.label}</div>
        <div class="mini-score-value" style="color:${item.color}">${score}</div>
        <div class="mini-score-sub">/100</div>
        <div class="mini-score-bar">
          <div class="mini-score-bar-fill" style="width:${score}%;background:${item.color}"></div>
        </div>
      </div>
    `;
  }).join('');
}

// ─── Resume Preview ────────────────────────────
function renderResumePreview(data) {
  const preview = document.getElementById('resumePreview');
  const highlighted = data.resume_preview?.highlighted || '';
  preview.innerHTML = highlighted || '<em style="color:#5a5a7a">No preview available</em>';
}

// ─── Rewrites ──────────────────────────────────
function renderRewrites(data) {
  const sugg = data.suggestions || {};

  // Bullet rewrites
  const rewrites = sugg.bullet_rewrites || [];
  const rewritesList = document.getElementById('rewritesList');
  if (rewrites.length > 0) {
    rewritesList.innerHTML = rewrites.map(r => `
      <div class="rewrite-item">
        <div class="rewrite-before-label">BEFORE — Weak</div>
        <div class="rewrite-before">${r.original}</div>
        <div class="rewrite-after-label">AFTER — Improved</div>
        <div class="rewrite-after">${r.suggested}</div>
        <div class="rewrite-improvement">✦ ${r.improvement}</div>
      </div>
    `).join('');
  } else {
    rewritesList.innerHTML = `
      <div style="padding:20px;text-align:center;color:#5a5a7a;font-size:13px;">
        No weak bullet points detected — your bullets are well-formed!
      </div>
    `;
  }

  // Keyword advice (replaces old fabricated sentences)
  const kwAdvice = sugg.missing_keyword_advice || [];
  document.getElementById('keywordSentences').innerHTML = kwAdvice.length > 0
    ? kwAdvice.map(s => `
        <div class="kw-sentence-item">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span class="kw-skill-badge">${s.skill}</span>
            <span style="font-size:10px;padding:2px 7px;border-radius:9px;background:${
              s.jd_importance === 'required' ? '#6c63ff22' : '#5a5a7a22'
            };color:${
              s.jd_importance === 'required' ? '#6c63ff' : '#9494b8'
            }">${s.jd_importance || 'moderate'}</span>
            ${s.estimated_score_boost ? `<span style="font-size:10px;color:#22c55e">${s.estimated_score_boost}</span>` : ''}
          </div>
          <div class="kw-sentence" style="font-style:normal;margin-bottom:4px">${s.why_it_matters}</div>
          <div class="kw-placement">💡 ${s.role_context}</div>
          <div class="kw-placement">📍 Add to: ${s.where_to_add}</div>
          ${s.context_in_jd ? `<div class="kw-placement" style="margin-top:4px;opacity:.7">📄 In JD: ${s.context_in_jd}</div>` : ''}
        </div>
      `).join('')
    : '<p style="font-size:13px;color:#5a5a7a">No missing skill recommendations — great coverage!</p>';

  // Quick wins
  const wins = sugg.quick_wins || [];
  document.getElementById('quickWins').innerHTML = `
    <div class="quick-win-list">${
      wins.map(w => `
        <div class="quick-win">
          <span class="quick-win-icon">★</span>
          <span>${w}</span>
        </div>
      `).join('')
    }</div>
  `;
}

// ─── Skill Gaps ────────────────────────────────
function renderSkillGaps(data) {
  const missing = data.missing_skills || [];
  const el = document.getElementById('skillGapList');

  if (missing.length === 0) {
    el.innerHTML = '<p style="font-size:13px;color:#5a5a7a;text-align:center;padding:30px 0">No significant skill gaps detected!</p>';
    return;
  }

  el.innerHTML = missing.slice(0, 15).map((skill, i) => `
    <div class="skill-gap-item">
      <div class="gap-rank">#${i+1}</div>
      <div class="gap-info">
        <div class="gap-name">${skill.name}</div>
        ${skill.context ? `<div class="gap-context">${skill.context}</div>` : ''}
        <span class="gap-category">${formatCategory(skill.category)}</span>
      </div>
      <div class="gap-impact">
        <div class="gap-impact-num">+${skill.impact_score || 5}%</div>
        <div class="gap-impact-label">score boost<br>if added</div>
      </div>
    </div>
  `).join('');
}

// ─── Optimize Mode ─────────────────────────────
function renderOptimizeMode(data) {
  const sugg = data.suggestions || {};
  const opt = sugg.ats_optimization || {};
  const sectionImprovements = sugg.section_improvements || [];
  const el = document.getElementById('optimizeContent');

  let html = '';

  // Missing keywords
  if (opt.missing_keywords && opt.missing_keywords.length > 0) {
    html += `
      <div class="opt-section">
        <div class="opt-section-title">Missing Keywords to Add</div>
        <div class="opt-keyword-grid">${
          opt.missing_keywords.map(k => `<span class="opt-keyword">${k}</span>`).join('')
        }</div>
      </div>
    `;
  }

  // Insertion sentences
  if (opt.insertion_sentences && opt.insertion_sentences.length > 0) {
    html += `
      <div class="opt-section">
        <div class="opt-section-title">Suggested Sentences to Insert</div>
        ${opt.insertion_sentences.map(s => `
          <div class="opt-sentence-item">
            <div class="opt-sentence-kw">Integrates: ${s.keyword}</div>
            <div class="opt-sentence-text">"${s.sentence}"</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  // Section actions
  if (sectionImprovements.length > 0) {
    html += `
      <div class="opt-section">
        <div class="opt-section-title">Section Action Plan</div>
        ${sectionImprovements.map(s => `
          <div class="opt-action ${s.priority === 'critical' ? 'critical' : 'recommended'}">
            <strong>[${s.priority.toUpperCase()}]</strong>&nbsp;${s.advice}
          </div>
        `).join('')}
      </div>
    `;
  }

  // Formatting tips
  if (opt.formatting_tips && opt.formatting_tips.length > 0) {
    html += `
      <div class="opt-section">
        <div class="opt-section-title">ATS Formatting Tips</div>
        ${opt.formatting_tips.map(tip => `<div class="opt-format-tip">${tip}</div>`).join('')}
      </div>
    `;
  }

  // Match details
  const matchDetails = data.match_details || [];
  if (matchDetails.length > 0) {
    html += `
      <div class="opt-section">
        <div class="opt-section-title">Keyword Match Details</div>
        ${matchDetails.map(m => `
          <div class="match-detail-row">
            <span class="match-kw">${m.keyword}</span>
            <span class="match-type-badge ${m.match_type}">${m.match_type}</span>
            <span class="match-explain">${m.explanation}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  el.innerHTML = html || '<p style="font-size:13px;color:#5a5a7a;padding:20px 0">Run analysis to see optimization recommendations</p>';
}

// ─── Tab switching (right panel) ───────────────
document.querySelectorAll('.right-tab').forEach(tab => {
  tab.addEventListener('click', () => activateRightTab(tab.dataset.rtab));
});

function activateRightTab(tabId) {
  document.querySelectorAll('.right-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.right-panel-content').forEach(p => p.classList.remove('active'));
  const tab = document.querySelector(`.right-tab[data-rtab="${tabId}"]`);
  const panel = document.getElementById('rtab-' + tabId);
  if (tab) tab.classList.add('active');
  if (panel) panel.classList.add('active');
}

// ─── Helpers ───────────────────────────────────
function animateNumber(id, from, to, duration) {
  const el = document.getElementById(id);
  const start = performance.now();
  const update = (now) => {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.round(from + (to - from) * eased);
    if (progress < 1) requestAnimationFrame(update);
  };
  requestAnimationFrame(update);
}

function formatCategory(cat) {
  const map = {
    tools: '🔧 Tool', techniques: '⚙️ Technique', soft_skills: '💬 Soft Skill',
    domain_knowledge: '🧠 Domain', languages_frameworks: '💻 Language/Framework', general: '📦 General'
  };
  return map[cat] || cat;
}

function showError(msg) {
  errorMsg.textContent = msg;
  errorToast.classList.remove('hidden');
  setTimeout(() => errorToast.classList.add('hidden'), 5000);
}

// ─── Init ──────────────────────────────────────
showScreen(uploadScreen);
