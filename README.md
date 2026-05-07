# ResumeIQ — ATS Resume Analyzer & Optimizer

A production-quality ATS simulation system that analyzes resumes against job descriptions using a multi-factor NLP pipeline — similar to how enterprise ATS systems at top companies work.

---

## 🚀 Quick Start

### 1. Setup (first time only)

```bash
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run

```bash
source venv/bin/activate
python app.py
```

Open **http://localhost:5000** in your browser. You will see the new Project Intro page. Click **Get Started** to access the analyzer at `/analyzer`.

---

## 🌐 Deployment (Render)

This project is configured for deployment on **Render**.

### Outbound IP Addresses
If you need to connect this service to external databases or APIs that require whitelisting, use these Render outbound IP ranges:
- `74.220.49.0/24`
- `74.220.57.0/24`

### GitHub Integration
Connect your GitHub repository to Render:
**Repo:** [https://github.com/santoshmutyala26-lgtm/ATS_RESUME_ANALYSER](https://github.com/santoshmutyala26-lgtm/ATS_RESUME_ANALYSER)

---

## 📁 Project Structure

```
ats_analyzer/
├── app.py                    # Flask backend — API routes
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-command setup script
│
├── utils/
│   ├── __init__.py
│   ├── text_extractor.py     # PDF text extraction (pdfminer + pypdf fallback)
│   ├── nlp_pipeline.py       # NLP: tokenization, phrase extraction, embeddings
│   ├── scoring_engine.py     # Multi-factor ATS scoring engine
│   ├── skill_analyzer.py     # Skill gap intelligence & clustering
│   └── optimizer.py          # Resume rewrite & optimization suggestions
│
├── templates/
│   └── index.html            # Main frontend (SaaS dashboard)
│
└── static/
    ├── css/style.css         # Premium dark UI stylesheet
    └── js/app.js             # Frontend logic, charts, interactions
```

---

## 🧠 How the Scoring Works

| Factor | Weight | Method |
|--------|--------|--------|
| Keyword Match | 30% | Exact + fuzzy matching via SequenceMatcher |
| Semantic Similarity | 30% | Cosine similarity via Sentence Transformers embeddings |
| Skill Coverage | 20% | Taxonomy-based skill matching (ESCO-inspired) |
| Experience Relevance | 20% | Action verbs, quantification, JD keyword density in exp section |

**Final ATS Score = Weighted average with realism caps**

Scores are capped to avoid fake 100% results. A resume needs to truly match across all dimensions.

---

## ✨ Features

- **Circular ATS Score** with animated ring
- **Radar chart** — match profile across 4 dimensions
- **Keyword density analysis** — bar chart with optimal/low/high indicators
- **Section health detector** — identifies missing or weak resume sections
- **Skill gap intelligence** — missing skills ranked by score impact (+X%)
- **AI bullet point rewrites** — before/after with strong action verbs
- **Keyword sentence suggestions** — context-aware sentences to add
- **ATS optimization mode** — complete action plan
- **Skill clustering** — grouped by Tools, Techniques, Soft Skills, Domain, Languages
- **Highlighted resume preview** — matched keywords highlighted green

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Flask 3.0 |
| PDF Parsing | pdfminer.six + pypdf |
| NLP | spaCy en_core_web_sm |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Similarity | scikit-learn cosine + SequenceMatcher |
| Frontend | HTML + Tailwind (CDN) + Vanilla JS |
| Charts | Chart.js 4.4 |
| Fonts | Syne + DM Sans (Google Fonts) |

---

## 🔧 Configuration

Edit `utils/nlp_pipeline.py` → `SKILL_TAXONOMY` to add domain-specific skills.

Edit `utils/skill_analyzer.py` → `SKILL_IMPORTANCE` to tune skill weight/impact scores.

Edit `utils/scoring_engine.py` → `WEIGHTS` to adjust scoring factor weights.

---

## 📝 Notes

- **Text-based PDFs only** — scanned image PDFs require OCR (not included)
- **First startup** is slower — Sentence Transformers downloads a ~90MB model
- All processing is **local** — no data is sent to external APIs
- For production deployment, use **Gunicorn**: `gunicorn app:app -w 4 -b 0.0.0.0:5000`

---

## 📄 License

MIT License — free to use, modify, and distribute.
