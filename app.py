"""
ATS Resume Analyzer & Optimizer — Production Backend
Flask API with NLP pipeline, scoring engine, and optimization suggestions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import json
import re
import math
from utils.text_extractor import extract_text_from_pdf
from utils.nlp_pipeline import NLPPipeline
from utils.scoring_engine import ScoringEngine
from utils.skill_analyzer import SkillAnalyzer
from utils.optimizer import ResumeOptimizer

app = Flask(__name__)
CORS(app)

# Initialize pipeline components (lazy-loaded for performance)
nlp_pipeline = None
scoring_engine = None
skill_analyzer = None
optimizer = None

def get_pipeline():
    global nlp_pipeline, scoring_engine, skill_analyzer, optimizer
    if nlp_pipeline is None:
        nlp_pipeline = NLPPipeline()
        scoring_engine = ScoringEngine()
        skill_analyzer = SkillAnalyzer()
        optimizer = ResumeOptimizer()
    return nlp_pipeline, scoring_engine, skill_analyzer, optimizer


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    Accepts: resume (PDF file) + job_description (text)
    Returns: comprehensive ATS analysis JSON
    """
    try:
        # --- Input validation ---
        if "resume" not in request.files:
            return jsonify({"error": "No resume file provided"}), 400
        if "job_description" not in request.form:
            return jsonify({"error": "No job description provided"}), 400

        resume_file = request.files["resume"]
        job_description = request.form["job_description"].strip()

        if not resume_file.filename.endswith(".pdf"):
            return jsonify({"error": "Resume must be a PDF file"}), 400
        if len(job_description) < 50:
            return jsonify({"error": "Job description is too short"}), 400

        # --- Text Extraction ---
        resume_bytes = resume_file.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        if not resume_text or len(resume_text) < 100:
            return jsonify({"error": "Could not extract text from PDF. Ensure it's not scanned/image-only."}), 400

        # --- NLP Pipeline ---
        pipeline, scorer, skill_ana, opt = get_pipeline()

        resume_doc = pipeline.process(resume_text)
        jd_doc = pipeline.process(job_description)

        # --- Section Analysis (must run before scoring for experience bonus) ---
        sections = pipeline.detect_sections(resume_text)

        # --- Multi-factor Scoring ---
        # Pass sections so experience score can use section-presence bonus (Fix 5)
        scores = scorer.compute_scores(resume_doc, jd_doc, sections)

        # --- Skill Analysis ---
        skill_report = skill_ana.analyze(resume_doc, jd_doc)

        # --- Keyword Density ---
        keyword_density = pipeline.keyword_density(resume_text, jd_doc["keywords"])

        # --- Optimization Suggestions ---
        suggestions = opt.generate_suggestions(resume_doc, jd_doc, skill_report, sections)

        # --- Build Response ---
        response = {
            "ats_score": scores["final_score"],
            "score_breakdown": {
                "keyword_match": scores["keyword_score"],
                "semantic_similarity": scores["semantic_score"],
                "skill_coverage": scores["skill_score"],
                "experience_relevance": scores["experience_score"],
            },
            "weights": {
                "keyword_match": 30,
                "semantic_similarity": 30,
                "skill_coverage": 20,
                "experience_relevance": 20,
            },
            "matched_skills": skill_report["matched"],
            # Missing skills include learning_advice so frontend can render
            # honest 'consider learning X' suggestions — never fabricated lines
            "missing_skills": skill_report["missing"],
            "skill_clusters": skill_report["clusters"],
            "keyword_density": keyword_density,
            "sections": sections,
            "suggestions": suggestions,
            # Renamed key from missing_keyword_sentences → missing_keyword_advice
            "missing_keyword_advice": suggestions.get("missing_keyword_advice", []),
            "match_details": scores["match_details"],
            # Full keyword match log — every matched keyword with metadata
            # This is the authoritative list; match_details contains ALL JD keywords
            # including non-matches. keyword_match_log is ONLY the matched ones.
            "keyword_match_log": _build_keyword_match_log(
                scores["match_details"],
                skill_report["matched_keywords"],
            ),
            "resume_preview": {
                "text": resume_text[:3000],
                "highlighted": _build_highlighted(
                    resume_text[:3000],
                    skill_report["matched_keywords"],
                    skill_report["missing_keywords"],
                ),
            },
            "score_label": _score_label(scores["final_score"]),
            "score_color": _score_color(scores["final_score"]),
            # Role detection — drives weighting decisions
            "detected_role": scores.get("detected_role", skill_report.get("detected_role", "unknown")),
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


def _build_keyword_match_log(match_details: list, matched_keywords: list) -> list:
    """
    Build the full keyword match log — every matched keyword annotated with metadata.
    Merges scoring engine match_details (which covers all JD keywords) with
    the skill_analyzer matched_keywords list (taxonomy-matched skills).
    Returns a deduplicated, sorted list of matched keyword records.
    """
    log = []
    seen = set()

    # From match_details: all keywords that had an exact, synonym, fuzzy, or partial match
    # Fix 1: include 'synonym' type — prevents synonym-matched keywords from leaking into missing
    for m in match_details:
        if m["match_type"] in ("exact", "synonym", "fuzzy", "partial"):
            kw = m["keyword"].lower()
            if kw not in seen:
                seen.add(kw)
                log.append({
                    "keyword": m["keyword"],
                    "match_type": m["match_type"],
                    "confidence": m["confidence"],
                    "jd_importance": m.get("jd_importance", "moderate"),
                    "role_alignment": m.get("role_alignment", "neutral"),
                    "explanation": m.get("explanation", ""),
                    "source": "keyword_match",
                })

    # From matched_keywords: taxonomy skills confirmed in both docs
    for kw in matched_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            log.append({
                "keyword": kw,
                "match_type": "exact",
                "confidence": 1.0,
                "jd_importance": "moderate",
                "role_alignment": "neutral",
                "explanation": f"'{kw}' found in both resume and job description",
                "source": "skill_match",
            })

    # Sort: exact first, then by keyword alpha
    log.sort(key=lambda x: (x["match_type"] != "exact", x["keyword"].lower()))
    return log


def _build_highlighted(text, matched_kw, missing_kw):
    """
    Build highlighted HTML from resume text.
    Highlights ALL occurrences of every matched keyword (green).

    Uses a safe, span-based approach that:
    - Escapes HTML BEFORE any highlighting
    - Collects all match positions, then builds output in one pass
    - Prevents nested / overlapping highlights
    - Preserves original casing of matched text
    - Supports multi-word phrases, hyphenated words, case-insensitive
    """
    import html as _html

    escaped = _html.escape(text)

    # ── 1. Collect all keyword match spans (longest-first priority) ──
    # Sort keywords longest-first so "machine learning" beats "machine"
    sorted_kws = sorted(set(matched_kw), key=len, reverse=True)

    spans = []  # list of (start, end, matched_text) — non-overlapping
    occupied = set()  # character positions already claimed

    for kw in sorted_kws:
        escaped_kw = _html.escape(kw)
        # Build pattern with word boundaries for whole-word matching
        # For hyphenated / multi-word keywords the boundaries are at the edges
        pat = re.compile(
            r'\b' + re.escape(escaped_kw) + r'\b',
            re.IGNORECASE,
        )
        for m in pat.finditer(escaped):
            pos_range = range(m.start(), m.end())
            # Skip if any character in this match is already highlighted
            if occupied.isdisjoint(pos_range):
                spans.append((m.start(), m.end(), m.group()))
                occupied.update(pos_range)

    # ── 2. Build output string in one pass (no recursive replacement) ──
    spans.sort()  # sort by start position
    parts = []
    cursor = 0
    for start, end, matched_text in spans:
        parts.append(escaped[cursor:start])
        parts.append(f'<mark class="match">{matched_text}</mark>')
        cursor = end
    parts.append(escaped[cursor:])

    highlighted = "".join(parts)

    # ── 3. Newlines → <br> ──
    highlighted = highlighted.replace("\n", "<br>")

    # ── 4. Safety: strip any malformed/orphan tags that aren't valid ──
    # Only <mark …>, </mark>, and <br> are expected
    highlighted = re.sub(
        r'<(?!/?(mark\b|br\b)[^>]*>)[^>]*>',
        lambda m: _html.escape(m.group()),
        highlighted,
    )

    return highlighted


def _score_label(score):
    if score >= 80: return "Excellent Match"
    if score >= 65: return "Good Match"
    if score >= 50: return "Moderate Match"
    if score >= 35: return "Weak Match"
    return "Poor Match"


def _score_color(score):
    if score >= 80: return "#22c55e"
    if score >= 65: return "#84cc16"
    if score >= 50: return "#f59e0b"
    if score >= 35: return "#f97316"
    return "#ef4444"


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(debug=debug, host="0.0.0.0", port=port)
