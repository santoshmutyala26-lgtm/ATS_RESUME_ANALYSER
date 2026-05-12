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
    return (nlp_pipeline, scoring_engine, skill_analyzer, optimizer)
@app.route('/')
def intro():
    return render_template('intro.html')
@app.route('/analyzer')
def analyzer():
    return render_template('index.html')
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'resume' not in request.files:
            return (jsonify({'error': 'No resume file provided'}), 400)
        if 'job_description' not in request.form:
            return (jsonify({'error': 'No job description provided'}), 400)
        resume_file = request.files['resume']
        job_description = request.form['job_description'].strip()
        if not resume_file.filename.endswith('.pdf'):
            return (jsonify({'error': 'Resume must be a PDF file'}), 400)
        if len(job_description) < 50:
            return (jsonify({'error': 'Job description is too short'}), 400)
        resume_bytes = resume_file.read()
        resume_text = extract_text_from_pdf(resume_bytes)
        if not resume_text or len(resume_text) < 100:
            return (jsonify({'error': "Could not extract text from PDF. Ensure it's not scanned/image-only."}), 400)
        pipeline, scorer, skill_ana, opt = get_pipeline()
        resume_doc = pipeline.process(resume_text)
        jd_doc = pipeline.process(job_description)
        sections = pipeline.detect_sections(resume_text)
        scores = scorer.compute_scores(resume_doc, jd_doc, sections)
        skill_report = skill_ana.analyze(resume_doc, jd_doc)
        keyword_density = pipeline.keyword_density(resume_text, jd_doc['keywords'])
        suggestions = opt.generate_suggestions(resume_doc, jd_doc, skill_report, sections)
        response = {'ats_score': scores['final_score'], 'score_breakdown': {'keyword_match': scores['keyword_score'], 'semantic_similarity': scores['semantic_score'], 'skill_coverage': scores['skill_score'], 'experience_relevance': scores['experience_score']}, 'weights': {'keyword_match': 30, 'semantic_similarity': 30, 'skill_coverage': 20, 'experience_relevance': 20}, 'matched_skills': skill_report['matched'], 'missing_skills': skill_report['missing'], 'skill_clusters': skill_report['clusters'], 'keyword_density': keyword_density, 'sections': sections, 'suggestions': suggestions, 'missing_keyword_advice': suggestions.get('missing_keyword_advice', []), 'match_details': scores['match_details'], 'keyword_match_log': _build_keyword_match_log(scores['match_details'], skill_report['matched_keywords']), 'resume_preview': {'text': resume_text[:3000], 'highlighted': _build_highlighted(resume_text[:3000], skill_report['matched_keywords'], skill_report['missing_keywords'])}, 'score_label': _score_label(scores['final_score']), 'score_color': _score_color(scores['final_score']), 'detected_role': scores.get('detected_role', skill_report.get('detected_role', 'unknown'))}
        return jsonify(response)
    except Exception as e:
        app.logger.error(f'Analysis error: {e}', exc_info=True)
        return (jsonify({'error': f'Analysis failed: {str(e)}'}), 500)
def _build_keyword_match_log(match_details: list, matched_keywords: list) -> list:
    log = []
    seen = set()
    for m in match_details:
        if m['match_type'] in ('exact', 'synonym', 'fuzzy', 'partial'):
            kw = m['keyword'].lower()
            if kw not in seen:
                seen.add(kw)
                log.append({'keyword': m['keyword'], 'match_type': m['match_type'], 'confidence': m['confidence'], 'jd_importance': m.get('jd_importance', 'moderate'), 'role_alignment': m.get('role_alignment', 'neutral'), 'explanation': m.get('explanation', ''), 'source': 'keyword_match'})
    for kw in matched_keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            log.append({'keyword': kw, 'match_type': 'exact', 'confidence': 1.0, 'jd_importance': 'moderate', 'role_alignment': 'neutral', 'explanation': f"'{kw}' found in both resume and job description", 'source': 'skill_match'})
    log.sort(key=lambda x: (x['match_type'] != 'exact', x['keyword'].lower()))
    return log
def _build_highlighted(text, matched_kw, missing_kw):
    import html as _html
    escaped = _html.escape(text)
    sorted_kws = sorted(set(matched_kw), key=len, reverse=True)
    spans = []
    occupied = set()
    for kw in sorted_kws:
        escaped_kw = _html.escape(kw)
        pat = re.compile('\\b' + re.escape(escaped_kw) + '\\b', re.IGNORECASE)
        for m in pat.finditer(escaped):
            pos_range = range(m.start(), m.end())
            if occupied.isdisjoint(pos_range):
                spans.append((m.start(), m.end(), m.group()))
                occupied.update(pos_range)
    spans.sort()
    parts = []
    cursor = 0
    for start, end, matched_text in spans:
        parts.append(escaped[cursor:start])
        parts.append(f'<mark class="match">{matched_text}</mark>')
        cursor = end
    parts.append(escaped[cursor:])
    highlighted = ''.join(parts)
    highlighted = highlighted.replace('\n', '<br>')
    highlighted = re.sub('<(?!/?(mark\\b|br\\b)[^>]*>)[^>]*>', lambda m: _html.escape(m.group()), highlighted)
    return highlighted
def _score_label(score):
    if score >= 80:
        return 'Excellent Match'
    if score >= 65:
        return 'Good Match'
    if score >= 50:
        return 'Moderate Match'
    if score >= 35:
        return 'Weak Match'
    return 'Poor Match'
def _score_color(score):
    if score >= 80:
        return '#22c55e'
    if score >= 65:
        return '#84cc16'
    if score >= 50:
        return '#f59e0b'
    if score >= 35:
        return '#f97316'
    return '#ef4444'
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(debug=debug, host='0.0.0.0', port=port)
