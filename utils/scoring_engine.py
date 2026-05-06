"""
Scoring Engine — Intelligent, Weighted ATS Score Computation
Weights: Keyword Match 30% | Semantic Similarity 30% | Skill Coverage 20% | Experience Relevance 20%

Classification-aware scoring pipeline:
  - JD terms classified as: technical (1.0×) | soft_skill (0.25×) | filler (0×, ignored)
  - Required vs optional skill detection via optionality language
  - Role-alignment multipliers per role family
  - Semantic synonym matching: predictive modeling ≈ machine learning
  - Experience scored by technical overlap + internships/projects, not just verbs/numbers
"""

import re
import math
from typing import Dict, Any, List, Tuple, Optional
from difflib import SequenceMatcher
from utils.nlp_pipeline import (
    GENERIC_WORDS, STOP_WORDS, EXPERIENCE_ALIASES,
    classify_jd_skills, detect_role_profile, get_role_skill_weight,
    SKILL_TAXONOMY, SEMANTIC_SYNONYMS, classify_jd_term,
)

# ── Term-type ATS weight multipliers ──────────────────────────────────────────
# These reflect how much each category of JD term should influence the ATS score.
# Filler words (motivated, positive, goal...) contribute ZERO — missing them
# must NEVER reduce the score or appear as recommendations.
TERM_TYPE_WEIGHTS = {
    "technical":  1.0,    # full ATS signal
    "soft_skill": 0.25,   # real but minimal ATS signal
    "filler":     0.0,    # zero — completely ignored
}


class ScoringEngine:
    """
    Recruiter-grade, multi-factor ATS score computation.

    Factor weights:
      keyword    30% — classification-aware keyword matching (technical > soft > filler=0)
      semantic   30% — cosine embedding similarity + synonym expansion
      skill      20% — weighted skill taxonomy coverage
      experience 20% — technical overlap + internships/projects + achievements
    """

    WEIGHTS = {
        "keyword":    0.30,
        "semantic":   0.30,
        "skill":      0.20,
        "experience": 0.20,
    }

    def compute_scores(self, resume_doc: Dict, jd_doc: Dict, sections: Dict = None) -> Dict[str, Any]:
        """Compute all sub-scores and combine into a final ATS score."""
        role = detect_role_profile(jd_doc["text_lower"])

        keyword_score, keyword_details = self._keyword_score(resume_doc, jd_doc, role)
        semantic_score                  = self._semantic_score(resume_doc, jd_doc)
        skill_score                     = self._skill_coverage_score(resume_doc, jd_doc, role)
        experience_score                = self._experience_relevance_score(resume_doc, jd_doc, role, sections)

        final = (
            keyword_score    * self.WEIGHTS["keyword"]
            + semantic_score * self.WEIGHTS["semantic"]
            + skill_score    * self.WEIGHTS["skill"]
            + experience_score * self.WEIGHTS["experience"]
        )

        # ── Fix 8: Realism floor / cap ────────────────────────────────────────
        # Cap only when BOTH keyword AND skill are genuinely poor
        if keyword_score < 15 and skill_score < 10:
            final = min(final, 40)
        elif keyword_score < 25 and skill_score < 20:
            final = min(final, 55)

        # Protect high-semantic / high-experience candidates from unfair low scores.
        # Candidates with strong semantic relevance + real experience signal
        # should never fall below a realistic floor (they have genuine relevance).
        if semantic_score >= 60 or experience_score >= 50:
            final = max(final, 35)
        if semantic_score >= 55 and experience_score >= 40:
            final = max(final, 45)   # Fix 8: strong relevance → never below 45
        if semantic_score >= 65 and experience_score >= 50:
            final = max(final, 52)   # Fix 8: very strong relevance floor

        final = max(5, min(98, round(final)))

        return {
            "final_score":      final,
            "keyword_score":    round(keyword_score),
            "semantic_score":   round(semantic_score),
            "skill_score":      round(skill_score),
            "experience_score": round(experience_score),
            "match_details":    keyword_details,
            "detected_role":    role,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Keyword Score — term-type × optionality × role-alignment × synonym
    # ─────────────────────────────────────────────────────────────────────────
    def _keyword_score(
        self, resume_doc: Dict, jd_doc: Dict, role: str
    ) -> Tuple[float, List[Dict]]:
        """
        Three-dimensional keyword weight:
          effective_weight = term_type_w × jd_importance_w × role_alignment_w

        term_type_w:
          technical  = 1.0   (tools, languages, methods, domain concepts)
          soft_skill = 0.25  (communication, teamwork — real but low ATS signal)
          filler     = 0.0   (motivated, positive, goal — ignored completely)

        When a keyword is not found literally, synonym matching gives 0.6× credit.
        """
        resume_text = resume_doc["text_lower"]
        jd_text     = jd_doc["text_lower"]

        all_taxonomy_skills = {
            s for skills in SKILL_TAXONOMY.values() for s in skills
        }
        raw_keywords = [
            kw for kw in jd_doc["keywords"][:60]
            if kw not in GENERIC_WORDS and kw not in STOP_WORDS and len(kw) > 2
        ]
        jd_skills_list = list(
            {s for s in all_taxonomy_skills
             if re.search(r"\b" + re.escape(s) + r"\b", jd_text)}
        )
        all_kw = list(dict.fromkeys(raw_keywords + jd_skills_list))

        if not all_kw:
            return 0.0, []

        jd_weights = classify_jd_skills(jd_text, all_kw)

        match_details  = []
        weighted_score = 0.0
        total_weight   = 0.0
        checked        = set()

        for kw in all_kw:
            if kw in checked:
                continue
            checked.add(kw)

            # ── Classify term type ────────────────────────────────────────────
            term_type   = classify_jd_term(kw)
            type_w      = TERM_TYPE_WEIGHTS[term_type]

            # Skip fillers entirely — they must never affect the score
            if type_w == 0.0:
                continue

            jd_w        = jd_weights.get(kw, 0.5)
            role_w      = get_role_skill_weight(kw, role)
            combined_w  = type_w * jd_w * role_w

            total_weight += combined_w

            # ── Match: exact → synonym → fuzzy ───────────────────────────────
            match_type = "none"
            confidence = 0.0
            match_credit = 0.0

            if re.search(r"\b" + re.escape(kw) + r"\b", resume_text):
                match_type   = "exact"
                confidence   = 1.0
                match_credit = 1.0

            else:
                # Semantic synonym check — before slow fuzzy match
                syn_hit = _check_synonyms(kw, resume_text)
                if syn_hit:
                    match_type   = "synonym"
                    confidence   = 0.85
                    match_credit = 0.6   # partial credit for equivalent term
                else:
                    # Fuzzy token match (last resort)
                    best_ratio = max(
                        (SequenceMatcher(None, kw, tok).ratio()
                         for tok in resume_doc["tokens"]),
                        default=0.0,
                    )
                    if best_ratio >= 0.85:
                        match_type   = "fuzzy"
                        confidence   = best_ratio
                        match_credit = 0.7
                    elif best_ratio >= 0.70:
                        match_type   = "partial"
                        confidence   = best_ratio
                        match_credit = 0.3

            weighted_score += combined_w * match_credit

            match_details.append({
                "keyword":       kw,
                "term_type":     term_type,
                "match_type":    match_type,
                "confidence":    round(confidence, 2),
                "jd_importance": (
                    "required" if jd_w >= 0.75
                    else "optional" if jd_w <= 0.35
                    else "moderate"
                ),
                "role_alignment": (
                    "high"    if role_w >= 0.9
                    else "low" if role_w <= 0.55
                    else "neutral"
                ),
                "explanation": _explain_match(kw, match_type, confidence,
                                              jd_w, role_w, term_type),
            })

        if total_weight == 0:
            return 0.0, []

        # Phrase-level bonus for multi-word technical phrases found exactly
        phrase_bonus = sum(
            0.4 for p in jd_doc["phrases"][:15]
            if (len(p.split()) > 1
                and p in resume_text
                and p not in GENERIC_WORDS
                and classify_jd_term(p) != "filler")
        )

        raw   = (weighted_score / total_weight) * 100 + phrase_bonus
        score = min(95.0, raw)

        # Sort: exact/synonym first, then by combined weight desc
        match_details.sort(
            key=lambda x: (
                0 if x["match_type"] in ("exact", "synonym") else
                1 if x["match_type"] in ("fuzzy", "partial") else 2,
                -jd_weights.get(x["keyword"], 0),
            )
        )
        return score, match_details[:35]

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Semantic Score — embedding cosine + synonym-expanded Jaccard fallback
    # ─────────────────────────────────────────────────────────────────────────
    def _semantic_score(self, resume_doc: Dict, jd_doc: Dict) -> float:
        """
        Primary: cosine similarity between sentence embeddings.
        Fallback: synonym-aware Jaccard — expands each JD token with its synonyms
                  before computing overlap, so 'predictive modeling' in JD can match
                  'machine learning' tokens in resume.
        """
        resume_emb = resume_doc.get("embedding")
        jd_emb     = jd_doc.get("embedding")

        if resume_emb and jd_emb:
            dot   = sum(a * b for a, b in zip(resume_emb, jd_emb))
            sim   = max(0.0, dot)
            score = (sim ** 0.6) * 100
            return min(95.0, score)

        # Synonym-expanded Jaccard fallback
        resume_set = set(resume_doc["tokens"]) - GENERIC_WORDS - STOP_WORDS
        jd_set     = set(jd_doc["tokens"])     - GENERIC_WORDS - STOP_WORDS

        # Expand JD set with synonyms of all JD tokens
        expanded_jd = set(jd_set)
        for tok in jd_set:
            for syn in SEMANTIC_SYNONYMS.get(tok, []):
                expanded_jd.update(syn.lower().split())

        if not expanded_jd:
            return 0.0

        intersection = resume_set & expanded_jd
        union        = resume_set | expanded_jd
        jaccard      = len(intersection) / len(union) if union else 0.0
        return min(85.0, jaccard * 300)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Skill Coverage Score — role-weighted, optionality-aware, filler-free
    # ─────────────────────────────────────────────────────────────────────────
    def _skill_coverage_score(
        self, resume_doc: Dict, jd_doc: Dict, role: str
    ) -> float:
        """
        Only taxonomy skills (technical) count toward coverage.
        Soft skills get reduced weight; filler words are excluded entirely.
        Synonym matching gives credit when an equivalent term is present.
        """
        jd_text     = jd_doc["text_lower"]
        resume_text = resume_doc["text_lower"]

        all_taxonomy_skills = {
            s for skills in SKILL_TAXONOMY.values() for s in skills
        }
        # Only include TECHNICAL taxonomy skills in the JD
        jd_skills_found = [
            s for s in all_taxonomy_skills
            if re.search(r"\b" + re.escape(s) + r"\b", jd_text)
            and classify_jd_term(s) in ("technical", "soft_skill")
        ]

        if not jd_skills_found:
            # Fallback: technical token overlap
            jd_kw_set = {
                t for t in jd_doc["tokens"]
                if t not in GENERIC_WORDS
                and t not in STOP_WORDS
                and classify_jd_term(t) != "filler"
            }
            resume_kw_set = set(resume_doc["tokens"])
            overlap = jd_kw_set & resume_kw_set
            return min(70.0, (len(overlap) / max(len(jd_kw_set), 1)) * 150)

        jd_weights = classify_jd_skills(jd_text, jd_skills_found)

        total_w   = 0.0
        matched_w = 0.0

        for skill in jd_skills_found:
            term_type  = classify_jd_term(skill)
            type_w     = TERM_TYPE_WEIGHTS.get(term_type, 0.75)
            jd_w       = jd_weights.get(skill, 0.5)
            role_w     = get_role_skill_weight(skill, role)
            eff_w      = type_w * jd_w * role_w

            total_w += eff_w

            # Exact match
            if re.search(r"\b" + re.escape(skill) + r"\b", resume_text):
                matched_w += eff_w
            # Synonym match — partial credit
            elif _check_synonyms(skill, resume_text):
                matched_w += eff_w * 0.6

        if total_w == 0:
            return 0.0

        coverage = matched_w / total_w
        score    = coverage * 100

        # Hard penalty only if REQUIRED CORE technical skills are widely missing
        required_core = [
            s for s in jd_skills_found
            if classify_jd_term(s) == "technical"
            and jd_weights.get(s, 0.5) >= 0.75
            and get_role_skill_weight(s, role) >= 0.9
        ]
        if required_core:
            matched_core = sum(
                1 for s in required_core
                if (re.search(r"\b" + re.escape(s) + r"\b", resume_text)
                    or _check_synonyms(s, resume_text))
            )
            core_coverage = matched_core / len(required_core)
            if core_coverage < 0.2:
                score *= 0.65
            elif core_coverage < 0.4:
                score *= 0.80

        return min(95.0, score)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Experience Relevance — technical overlap + internships + projects
    # ─────────────────────────────────────────────────────────────────────────
    def _experience_relevance_score(
        self, resume_doc: Dict, jd_doc: Dict, role: str, sections: Dict = None
    ) -> float:
        """
        Recruiter-grade experience scoring. Rewards:
          a) Technical skill overlap with JD (primary signal)
          b) Internships / co-ops / junior roles (practical experience)
          c) Personal / academic projects (proof of application)
          d) Quantified achievements (impact evidence)
          e) Strong action verbs (presentation quality)
          f) [Fix 5] Section-presence bonus for internship/projects/research

        Does NOT penalise candidates for:
          - Not having years of experience (junior-friendly)
          - Missing optional/filler JD words
          - Being a student / recent graduate
        """
        resume_text = resume_doc["text_lower"]
        jd_text     = jd_doc["text_lower"]

        # ── a) Technical overlap — primary signal (max 35 pts) ───────────────
        jd_technical_skills = [
            s for s in (
                {sk for skills in SKILL_TAXONOMY.values() for sk in skills}
            )
            if re.search(r"\b" + re.escape(s) + r"\b", jd_text)
            and classify_jd_term(s) == "technical"
        ]
        if jd_technical_skills:
            matched = sum(
                1 for s in jd_technical_skills
                if (re.search(r"\b" + re.escape(s) + r"\b", resume_text)
                    or _check_synonyms(s, resume_text))
            )
            tech_overlap = min(35, (matched / max(len(jd_technical_skills), 1)) * 45)
        else:
            tech_overlap = 10  # no skills in JD → neutral, don't penalise

        # ── b) Internship / placement / junior role detection (max 20 pts) ───
        internship_patterns = [
            r"\binternship\b", r"\bintern\b", r"\bco[\s\-]op\b",
            r"\bplacement\b", r"\bapprentice\b", r"\bgraduate\s+(?:role|position|analyst)\b",
            r"\bjunior\b", r"\bassociate\b", r"\bentry[\s\-]level\b",
            r"\btraineeship\b", r"\bwork\s+(?:placement|experience)\b",
            r"\bfreelance\b", r"\bcontract\b",  # Fix 5: freelance counts
        ]
        internship_hits = sum(
            4 for p in internship_patterns
            if re.search(p, resume_text)
        )
        internship_score = min(20, internship_hits)

        # ── c) Project / portfolio detection (max 20 pts) ────────────────────
        project_patterns = [
            r"\bproject\b", r"\bbuilt\b", r"\bdeveloped\b", r"\bcreated\b",
            r"\bimplemented\b", r"\bdesigned\b", r"\bdeployed\b",
            r"\bcapstone\b", r"\bthesis\b", r"\bdissertation\b",
            r"\bside\s+project\b", r"\bpersonal\s+project\b", r"\bopen[\s\-]source\b",
            r"\bgithub\b", r"\bportfolio\b", r"\bkaggle\b",
            r"\bresearch\b",  # Fix 5: research counts as work
        ]
        project_hits = sum(
            3 for p in project_patterns
            if re.search(p, resume_text)
        )
        project_score = min(20, project_hits)

        # ── Fix 5: Section-presence bonus (max 15 pts) ───────────────────────
        # If the resume explicitly has internship/projects/research SECTIONS,
        # award bonus points — this is a strong signal of hands-on experience.
        section_bonus = 0
        if sections:
            detected_sections = sections.get("sections", {})
            if detected_sections.get("experience", {}).get("present"):
                section_bonus += 8   # any form of experience section
            if detected_sections.get("projects", {}).get("present"):
                section_bonus += 5   # projects section explicitly present
        # Also scan raw text for aliases (catches non-section-header mentions)
        alias_bonus = sum(
            2 for alias in EXPERIENCE_ALIASES
            if re.search(r"\b" + re.escape(alias) + r"\b", resume_text)
        )
        section_bonus = min(15, section_bonus + min(6, alias_bonus))

        # ── d) Quantified achievements (max 15 pts) ──────────────────────────
        quantified = len(re.findall(
            r"\d+\s*%|\d+x|\$\s*\d+|\d+\s*(?:million|billion|k\b|users|customers|"
            r"models?|queries|records|datasets?)",
            resume_text
        ))
        quant_score = min(15, quantified * 3)

        # ── e) Strong action verbs (max 10 pts) ──────────────────────────────
        action_verbs = [
            "led", "built", "designed", "implemented", "architected",
            "launched", "optimized", "automated", "mentored", "drove",
            "spearheaded", "reduced", "increased", "delivered", "transformed",
            "trained", "deployed", "predicted", "modelled", "modeled",
            "analyzed", "conducted", "coordinated", "developed", "managed",
        ]
        verb_count  = sum(1 for v in action_verbs if re.search(r"\b" + v + r"\b", resume_text))
        verb_score  = min(10, verb_count * 2)

        total = tech_overlap + internship_score + project_score + section_bonus + quant_score + verb_score
        return min(95.0, total)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: semantic synonym matching
# ─────────────────────────────────────────────────────────────────────────────
def _check_synonyms(keyword: str, text: str) -> bool:
    """
    Return True if any semantic synonym of `keyword` appears in `text`.
    Used to give credit when resume says 'machine learning' for JD's 'predictive modeling'.
    """
    synonyms = SEMANTIC_SYNONYMS.get(keyword.lower(), [])
    for syn in synonyms:
        if re.search(r"\b" + re.escape(syn.lower()) + r"\b", text):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Helper: match explanation
# ─────────────────────────────────────────────────────────────────────────────
def _explain_match(
    keyword:   str,
    match_type: str,
    confidence: float,
    jd_weight:  float,
    role_weight: float,
    term_type:  str = "technical",
) -> str:
    importance = (
        "required skill"          if jd_weight >= 0.75
        else "optional/preferred" if jd_weight <= 0.35
        else "moderately important"
    )
    alignment = (
        "highly role-relevant"   if role_weight >= 0.9
        else "off-profile"       if role_weight <= 0.55
        else "role-relevant"
    )
    type_note = (
        " [soft skill — low ATS weight]" if term_type == "soft_skill" else ""
    )

    if match_type == "exact":
        return f"✓ '{keyword}' found in resume [{importance}, {alignment}]{type_note}"
    elif match_type == "synonym":
        return (
            f"~ '{keyword}' matched via equivalent term in resume "
            f"[{importance}, {alignment}] — consider using the exact phrase for full credit"
        )
    elif match_type == "fuzzy":
        return (
            f"~ '{keyword}' partially matched ({int(confidence*100)}% similar) "
            f"[{importance}] — use the exact term for full credit"
        )
    elif match_type == "partial":
        return (
            f"≈ '{keyword}' weakly matched ({int(confidence*100)}% similar) "
            f"[{importance}] — add the exact keyword"
        )
    else:
        if jd_weight <= 0.35:
            return f"○ '{keyword}' not found — optional/preferred skill, low score impact"
        elif role_weight <= 0.55:
            return f"○ '{keyword}' not found — off-profile for this role, minimal impact"
        else:
            return f"✗ '{keyword}' not found — {importance}, consider adding it"
