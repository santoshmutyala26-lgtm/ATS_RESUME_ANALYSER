"""
Skill Analyzer — Recruiter-grade Skill Gap Intelligence & Clustering
Only surfaces real technical skills — never generic language words.
"""

import re
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict
from utils.nlp_pipeline import (
    SKILL_TAXONOMY, GENERIC_WORDS, STOP_WORDS, EXPERIENCE_ALIASES,
    classify_jd_skills, detect_role_profile, get_role_skill_weight,
    classify_jd_term, classify_keyword_category, SEMANTIC_SYNONYMS,
)


# ─────────────────────────────────────────────
# High-impact skill weights (domain-specific importance)
# Only REAL skills listed here — no generic words
# ─────────────────────────────────────────────
SKILL_IMPORTANCE = {
    # Engineering/Dev
    "python": 0.9, "javascript": 0.85, "java": 0.8, "sql": 0.88,
    "react": 0.8, "node": 0.75, "typescript": 0.8, "go": 0.75,
    "docker": 0.8, "kubernetes": 0.82, "aws": 0.85, "git": 0.75,
    "ci/cd": 0.78, "rest api": 0.75, "microservices": 0.78,
    "machine learning": 0.88, "deep learning": 0.85, "pytorch": 0.82,
    "tensorflow": 0.8, "nlp": 0.82, "data pipeline": 0.78,
    "postgresql": 0.75, "mongodb": 0.72, "redis": 0.7,
    "kafka": 0.75, "spark": 0.78, "airflow": 0.75,
    # Data / BI
    "power bi": 0.82, "tableau": 0.8, "looker": 0.72,
    "snowflake": 0.78, "bigquery": 0.78, "redshift": 0.75,
    "etl": 0.80, "dbt": 0.76, "data warehousing": 0.78,
    "data modeling": 0.78, "data visualization": 0.75,
    "data analysis": 0.78, "statistical modeling": 0.78,
    "a/b testing": 0.75, "feature engineering": 0.8,
    "spss": 0.7, "sas": 0.7, "stata": 0.68, "r": 0.75,
    "pandas": 0.8, "numpy": 0.75, "sklearn": 0.78,
    "xgboost": 0.75, "lightgbm": 0.74, "pyspark": 0.78,
    "statistics": 0.80, "hypothesis testing": 0.77, "time series analysis": 0.76,
    "business intelligence": 0.78, "database design": 0.75,
    "excel": 0.70, "google analytics": 0.68,
    # Management
    "agile": 0.7, "scrum": 0.68, "project management": 0.72,
    "leadership": 0.72, "communication": 0.65, "mentoring": 0.68,
    # Default for unlisted skills
    "_default": 0.5,
}

# ─────────────────────────────────────────────
# Skills that are worth recommending to learn
# Maps skill → explanation of WHY it matters contextually
# ─────────────────────────────────────────────
LEARNING_RATIONALE = {
    "sql": "SQL is the #1 query language for data roles — nearly all BI and analytics JDs require it.",
    "python": "Python dominates data science, analytics, and automation; most ML/DS pipelines are Python-first.",
    "power bi": "Power BI is the leading BI tool in enterprise environments and explicitly requested in this JD.",
    "tableau": "Tableau is the most common data visualization tool in BI-heavy analyst roles.",
    "machine learning": "ML is increasingly expected for senior data and analytics roles to move beyond descriptive stats.",
    "etl": "ETL proficiency is fundamental for data engineering and warehousing roles.",
    "data modeling": "Data modeling underpins database design and is critical for BI/analytics architects.",
    "snowflake": "Snowflake is the fastest-growing cloud data warehouse; commonly required in modern data stacks.",
    "dbt": "dbt (data build tool) is the standard for SQL-based data transformation in modern data stacks.",
    "spark": "Apache Spark is essential for processing large-scale datasets beyond single-machine capacity.",
    "airflow": "Apache Airflow is the leading workflow orchestration tool for data pipelines.",
    "docker": "Docker containerization is standard practice in DevOps and data engineering deployments.",
    "kubernetes": "Kubernetes is required for orchestrating containers at scale in cloud-native environments.",
    "aws": "AWS is the dominant cloud platform — S3, Redshift, Glue, and Lambda are common data tools.",
    "azure": "Azure is widely used in enterprise data ecosystems, especially with Microsoft stacks.",
    "tensorflow": "TensorFlow is a primary framework for building and deploying ML/DL models at scale.",
    "pytorch": "PyTorch is the leading research and production deep learning framework.",
    "r": "R is valued for advanced statistical analysis, especially in academia and pharma/biotech roles.",
    "spss": "SPSS is commonly used in research and social-science data analysis roles.",
    "statistics": "Strong statistics fundamentals are required for credible data analysis and modelling.",
    "a/b testing": "A/B testing is a core skill for product analytics and growth-focused data roles.",
    "data warehousing": "Data warehouse design is critical for enterprise BI and reporting infrastructure.",
    "database design": "Database design expertise is needed to build efficient, scalable data architectures.",
    "hypothesis testing": "Hypothesis testing is foundational for rigorous, evidence-based data analysis.",
    "time series analysis": "Time series skills are needed for forecasting, trend analysis, and financial modelling.",
    "agile": "Agile methodology is practiced by most modern tech and data teams.",
    "git": "Git version control is expected in every professional software and data engineering role.",
    "_default": "This skill is explicitly required in the job description and would strengthen your application.",
}


class SkillAnalyzer:
    """
    Analyzes skill coverage, detects gaps, and clusters skills.
    Only surfaces real technical skills — never generic language words.
    """

    def analyze(self, resume_doc: Dict, jd_doc: Dict) -> Dict[str, Any]:
        """
        Full recruiter-grade skill gap analysis.
        Tags each missing skill as required/optional and by role alignment.

        Fix 1: Match Resolution Layer — eliminates contradictions.
        A keyword can only be in ONE state: exact | synonym | semantic | partial | missing.
        If a keyword has a synonym/semantic match in the resume, it is NOT missing.
        """
        resume_text_lower = resume_doc["text_lower"]
        jd_text_lower = jd_doc["text_lower"]

        # Detect role context once
        role = detect_role_profile(jd_text_lower)

        # Scan taxonomy skills from raw text (most reliable)
        resume_skills = self._scan_skills_from_text(resume_text_lower)
        jd_skills = self._scan_skills_from_text(jd_text_lower)

        # Additionally pull skills detected by NLP pipeline
        resume_skills |= set(resume_doc["skills"])
        jd_skills |= set(jd_doc["skills"])

        # Extract high-confidence technical terms from JD keywords
        jd_keyword_skills = self._extract_technical_keywords(jd_doc["keywords"])
        jd_skills |= jd_keyword_skills

        # Compute per-skill JD importance weights (required vs optional)
        jd_weights = classify_jd_skills(jd_text_lower, list(jd_skills))

        # Exact skill gap (set intersection/difference)
        matched = resume_skills & jd_skills
        raw_missing = jd_skills - resume_skills

        # ── Fix 1: Match Resolution Layer ────────────────────────────────────
        # Remove from missing any skill that has a SYNONYM present in the resume.
        # A skill matched by synonym = NOT missing, NEVER penalize, NEVER recommend.
        synonym_matched: Set[str] = set()
        truly_missing: Set[str] = set()
        for skill in raw_missing:
            if self._synonym_present_in_text(skill, resume_text_lower):
                synonym_matched.add(skill)
            else:
                truly_missing.add(skill)

        # Merge synonym-matched into the matched set (they are effectively matched)
        matched = matched | synonym_matched
        missing = truly_missing
        extra = resume_skills - jd_skills

        # Build detailed skill objects
        matched_details = self._build_skill_details(
            matched, "matched", jd_text_lower, jd_weights, role
        )
        missing_details = self._build_skill_details(
            missing, "missing", jd_text_lower, jd_weights, role
        )
        missing_details = self._add_impact_and_rationale(
            missing_details, jd_text_lower, role
        )

        # Sort missing: required + high role-alignment first
        missing_details.sort(
            key=lambda x: (
                -x.get("jd_importance_weight", 0.5),
                -x.get("role_alignment_weight", 0.75),
                -x["impact_score"],
            )
        )

        # ── Fix 1: Build matched_keywords with mutual exclusion guarantee ────
        # matched_keywords are keywords confirmed in BOTH docs (exact or synonym).
        # A keyword CANNOT appear in both matched_keywords and missing_keywords.
        matched_keywords_set: Set[str] = set(matched)   # exact taxonomy matches

        for s in jd_doc["keywords"][:25]:
            s_lower = s.lower()
            # Hard block fillers — they must NEVER appear in matched or missing
            if s_lower in GENERIC_WORDS or s_lower in STOP_WORDS:
                continue
            if classify_jd_term(s_lower) == "filler":
                continue
            # Exact match in resume
            if self._in_text(s_lower, resume_text_lower):
                matched_keywords_set.add(s_lower)
            # Synonym match — counts as matched, NOT missing
            elif any(
                self._in_text(syn, resume_text_lower)
                for syn in SEMANTIC_SYNONYMS.get(s_lower, [])
            ):
                matched_keywords_set.add(s_lower)

        # Only highlight MEANINGFUL missing skills (required/moderate, role-aligned)
        # Strict exclusion: nothing in matched_keywords can be in missing_keywords
        missing_keywords = [
            s["name"] for s in missing_details
            if s["name"] not in matched_keywords_set          # Fix 1: mutual exclusion
            and not self._in_text(s["name"], resume_text_lower)
            and not self._synonym_present_in_text(s["name"], resume_text_lower)  # Fix 1
            and s.get("jd_importance_weight", 0.5) >= 0.5
            and classify_jd_term(s["name"]) != "filler"        # Fix 2: no fillers
        ]

        clusters = self._cluster_skills(resume_skills)

        return {
            "matched": matched_details,
            "missing": missing_details,
            "extra_skills": list(extra)[:10],
            "clusters": clusters,
            "matched_keywords": list(matched_keywords_set)[:40],
            "missing_keywords": list(set(missing_keywords))[:20],
            "coverage_pct": round(len(matched) / max(len(jd_skills), 1) * 100),
            "detected_role": role,
        }

    def _scan_skills_from_text(self, text: str) -> Set[str]:
        """Scan text for all skills in the taxonomy — using whole-word boundary matching."""
        found = set()
        for category, skills in SKILL_TAXONOMY.items():
            for skill in skills:
                if re.search(r"\b" + re.escape(skill) + r"\b", text):
                    found.add(skill)
        return found

    def _extract_technical_keywords(self, keywords: List[str]) -> Set[str]:
        """
        Convert JD keywords to skill-like terms.
        Uses classify_jd_term() to guarantee only 'technical' or 'soft_skill' terms
        are returned — filler words are hard-blocked.
        """
        result = set()
        all_taxonomy_skills = {
            skill for skills in SKILL_TAXONOMY.values() for skill in skills
        }
        for kw in keywords:
            kw_lower = kw.lower().strip()

            # Hard block: filler terms must NEVER appear in skill lists
            if classify_jd_term(kw_lower) == "filler":
                continue
            if len(kw_lower) <= 2:
                continue

            # Accept if it's a known taxonomy skill
            if kw_lower in all_taxonomy_skills:
                result.add(kw_lower)
            # Accept if it contains a taxonomy skill as a substring
            elif any(skill in kw_lower for skill in all_taxonomy_skills if len(skill) > 3):
                for skill in all_taxonomy_skills:
                    if len(skill) > 3 and skill in kw_lower:
                        result.add(skill)
        return result

    def _in_text(self, skill: str, text: str) -> bool:
        """Check if skill appears in text using whole-word boundary."""
        return bool(re.search(r"\b" + re.escape(skill) + r"\b", text))

    def _synonym_present_in_text(self, skill: str, text: str) -> bool:
        """
        Fix 1: Check if ANY semantic synonym of the skill is present in the resume.
        Used to prevent synonym-matched skills from appearing in missing_skills.
        """
        synonyms = SEMANTIC_SYNONYMS.get(skill.lower(), [])
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn.lower()) + r"\b", text):
                return True
        return False

    def _build_skill_details(
        self,
        skills: Set[str],
        status: str,
        jd_text: str,
        jd_weights: Dict[str, float] = None,
        role: str = "unknown",
    ) -> List[Dict]:
        """Build detailed skill objects. Filler terms are hard-blocked."""
        if jd_weights is None:
            jd_weights = {}
        details = []
        for skill in skills:
            # Triple guard: never surface filler as a skill
            if skill in GENERIC_WORDS or skill in STOP_WORDS:
                continue
            if classify_jd_term(skill) == "filler":
                continue
            category = self._get_skill_category(skill)
            context = self._find_context(skill, jd_text)
            jd_w = jd_weights.get(skill, 0.5)
            role_w = get_role_skill_weight(skill, role)

            # Human-readable importance tags
            jd_tag = (
                "required" if jd_w >= 0.75
                else "optional" if jd_w <= 0.35
                else "moderate"
            )
            role_tag = (
                "high" if role_w >= 0.9
                else "low" if role_w <= 0.55
                else "neutral"
            )

            details.append({
                "name": skill,
                "status": status,
                "category": category,
                "context": context,
                "impact_score": 0,
                "learning_advice": "",
                "jd_importance": jd_tag,
                "jd_importance_weight": jd_w,
                "role_alignment": role_tag,
                "role_alignment_weight": role_w,
            })
        return details

    def _add_impact_and_rationale(
        self, missing_skills: List[Dict], jd_text: str, role: str = "unknown"
    ) -> List[Dict]:
        """
        Estimate score improvement and add learning rationale for each missing skill.
        Impact is scaled by JD importance and role alignment — optional or off-profile
        skills get much lower impact scores.
        """
        for skill in missing_skills:
            base_importance = SKILL_IMPORTANCE.get(
                skill["name"], SKILL_IMPORTANCE["_default"]
            )
            jd_w = skill.get("jd_importance_weight", 0.5)
            role_w = skill.get("role_alignment_weight", 0.75)

            # Scale impact by both JD importance and role alignment
            raw_impact = base_importance * jd_w * role_w * 20
            skill["impact_score"] = round(min(15, max(1, raw_impact)))

            rationale = LEARNING_RATIONALE.get(
                skill["name"], LEARNING_RATIONALE["_default"]
            )
            # If skill is optional or off-profile, soften the rationale
            if skill.get("jd_importance") == "optional":
                rationale = (
                    f"[Optional] {rationale} — this skill was listed as preferred/optional "
                    f"in the JD, not a hard requirement."
                )
            elif skill.get("role_alignment") == "low":
                rationale = (
                    f"[Off-profile] {rationale} — this skill is less typical for "
                    f"this role type; focus on higher-priority skills first."
                )
            skill["learning_advice"] = rationale

        return missing_skills

    def _get_skill_category(self, skill: str) -> str:
        """Look up skill category in taxonomy."""
        for category, skills in SKILL_TAXONOMY.items():
            if skill in skills:
                return category
        return "general"

    def _find_context(self, skill: str, jd_text: str) -> str:
        """Find surrounding context where skill appears in JD."""
        match = re.search(r".{0,50}\b" + re.escape(skill) + r"\b.{0,50}", jd_text)
        if match:
            return "..." + match.group().strip() + "..."
        return ""

    def _cluster_skills(self, skills: Set[str]) -> Dict[str, List[str]]:
        """Group skills into meaningful clusters."""
        clusters = defaultdict(list)
        uncategorized = []

        for skill in sorted(skills):
            if skill in GENERIC_WORDS or skill in STOP_WORDS:
                continue
            category = self._get_skill_category(skill)
            if category != "general":
                clusters[category].append(skill)
            else:
                uncategorized.append(skill)

        if uncategorized:
            clusters["general"] = uncategorized[:15]

        return dict(clusters)
