import re
from typing import Dict, List, Any
from utils.nlp_pipeline import GENERIC_WORDS, STOP_WORDS, classify_jd_term
NEVER_RECOMMEND: set = {'highly', 'motivated', 'conditions', 'adept', 'attention', 'positive', 'goal', 'goals', 'mindset', 'attitude', 'passionate', 'driven', 'dynamic', 'energetic', 'enthusiastic', 'ambitious', 'dedicated', 'reliable', 'flexible', 'organized', 'hardworking', 'innovative', 'creative', 'fast', 'paced', 'agility', 'self', 'starter', 'growth', 'learning', 'ownership', 'accountability', 'initiative', 'curiosity', 'curious', 'willing', 'eager', 'proactive', 'autonomous', 'independent', 'ability', 'skills', 'knowledge', 'experience', 'understanding', 'proficiency', 'familiarity', 'competency', 'expertise'} | GENERIC_WORDS | STOP_WORDS
WEAK_PATTERNS = [('\\bworked on\\b', 'worked on'), ('\\bhelped with\\b', 'helped with'), ('\\bassisted in\\b', 'assisted in'), ('\\bresponsible for\\b', 'responsible for'), ('\\binvolved in\\b', 'involved in'), ('\\bpart of\\b', 'part of'), ('\\bdid\\b', 'did'), ('\\bwas\\s+in\\s+charge\\b', 'was in charge')]
VERB_UPGRADES = {'worked on': 'developed', 'helped with': 'contributed to', 'assisted in': 'supported', 'responsible for': 'owned', 'involved in': 'spearheaded', 'was in charge': 'led', 'did': 'executed', 'made': 'engineered', 'used': 'leveraged', 'showed': 'demonstrated', 'fixed': 'resolved', 'wrote': 'authored', 'got': 'achieved', 'tried to': 'successfully'}
SECTION_ADVICE = {'summary': {'missing': 'Add a professional summary (3-4 lines) that mirrors the job title and 2-3 core qualifications from the JD. This is the first thing ATS and recruiters read.', 'weak': 'Strengthen your summary by including: target role title, years of experience, 2-3 top skills that match the JD, and a quantified achievement.'}, 'skills': {'missing': 'Add a dedicated Skills section. ATS systems specifically scan for this section. List technical skills, tools, and certifications in a clean format.', 'weak': 'Expand your Skills section to cover more technical keywords from the job description. Group by category: Languages, Frameworks, Tools, Databases.'}, 'experience': {'missing': "Your Experience section is missing or not detected. Ensure it has a clear header ('Work Experience' or 'Professional Experience').", 'weak': 'Improve bullet points with: strong action verbs, quantified results (%, $, x), and specific technologies mentioned in the JD.'}, 'projects': {'missing': 'Add a Projects section highlighting relevant work. Include: project name, tech stack used, your role, and measurable outcomes.', 'weak': 'Enhance project descriptions with: technologies used (matching JD keywords), your specific contribution, and impact metrics.'}, 'achievements': {'missing': 'Consider adding an Achievements section with 2-3 standout wins — quantified awards, recognitions, or record-breaking results.', 'weak': "Make achievements more impactful by adding numbers: 'Increased revenue by $2M' beats 'Increased revenue'."}, 'certifications': {'missing': 'If you have relevant certifications (AWS, Google Cloud, PMP, etc.), add a Certifications section — it significantly boosts ATS scores.', 'weak': 'Expand certifications with: certification authority, date earned, and credential ID where applicable.'}}
REWRITE_TEMPLATES = [{'pattern': '(analyzed|studied|reviewed).{0,20}(data|metrics|reports|trends)', 'improvement': 'Replace passive phrasing with an action-result structure: specify WHAT you analyzed, the method/tool used, and the outcome or business impact.', 'example_structure': 'Analyzed [dataset/metrics] using [tool/method] → revealed [insight], enabling [outcome]', 'category': 'analysis'}, {'pattern': '(built|developed|created|wrote).{0,20}(website|web app|frontend|ui)', 'improvement': 'Quantify impact and specify the tech stack: how many users, what performance gains, which frameworks?', 'example_structure': 'Developed [type of app] using [tech stack] → served [N] users / reduced [metric] by [X%]', 'category': 'frontend'}, {'pattern': '(managed|ran|led).{0,20}(team|project|group)', 'improvement': 'Add team size, project scope, delivery outcome, and any cross-functional collaboration.', 'example_structure': 'Led [N]-person [team type] to deliver [project/feature] on schedule with [outcome/metric]', 'category': 'management'}, {'pattern': '(worked|helped).{0,20}(machine learning|ml|ai|model)', 'improvement': 'Specify your exact contribution: did you design the model, tune hyperparameters, deploy it? Add accuracy/performance metrics.', 'example_structure': 'Built/deployed [model type] achieving [accuracy/F1] → reduced [manual effort/error] by [X%]', 'category': 'ml'}, {'pattern': '(fixed|resolved|debugged).{0,20}(bug|issue|problem|error)', 'improvement': 'Quantify the scale: how many bugs, what was the user impact, and how much did it improve stability or user experience?', 'example_structure': 'Resolved [N] [critical/high-priority] bugs → improved uptime to [X%] / reduced [metric] by [X]', 'category': 'engineering'}, {'pattern': '(wrote|created|developed).{0,20}(api|endpoint|service|backend)', 'improvement': 'Describe scale (requests/day), reliability (uptime %), and the tech stack used.', 'example_structure': 'Designed [REST/GraphQL] API handling [N] requests/day with [X%] uptime using [tech stack]', 'category': 'backend'}, {'pattern': '(improved|optimized|enhanced).{0,20}(performance|speed|efficiency)', 'improvement': 'Always include the baseline, the improvement method, and the measurable result.', 'example_structure': 'Optimized [process/query] using [method] → reduced [metric] from [X] to [Y] ([%] improvement)', 'category': 'optimization'}, {'pattern': 'worked on (data|database|sql|analytics)', 'improvement': 'Be specific about the data size, the tools used, and the business outcome of the work.', 'example_structure': 'Processed/cleaned [N] records using [tool/SQL] → improved [data quality/accuracy] by [X%]', 'category': 'data'}]
class ResumeOptimizer:
    def generate_suggestions(self, resume_doc: Dict, jd_doc: Dict, skill_report: Dict, sections: Dict) -> Dict[str, Any]:
        return {'bullet_rewrites': self._suggest_bullet_rewrites(resume_doc['text']), 'missing_keyword_advice': self._suggest_learning_advice(skill_report['missing'], jd_doc), 'section_improvements': self._suggest_section_improvements(sections), 'quick_wins': self._quick_wins(skill_report['missing'][:5], sections, resume_doc), 'ats_optimization': self._ats_optimization_mode(skill_report['missing'], jd_doc, sections, resume_doc)}
    def _suggest_bullet_rewrites(self, resume_text: str) -> List[Dict]:
        rewrites = []
        lines = resume_text.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 20 or len(line) > 300:
                continue
            for pattern, phrase in WEAK_PATTERNS:
                if re.search(pattern, line.lower()):
                    for template in REWRITE_TEMPLATES:
                        if re.search(template['pattern'], line.lower()):
                            rewrites.append({'original': line[:150], 'improvement_tip': template['improvement'], 'example_structure': template['example_structure'], 'category': template['category']})
                            break
                    else:
                        improved_line = self._generic_rewrite(line)
                        if improved_line != line:
                            rewrites.append({'original': line[:150], 'improvement_tip': f"Replace '{phrase}' with a stronger action verb and add a quantified outcome (%, $, time saved).", 'example_structure': improved_line[:200], 'category': 'general'})
                    break
        return rewrites[:6]
    def _generic_rewrite(self, line: str) -> str:
        result = line
        for weak, strong in VERB_UPGRADES.items():
            pattern = '\\b' + re.escape(weak) + '\\b'
            result = re.sub(pattern, strong, result, flags=re.IGNORECASE, count=1)
        if result != line and (not re.search('\\d+%|\\d+x|\\$\\d+', result)):
            result += ' (add quantified impact: %, $, or time saved)'
        return result
    def _suggest_learning_advice(self, missing_skills: List[Dict], jd_doc: Dict) -> List[Dict]:
        suggestions = []
        jd_text_lower = jd_doc.get('text_lower', '')
        role_context = self._infer_role_context(jd_text_lower)
        for skill_obj in missing_skills[:12]:
            skill = skill_obj['name']
            if skill in NEVER_RECOMMEND:
                continue
            if classify_jd_term(skill) == 'filler':
                continue
            jd_importance = skill_obj.get('jd_importance', 'moderate')
            role_alignment = skill_obj.get('role_alignment', 'neutral')
            if jd_importance == 'optional' and role_alignment == 'low':
                continue
            if role_alignment == 'low' and jd_importance != 'required':
                continue
            category = skill_obj.get('category', 'general')
            context = skill_obj.get('context', '')
            learning_advice = skill_obj.get('learning_advice', '')
            impact = skill_obj.get('impact_score', 5)
            if jd_importance == 'optional':
                action_msg = f'This is an optional/preferred skill in the JD. Consider learning {skill} if it aligns with your career direction.'
            elif role_alignment == 'low':
                action_msg = f"{skill} is off-profile for {role_context} roles. Consider it only if you're pivoting toward engineering."
            else:
                action_msg = f'Consider learning {skill} if targeting {role_context} roles.'
            suggestions.append({'skill': skill, 'why_it_matters': learning_advice, 'role_context': action_msg, 'where_to_add': _suggest_placement(category), 'context_in_jd': context[:120] if context else 'Mentioned as a key requirement in the JD', 'estimated_score_boost': f'+{impact}%', 'jd_importance': jd_importance, 'role_alignment': role_alignment, 'action': 'learn_and_add'})
            if len(suggestions) >= 8:
                break
        return suggestions
    def _infer_role_context(self, jd_text: str) -> str:
        if any((kw in jd_text for kw in ['data engineer', 'etl', 'pipeline', 'warehouse'])):
            return 'data engineering or ETL'
        if any((kw in jd_text for kw in ['machine learning', 'deep learning', 'model', 'pytorch', 'tensorflow'])):
            return 'machine learning or AI'
        if any((kw in jd_text for kw in ['power bi', 'tableau', 'dashboard', 'business intelligence', 'bi analyst'])):
            return 'BI or data analytics'
        if any((kw in jd_text for kw in ['data analyst', 'analytics', 'reporting', 'sql'])):
            return 'data analytics or reporting'
        if any((kw in jd_text for kw in ['frontend', 'react', 'angular', 'vue', 'ui/ux'])):
            return 'frontend or full-stack development'
        if any((kw in jd_text for kw in ['backend', 'api', 'microservices', 'django', 'node'])):
            return 'backend engineering'
        if any((kw in jd_text for kw in ['devops', 'kubernetes', 'docker', 'terraform', 'ci/cd'])):
            return 'DevOps or cloud engineering'
        return 'this type of'
    def _suggest_section_improvements(self, sections: Dict) -> List[Dict]:
        improvements = []
        for section in sections.get('missing_critical', []):
            if section in SECTION_ADVICE:
                improvements.append({'section': section, 'priority': 'critical', 'action': 'add', 'advice': SECTION_ADVICE[section]['missing']})
        for section in sections.get('missing_recommended', []):
            if section in SECTION_ADVICE:
                improvements.append({'section': section, 'priority': 'recommended', 'action': 'add', 'advice': SECTION_ADVICE[section]['missing']})
        for section, data in sections.get('sections', {}).items():
            if data.get('quality') == 'weak' and data.get('present') and (section in SECTION_ADVICE):
                improvements.append({'section': section, 'priority': 'improve', 'action': 'strengthen', 'advice': SECTION_ADVICE[section].get('weak', '')})
        return improvements[:8]
    def _quick_wins(self, top_missing: List[Dict], sections: Dict, resume_doc: Dict) -> List[str]:
        wins = []
        for s in sections.get('missing_critical', [])[:2]:
            wins.append(f"Add a '{s.title()}' section — ATS systems specifically look for this header")
        for skill_obj in top_missing[:5]:
            skill = skill_obj['name']
            if skill in NEVER_RECOMMEND:
                continue
            if classify_jd_term(skill) == 'filler':
                continue
            jd_imp = skill_obj.get('jd_importance', 'moderate')
            role_al = skill_obj.get('role_alignment', 'neutral')
            if jd_imp == 'optional' and role_al != 'high':
                continue
            if role_al == 'low':
                continue
            wins.append(f"Add '{skill}' to your Skills section if you have experience with it — could improve score by +{skill_obj.get('impact_score', 5)}%")
            if len(wins) >= 3:
                break
        quant = len(re.findall('\\d+%|\\d+x|\\$\\d+', resume_doc['text_lower']))
        if quant < 3:
            wins.append('Add quantified achievements (%, $, time saved) — resumes with numbers get 40% more recruiter callbacks')
        wins.append('Start each bullet with a past-tense action verb: Led, Built, Designed, Optimized')
        wins.append('Avoid tables, columns, and graphics — ATS parsers often skip text in complex layouts')
        return wins[:6]
    def _ats_optimization_mode(self, missing_skills: List[Dict], jd_doc: Dict, sections: Dict, resume_doc: Dict) -> Dict[str, Any]:
        top_keywords = [s['name'] for s in missing_skills[:10] if s['name'] not in GENERIC_WORDS and s['name'] not in STOP_WORDS]
        learning_items = []
        jd_text_lower = jd_doc.get('text_lower', '')
        role_context = self._infer_role_context(jd_text_lower)
        for skill_obj in missing_skills[:6]:
            skill = skill_obj['name']
            if skill in NEVER_RECOMMEND:
                continue
            if classify_jd_term(skill) == 'filler':
                continue
            learning_advice = skill_obj.get('learning_advice', LEARNING_RATIONALE_DEFAULT)
            learning_items.append({'skill': skill, 'why': learning_advice, 'suggestion': f'Consider learning {skill} if targeting {role_context} roles.'})
        section_actions = []
        for s in sections.get('missing_critical', []):
            section_actions.append(f"CRITICAL: Add '{s.title()}' section")
        for s in sections.get('missing_recommended', []):
            section_actions.append(f"RECOMMENDED: Add '{s.title()}' section")
        return {'missing_keywords': top_keywords, 'learning_recommendations': learning_items, 'section_actions': section_actions, 'formatting_tips': ["Use standard section headers: 'Work Experience', 'Education', 'Skills'", 'Save as .docx or simple .pdf — avoid creative templates with columns', 'Use standard fonts: Calibri, Arial, Times New Roman (10-12pt)', 'Keep to 1-2 pages maximum', "No tables, text boxes, or headers/footers (ATS can't read these)"]}
LEARNING_RATIONALE_DEFAULT = 'This skill is explicitly required in the job description and would strengthen your application.'
def _suggest_placement(category: str) -> str:
    placements = {'tools': 'Skills section or within a relevant job bullet point', 'techniques': 'Experience bullet points or Summary section', 'soft_skills': 'Summary section or as part of an achievement bullet', 'domain_knowledge': 'Summary section or Skills section', 'languages_frameworks': "Skills section under 'Technologies' or 'Languages'", 'general': 'Skills or Experience section'}
    return placements.get(category, placements['general'])
