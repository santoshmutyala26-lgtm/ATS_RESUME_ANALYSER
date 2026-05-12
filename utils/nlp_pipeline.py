import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional
STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those', 'who', 'what', 'which', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also', 'as', 'if', 'while', 'although', 'because', 'since', 'unless', 'until', 'though', 'whether', 'both', 'either', 'neither', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'then', 'there', 'they', 'through', 'under', 'until', 'up', 'very', 'was', 'were', 'what', 'when', 'where', 'while', 'who', 'with', 'work', 'worked', 'working', 'using', 'use', 'used', 'including', 'include', 'includes', 'ensure', 'ensuring', 'responsible', 'responsibilities', 'required', 'requirements', 'must', 'able', 'ability', 'across', 'based', 'strong', 'knowledge', 'experience', 'years', 'year', 'proven', 'track', 'record', 'demonstrated', 'ability', 'good', 'excellent', 'great', 'understanding', 'understanding', 'familiarity', 'proficiency', 'preferred', 'plus', 'bonus', 'nice', 'have', 'minimum', 'least', 'plus', 'passion', 'passionate', 'excited', 'opportunity', 'join', 'team', 'looking', 'seeking', 'role', 'position'}
GENERIC_WORDS = {'analyze', 'analyzing', 'analysis', 'analyses', 'collect', 'collecting', 'collection', 'develop', 'developing', 'development', 'implement', 'implementing', 'implementation', 'provide', 'providing', 'support', 'supporting', 'manage', 'managing', 'management', 'improve', 'improving', 'improvement', 'ensure', 'ensuring', 'maintain', 'maintaining', 'review', 'reviewing', 'report', 'reporting', 'create', 'creating', 'build', 'building', 'perform', 'performing', 'conduct', 'conducting', 'identify', 'identifying', 'define', 'defining', 'assess', 'assessing', 'evaluate', 'evaluating', 'monitor', 'monitoring', 'track', 'tracking', 'coordinate', 'coordinating', 'collaborate', 'collaborating', 'communicate', 'communicating', 'present', 'presenting', 'design', 'designing', 'significant', 'various', 'multiple', 'several', 'large', 'complex', 'advanced', 'simple', 'basic', 'key', 'core', 'critical', 'important', 'relevant', 'effective', 'efficient', 'high', 'low', 'new', 'existing', 'current', 'general', 'information', 'data', 'amount', 'amounts', 'level', 'levels', 'technique', 'techniques', 'method', 'methods', 'approach', 'approaches', 'package', 'packages', 'tool', 'tools', 'process', 'processes', 'system', 'systems', 'area', 'areas', 'domain', 'field', 'sector', 'project', 'projects', 'task', 'tasks', 'result', 'results', 'output', 'outcomes', 'solution', 'solutions', 'concept', 'concepts', 'aspect', 'aspects', 'component', 'components', 'resource', 'resources', 'environment', 'environments', 'platform', 'platforms', 'application', 'applications', 'service', 'services', 'regarding', 'related', 'associated', 'etc', 'e.g', 'i.e', 'per', 'via', 'motivated', 'motivation', 'highly', 'positive', 'goal', 'goals', 'oriented', 'goal-oriented', 'conditions', 'working', 'attitude', 'mindset', 'culture', 'fast', 'paced', 'fast-paced', 'flexible', 'organized', 'dedicated', 'committed', 'reliable', 'hardworking', 'responsible', 'productive', 'diverse', 'inclusive', 'startup', 'mission', 'vision', 'values', 'opportunities', 'eager', 'quick', 'learner', 'learners', 'self', 'starter', 'ambitious', 'enthusiastic', 'dynamic', 'driven', 'energetic', 'friendly', 'collaborative', 'innovative', 'creative', 'understand', 'understands', 'understanding', 'detail', 'oriented', 'detail-oriented', 'fast-learner', 'results-oriented', 'results', 'growth', 'continuous', 'learning', 'ownership', 'accountability', 'initiative', 'curiosity', 'curious', 'open', 'minded', 'open-minded', 'willing', 'independent', 'autonomous', 'self-motivated', 'adept', 'attention', 'well-versed', 'knowledgeable', 'seasoned', 'proficient', 'comfortable', 'confident', 'skilled'}
SECTION_PATTERNS = {'experience': '(work\\s+experience|professional\\s+experience|employment|career\\s+history|experience|internship|internships?|freelance|contract\\s+work|research\\s+experience|field\\s+experience|work\\s+history)', 'education': '(education|academic\\s+background|qualifications|degrees?|certifications?)', 'skills': '(skills?|technical\\s+skills?|competencies|expertise|proficiencies)', 'projects': '(projects?|personal\\s+projects?|side\\s+projects?|portfolio|capstone)', 'summary': '(summary|objective|profile|about\\s+me|professional\\s+summary|overview)', 'achievements': '(achievements?|accomplishments?|awards?|honors?|recognition)', 'certifications': '(certifications?|licenses?|credentials?|professional\\s+development)', 'publications': '(publications?|research|papers?|articles?)', 'languages': '(languages?|language\\s+proficiency)', 'interests': '(interests?|hobbies|activities|volunteering)'}
EXPERIENCE_ALIASES = {'internship', 'internships', 'freelance', 'freelancing', 'contract work', 'work placement', 'research experience', 'field experience', 'research', 'projects'}
SKILL_TAXONOMY = {'tools': {'git', 'github', 'gitlab', 'jira', 'confluence', 'docker', 'kubernetes', 'jenkins', 'circleci', 'terraform', 'ansible', 'vagrant', 'postman', 'figma', 'sketch', 'notion', 'slack', 'trello', 'asana', 'excel', 'tableau', 'power bi', 'powerbi', 'looker', 'grafana', 'kibana', 'datadog', 'splunk', 'newrelic', 'pagerduty', 'airflow', 'dbt', 'spark', 'hadoop', 'kafka', 'rabbitmq', 'redis', 'elasticsearch', 'mongodb', 'postgresql', 'mysql', 'sqlite', 'snowflake', 'bigquery', 'redshift', 'salesforce', 'hubspot', 'zendesk', 'servicenow', 'sap', 'oracle', 'spss', 'sas', 'stata', 'matlab', 'rstudio', 'jupyter', 'databricks', 'alteryx', 'talend', 'informatica', 'ssis', 'power query', 'microsoft office', 'google sheets', 'google analytics'}, 'techniques': {'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision', 'data mining', 'statistical modeling', 'a/b testing', 'regression', 'classification', 'clustering', 'neural networks', 'reinforcement learning', 'transfer learning', 'feature engineering', 'data preprocessing', 'etl', 'data pipeline', 'ci/cd', 'agile', 'scrum', 'kanban', 'tdd', 'bdd', 'microservices', 'rest api', 'graphql', 'devops', 'mlops', 'devsecops', 'pair programming', 'code review', 'unit testing', 'integration testing', 'load testing', 'api design', 'system design', 'database design', 'data modeling', 'object-oriented programming', 'functional programming', 'data analysis', 'data visualization', 'business intelligence', 'predictive analytics', 'prescriptive analytics', 'descriptive analytics', 'hypothesis testing', 'time series analysis', 'cohort analysis', 'funnel analysis', 'root cause analysis', 'sql architecture', 'data warehousing', 'data governance', 'data quality', 'data storytelling', 'dashboard design'}, 'soft_skills': {'leadership', 'communication', 'teamwork', 'collaboration', 'problem solving', 'critical thinking', 'time management', 'project management', 'mentoring', 'presentation', 'negotiation', 'conflict resolution', 'adaptability', 'creativity', 'innovation', 'strategic thinking', 'customer focus', 'attention to detail', 'analytical thinking', 'self-motivated', 'proactive'}, 'domain_knowledge': {'finance', 'healthcare', 'e-commerce', 'saas', 'fintech', 'edtech', 'cybersecurity', 'cloud computing', 'iot', 'blockchain', 'robotics', 'supply chain', 'marketing', 'product management', 'ux', 'ui', 'data science', 'software engineering', 'web development', 'mobile development', 'game development', 'embedded systems', 'distributed systems', 'high performance computing', 'real-time systems', 'statistics', 'mathematics', 'econometrics', 'operations research', 'risk analysis', 'actuarial science'}, 'languages_frameworks': {'python', 'javascript', 'java', 'c++', 'c#', 'go', 'rust', 'ruby', 'php', 'swift', 'kotlin', 'scala', 'r', 'sql', 'nosql', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'fastapi', 'spring', 'laravel', 'rails', 'express', 'nextjs', 'nuxtjs', 'tensorflow', 'pytorch', 'keras', 'sklearn', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'd3', 'three.js', 'html', 'css', 'sass', 'typescript', 'graphql', 'rest', 'soap', 'aws', 'azure', 'gcp', 'lambda', 's3', 'ec2', 'rds', 'cloudfront', 'pyspark', 'dask', 'polars', 'xgboost', 'lightgbm', 'catboost', 'hugging face', 'langchain', 'openai', 'power bi', 'looker studio'}}
PHRASE_DECOMPOSITIONS = {'database design development': ['database design', 'data modeling', 'sql'], 'database design': ['database design', 'data modeling'], 'data analysis techniques': ['data analysis', 'statistical modeling', 'sql'], 'analytical techniques': ['statistical modeling', 'hypothesis testing', 'data analysis'], 'reporting packages': ['tableau', 'power bi', 'excel'], 'statistical packages': ['spss', 'sas', 'stata', 'r'], 'data collection': ['data pipeline', 'etl', 'data engineering'], 'information systems': ['database design', 'sql', 'erp'], 'development techniques': ['software engineering', 'agile', 'ci/cd'], 'large amounts of data': ['big data', 'spark', 'hadoop', 'sql'], 'significant amounts': []}
SEMANTIC_SYNONYMS: Dict[str, List[str]] = {'machine learning': ['ml', 'predictive modeling', 'predictive modelling', 'statistical learning', 'ai', 'artificial intelligence', 'model training', 'model development'], 'predictive modeling': ['machine learning', 'ml', 'regression', 'classification', 'forecasting', 'predictive analytics', 'churn prediction', 'demand forecasting'], 'predictive analytics': ['predictive modeling', 'machine learning', 'ml', 'forecasting', 'regression'], 'deep learning': ['neural networks', 'cnn', 'rnn', 'lstm', 'transformer', 'dl', 'tensorflow', 'pytorch', 'keras'], 'nlp': ['natural language processing', 'text analysis', 'text mining', 'sentiment analysis', 'text classification'], 'natural language processing': ['nlp', 'text analysis', 'text mining', 'sentiment analysis'], 'computer vision': ['image processing', 'image recognition', 'object detection', 'opencv', 'image classification'], 'feature engineering': ['feature selection', 'feature extraction', 'data preprocessing', 'feature analysis'], 'feature analysis': ['feature engineering', 'data preprocessing', 'exploratory analysis', 'eda'], 'data analysis': ['data analytics', 'data exploration', 'eda', 'exploratory data analysis', 'analytical', 'analytics'], 'data analytics': ['data analysis', 'analytics', 'business analytics', 'eda', 'exploratory analysis'], 'data visualization': ['visualization', 'dashboards', 'charts', 'reporting', 'tableau', 'power bi', 'plotly', 'matplotlib', 'seaborn'], 'visualization': ['data visualization', 'dashboards', 'tableau', 'power bi', 'charts', 'graphs'], 'reporting': ['dashboards', 'reports', 'data visualization', 'business intelligence', 'bi', 'power bi', 'tableau'], 'business intelligence': ['bi', 'reporting', 'dashboards', 'power bi', 'tableau', 'looker', 'data visualization'], 'etl': ['data pipeline', 'data ingestion', 'data processing', 'extract transform load', 'elt', 'airflow', 'dbt'], 'data pipeline': ['etl', 'elt', 'data engineering', 'data ingestion', 'airflow', 'workflow', 'data flow'], 'data warehousing': ['data warehouse', 'dwh', 'data modeling', 'snowflake', 'redshift', 'bigquery', 'star schema'], 'sql': ['mysql', 'postgresql', 't-sql', 'pl-sql', 'queries', 'database queries', 'postgres', 'ms sql'], 'python': ['py', 'pandas', 'numpy', 'scikit-learn', 'sklearn', 'scipy', 'python3'], 'spark': ['pyspark', 'apache spark', 'spark sql', 'distributed computing'], 'pyspark': ['spark', 'apache spark', 'spark sql'], 'power bi': ['powerbi', 'power bi desktop', 'dax', 'power query'], 'tableau': ['tableau desktop', 'tableau server', 'tableau public'], 'excel': ['microsoft excel', 'spreadsheet', 'vba', 'pivot tables', 'spreadsheets'], 'statistical modeling': ['statistics', 'regression', 'hypothesis testing', 'statistical analysis', 'inferential statistics'], 'statistics': ['statistical modeling', 'statistical analysis', 'hypothesis testing', 'regression', 'probability'], 'hypothesis testing': ['statistical testing', 'a/b testing', 'significance testing', 't-test', 'chi-square'], 'a/b testing': ['hypothesis testing', 'experimentation', 'split testing', 'experiment design'], 'communication': ['presentation', 'presenting', 'stakeholder management', 'cross-functional', 'written communication'], 'collaboration': ['teamwork', 'cross-functional', 'team player', 'partnering'], 'problem solving': ['analytical thinking', 'critical thinking', 'troubleshooting', 'root cause analysis']}
_CORE_TECHNICAL_SET = {'python', 'sql', 'r', 'java', 'javascript', 'typescript', 'go', 'rust', 'scala', 'c++', 'c#', 'php', 'swift', 'kotlin', 'ruby', 'tableau', 'power bi', 'powerbi', 'looker', 'excel', 'google sheets', 'spss', 'sas', 'stata', 'matlab', 'rstudio', 'jupyter', 'databricks', 'alteryx', 'talend', 'informatica', 'ssis', 'power query', 'tensorflow', 'pytorch', 'keras', 'sklearn', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'hugging face', 'langchain', 'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'pyspark', 'spark', 'dask', 'polars', 'postgresql', 'mysql', 'sqlite', 'snowflake', 'bigquery', 'redshift', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'etl', 'dbt', 'airflow', 'kafka', 'rabbitmq', 'hadoop', 'aws', 'azure', 'gcp', 's3', 'ec2', 'lambda', 'rds', 'git', 'github', 'gitlab', 'docker', 'kubernetes', 'jenkins', 'terraform', 'ansible', 'ci/cd', 'react', 'angular', 'vue', 'node', 'django', 'flask', 'fastapi', 'rest api', 'graphql', 'spring', 'agile', 'scrum', 'kanban'}
_TECHNICAL_CONCEPT_SET = {'machine learning', 'deep learning', 'nlp', 'natural language processing', 'computer vision', 'data mining', 'statistical modeling', 'a/b testing', 'regression', 'classification', 'clustering', 'neural networks', 'reinforcement learning', 'transfer learning', 'feature engineering', 'data preprocessing', 'data pipeline', 'data analysis', 'data analytics', 'data visualization', 'business intelligence', 'predictive analytics', 'prescriptive analytics', 'descriptive analytics', 'hypothesis testing', 'time series analysis', 'cohort analysis', 'funnel analysis', 'root cause analysis', 'data warehousing', 'data governance', 'data quality', 'data storytelling', 'dashboard design', 'database design', 'data modeling', 'statistics', 'segmentation', 'forecasting', 'exploratory data analysis', 'microservices', 'devops', 'mlops', 'system design', 'api design', 'object-oriented programming', 'functional programming', 'data science', 'reporting', 'dashboards', 'analytics', 'finance', 'healthcare', 'cybersecurity', 'cloud computing', 'supply chain', 'risk analysis', 'operations research'}
_TECHNICAL_SKILLS: set = set()
_SOFT_SKILLS: set = set()
def _build_taxonomy_sets() -> None:
    global _TECHNICAL_SKILLS, _SOFT_SKILLS
    if _TECHNICAL_SKILLS:
        return
    for cat, skills in SKILL_TAXONOMY.items():
        if cat == 'soft_skills':
            _SOFT_SKILLS.update(skills)
        else:
            _TECHNICAL_SKILLS.update(skills)
    soft_synonym_keys = {'communication', 'collaboration', 'problem solving'}
    _TECHNICAL_SKILLS.update((k for k in SEMANTIC_SYNONYMS.keys() if k not in _SOFT_SKILLS and k not in soft_synonym_keys))
def classify_keyword_category(term: str) -> str:
    _build_taxonomy_sets()
    t = term.lower().strip()
    if t in GENERIC_WORDS or t in STOP_WORDS:
        return 'filler'
    if t in _SOFT_SKILLS:
        return 'soft_skill'
    if t in _CORE_TECHNICAL_SET:
        return 'core_technical'
    if t in _TECHNICAL_CONCEPT_SET:
        return 'technical_concept'
    if t in _TECHNICAL_SKILLS:
        return 'core_technical'
    if len(t.split()) == 1 and len(t) <= 2 and (t not in _TECHNICAL_SKILLS):
        return 'filler'
    words = t.split()
    if len(words) >= 2:
        non_stop = [w for w in words if w not in STOP_WORDS]
        if non_stop and all((w in GENERIC_WORDS for w in non_stop)):
            return 'filler'
    return 'technical_concept'
def classify_jd_term(term: str) -> str:
    cat = classify_keyword_category(term)
    if cat in ('core_technical', 'technical_concept'):
        return 'technical'
    return cat
OPTIONALITY_SIGNALS = ['\\betc\\.?\\b', '\\bor\\b', '\\bpreferred\\b', '\\bnice[\\s\\-]to[\\s\\-]have\\b', '\\bbonus\\b', '\\bplus\\b', '\\badvantage\\b', '\\bdesirable\\b', '\\boptional\\b', '\\bfamiliar(?:ity)?\\b', '\\bexposure to\\b', '\\bsuch as\\b', '\\bincluding but not limited to\\b', '\\blike\\b', '\\be\\.g\\.?']
_OPTIONALITY_RE = re.compile('|'.join(OPTIONALITY_SIGNALS), re.IGNORECASE)
ROLE_SKILL_PROFILES = {'data_analyst': {'high': {'sql', 'excel', 'power bi', 'tableau', 'looker', 'python', 'r', 'data analysis', 'data visualization', 'business intelligence', 'statistical modeling', 'statistics', 'reporting', 'dashboards', 'dashboard design', 'data storytelling', 'hypothesis testing', 'a/b testing', 'cohort analysis', 'funnel analysis', 'descriptive analytics', 'predictive analytics', 'google analytics', 'snowflake', 'bigquery', 'redshift', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 'time series analysis', 'root cause analysis', 'data quality'}, 'low': {'javascript', 'typescript', 'react', 'angular', 'vue', 'node', 'docker', 'kubernetes', 'terraform', 'ci/cd', 'devops', 'microservices', 'graphql', 'rest api', 'xml'}, 'triggers': ['data analyst', 'analyst', 'business analyst', 'bi analyst', 'data reporting', 'analytics', 'reporting']}, 'data_engineer': {'high': {'python', 'sql', 'spark', 'airflow', 'kafka', 'etl', 'dbt', 'data pipeline', 'data warehousing', 'data modeling', 'snowflake', 'bigquery', 'redshift', 'postgresql', 'mysql', 'databricks', 'pyspark', 'hadoop', 'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'ci/cd', 'git'}, 'low': {'javascript', 'react', 'angular', 'vue', 'html', 'css', 'spss', 'sas', 'stata', 'excel'}, 'triggers': ['data engineer', 'etl developer', 'data platform', 'pipeline engineer', 'data infrastructure']}, 'ml_engineer': {'high': {'python', 'tensorflow', 'pytorch', 'keras', 'sklearn', 'machine learning', 'deep learning', 'nlp', 'feature engineering', 'mlops', 'xgboost', 'lightgbm', 'pyspark', 'sql', 'docker', 'aws', 'azure', 'gcp', 'git', 'pandas', 'numpy'}, 'low': {'javascript', 'react', 'angular', 'spss', 'sas', 'excel'}, 'triggers': ['machine learning', 'ml engineer', 'ai engineer', 'deep learning', 'model', 'mlops']}, 'software_engineer': {'high': {'python', 'javascript', 'java', 'c++', 'c#', 'go', 'typescript', 'react', 'node', 'django', 'flask', 'fastapi', 'spring', 'rest api', 'docker', 'kubernetes', 'ci/cd', 'git', 'microservices', 'sql', 'postgresql', 'mongodb', 'redis', 'aws', 'azure', 'system design', 'api design'}, 'low': {'spss', 'sas', 'stata', 'tableau', 'power bi', 'excel'}, 'triggers': ['software engineer', 'backend', 'frontend', 'full stack', 'full-stack', 'developer', 'sde', 'swe']}}
def classify_jd_skills(jd_text: str, jd_skills: list) -> Dict[str, float]:
    jd_lower = jd_text.lower()
    weights: Dict[str, float] = {}
    freq_map: Dict[str, int] = {}
    for skill in jd_skills:
        count = len(re.findall('\\b' + re.escape(skill.lower()) + '\\b', jd_lower))
        freq_map[skill] = count
    max_freq = max(freq_map.values(), default=1)
    for skill in jd_skills:
        count = freq_map.get(skill, 0)
        if count == 0:
            weights[skill] = 0.25
            continue
        freq_weight = 1.0 if count >= 2 else 0.75 if count == 1 else 0.5
        optional_hits = 0
        total_hits = 0
        for m in re.finditer('\\b' + re.escape(skill.lower()) + '\\b', jd_lower):
            total_hits += 1
            window_start = max(0, m.start() - 80)
            window_end = min(len(jd_lower), m.end() + 80)
            window = jd_lower[window_start:window_end]
            if _OPTIONALITY_RE.search(window):
                optional_hits += 1
        if total_hits == 0:
            opt_ratio = 0.0
        else:
            opt_ratio = optional_hits / total_hits
        if opt_ratio >= 0.75:
            final = freq_weight * 0.25
        elif opt_ratio >= 0.4:
            final = freq_weight * 0.5
        else:
            final = freq_weight * 1.0
        weights[skill] = round(min(1.0, final), 3)
    return weights
def detect_role_profile(jd_text: str) -> str:
    jd_lower = jd_text.lower()
    best_role = 'unknown'
    best_count = 0
    for role, profile in ROLE_SKILL_PROFILES.items():
        count = sum((1 for t in profile['triggers'] if t in jd_lower))
        if count > best_count:
            best_count = count
            best_role = role
    return best_role
def get_role_skill_weight(skill: str, role: str) -> float:
    if role == 'unknown' or role not in ROLE_SKILL_PROFILES:
        return 0.75
    profile = ROLE_SKILL_PROFILES[role]
    if skill in profile['high']:
        return 1.0
    if skill in profile['low']:
        return 0.5
    return 0.75
class NLPPipeline:
    def __init__(self):
        self._encoder = None
        self._use_spacy = self._try_load_spacy()
    def _try_load_spacy(self):
        try:
            import spacy
            self._spacy_nlp = spacy.load('en_core_web_sm')
            return True
        except Exception:
            return False
    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self._encoder = None
        return self._encoder
    def process(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        tokens = self._tokenize(text)
        phrases = self._extract_phrases(text)
        keywords = self._extract_keywords(tokens, phrases)
        noun_chunks = self._extract_noun_chunks(text)
        skills = self._extract_skills(text_lower)
        embedding = self._encode(text[:2048])
        return {'text': text, 'text_lower': text_lower, 'tokens': tokens, 'phrases': phrases, 'keywords': keywords, 'noun_chunks': noun_chunks, 'skills': skills, 'embedding': embedding}
    def _tokenize(self, text: str) -> List[str]:
        words = re.findall('\\b[a-zA-Z][a-zA-Z0-9+#\\-\\.]*\\b', text.lower())
        return [w for w in words if w not in STOP_WORDS and w not in GENERIC_WORDS and (len(w) > 2)]
    def _extract_phrases(self, text: str) -> List[str]:
        raw_phrases = []
        if self._use_spacy:
            doc = self._spacy_nlp(text[:5000])
            raw_phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks if len(chunk.text.split()) > 1 and len(chunk.text) > 4]
        else:
            pattern = '\\b([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)\\b'
            matches = re.findall(pattern, text)
            raw_phrases.extend([m.lower() for m in matches])
            tech_pattern = '\\b([a-zA-Z][a-zA-Z0-9]*(?:[\\s\\-\\.][a-zA-Z][a-zA-Z0-9]*)+)\\b'
            matches2 = re.findall(tech_pattern, text)
            raw_phrases.extend([m.lower() for m in matches2 if 5 < len(m) < 40])
        expanded = []
        for phrase in raw_phrases:
            decomposed = self._decompose_phrase(phrase)
            if decomposed:
                expanded.extend(decomposed)
            elif not self._is_all_generic(phrase):
                expanded.append(phrase)
        return list(set(expanded))[:120]
    def _decompose_phrase(self, phrase: str) -> List[str]:
        phrase_lower = phrase.strip().lower()
        for vague, atoms in PHRASE_DECOMPOSITIONS.items():
            if vague in phrase_lower or phrase_lower == vague:
                return atoms
        return []
    def _is_all_generic(self, phrase: str) -> bool:
        words = re.findall('[a-zA-Z]+', phrase.lower())
        if not words:
            return True
        return all((w in GENERIC_WORDS or w in STOP_WORDS for w in words))
    def _extract_noun_chunks(self, text: str) -> List[str]:
        if self._use_spacy:
            doc = self._spacy_nlp(text[:5000])
            return [chunk.text.lower() for chunk in doc.noun_chunks if not self._is_all_generic(chunk.text)]
        return self._extract_phrases(text)
    def _extract_keywords(self, tokens: List[str], phrases: List[str]) -> List[str]:
        freq = Counter(tokens)
        scored = []
        total = sum(freq.values())
        for word, count in freq.items():
            if word in GENERIC_WORDS or word in STOP_WORDS:
                continue
            tf = count / total if total > 0 else 0
            idf_proxy = 1.0 / math.log(count + 2)
            score = tf * idf_proxy
            if count >= 1 and len(word) > 3:
                scored.append((word, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keywords = [w for w, _ in scored[:60]]
        phrase_words = [p for p in phrases if 3 < len(p) < 40 and (not self._is_all_generic(p))]
        keywords.extend(phrase_words[:30])
        return list(dict.fromkeys(keywords))
    def _extract_skills(self, text_lower: str) -> List[str]:
        found = set()
        for category, skill_set in SKILL_TAXONOMY.items():
            for skill in skill_set:
                if re.search('\\b' + re.escape(skill) + '\\b', text_lower):
                    found.add(skill)
        return list(found)
    def _encode(self, text: str) -> Optional[List[float]]:
        encoder = self._get_encoder()
        if encoder is None:
            return None
        try:
            embedding = encoder.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        except Exception:
            return None
    def detect_sections(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        result = {}
        present = []
        missing = []
        critical_sections = ['experience', 'education', 'skills']
        recommended_sections = ['summary', 'projects', 'achievements', 'certifications']
        for section, pattern in SECTION_PATTERNS.items():
            if re.search(pattern, text_lower):
                match = re.search(pattern, text_lower)
                if match:
                    section_start = match.start()
                    section_text = text_lower[section_start:section_start + 1000]
                    word_count = len(section_text.split())
                    quality = 'strong' if word_count > 80 else 'moderate' if word_count > 30 else 'weak'
                    result[section] = {'present': True, 'quality': quality, 'word_count': word_count}
                    present.append(section)
            else:
                result[section] = {'present': False, 'quality': None}
                missing.append(section)
        if 'experience' in missing:
            alias_found = any((re.search('\\b' + re.escape(alias) + '\\b', text_lower) for alias in EXPERIENCE_ALIASES))
            if alias_found:
                missing.remove('experience')
                present.append('experience')
                result['experience'] = {'present': True, 'quality': 'moderate', 'word_count': 0, 'note': 'Detected via internship/projects/research section'}
        priority_missing = [s for s in critical_sections if s in missing]
        recommended_missing = [s for s in recommended_sections if s in missing]
        return {'sections': result, 'present': present, 'missing_critical': priority_missing, 'missing_recommended': recommended_missing}
    def keyword_density(self, text: str, jd_keywords: List[str]) -> Dict[str, Any]:
        text_lower = text.lower()
        word_count = len(text.split())
        density_report = {}
        filtered_keywords = [kw for kw in jd_keywords[:30] if kw not in GENERIC_WORDS and kw not in STOP_WORDS and (len(kw) > 3)][:20]
        for kw in filtered_keywords:
            count = len(re.findall('\\b' + re.escape(kw.lower()) + '\\b', text_lower))
            freq_pct = round(count / word_count * 100, 2) if word_count > 0 else 0
            optimal = 'optimal' if 0.5 <= freq_pct <= 3 else 'too_low' if freq_pct < 0.5 else 'too_high'
            density_report[kw] = {'count': count, 'frequency_pct': freq_pct, 'status': optimal, 'recommendation': _density_recommendation(kw, count, optimal)}
        return density_report
def _density_recommendation(keyword: str, count: int, status: str) -> str:
    if status == 'too_low' and count == 0:
        return f"'{keyword}' not found — add it to skills or experience sections"
    elif status == 'too_low':
        return f"'{keyword}' appears {count}x — consider adding 1-2 more mentions"
    elif status == 'too_high':
        return f"'{keyword}' appears too often — may trigger spam filters"
    return f"'{keyword}' frequency is optimal"
