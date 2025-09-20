import os, re, zipfile
import pdfplumber
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def extract_text_from_pdf(path):
    txt = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt.append(page.extract_text() or "")
    return "\n".join(txt)



def load_cvs_from_zip(zip_file):
    data = {}
    with zipfile.ZipFile(zip_file, "r") as z:
        pdf_files = [f for f in z.namelist() if f.lower().endswith(".pdf")]
        for fn in pdf_files:
            cv_id = os.path.splitext(os.path.basename(fn))[0]
            with z.open(fn) as pdf:
                with open(f"temp_{cv_id}.pdf", "wb") as f:
                    f.write(pdf.read())
            data[cv_id] = extract_text_from_pdf(f"temp_{cv_id}.pdf")
            os.remove(f"temp_{cv_id}.pdf")
    return data




def extract_skills(text, keywords=None):
    t = text.lower()
    found = set()
    if keywords:
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw.lower())}\b", t):
                found.add(kw.lower())
    return found



def build_tfidf_lsa(corpus_texts, max_components=100):
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    X = tfidf.fit_transform(corpus_texts)
    n_samples, n_features = X.shape
    n_comp = min(max_components, n_samples - 1, n_features - 1)
    if n_comp < 1:
        return tfidf, None, X, None
    lsa = TruncatedSVD(n_components=n_comp, random_state=42)
    X_lsa = lsa.fit_transform(X)
    return tfidf, lsa, X, X_lsa



def match_cvs(job_desc, cv_texts, recruiter_keywords=None, skill_w=0.4, tfidf_w=0.3, lsa_w=0.3):
    ids = list(cv_texts.keys())
    cvs = [cv_texts[i] for i in ids]
    if not cvs:
        return pd.DataFrame()
    req_skills = extract_skills(job_desc, recruiter_keywords)
    cv_skills = [extract_skills(t, recruiter_keywords) for t in cvs]
    all_texts = cvs + [job_desc]
    tfidf, lsa, X_tfidf, X_lsa = build_tfidf_lsa(all_texts)
    tfidf_cvs = X_tfidf[:len(cvs)]
    tfidf_job = X_tfidf[len(cvs):]
    sim_tfidf = cosine_similarity(tfidf_cvs, tfidf_job).flatten()
    sim_lsa = np.zeros(len(cvs))
    if X_lsa is not None:
        lsa_cvs = X_lsa[:len(cvs)]
        lsa_job = X_lsa[len(cvs):]
        sim_lsa = cosine_similarity(lsa_cvs, lsa_job).flatten()

    records = []
    for i, cv_id in enumerate(ids):
        skill_score = (len(cv_skills[i] & req_skills) / len(req_skills)) if req_skills else 0
        score = skill_w*skill_score + tfidf_w*sim_tfidf[i] + lsa_w*sim_lsa[i]
        records.append({
            "cv_id": cv_id,
            "skill_score": round(skill_score, 2),
            "tfidf_sim": round(sim_tfidf[i], 2),
            "lsa_sim": round(float(sim_lsa[i]), 2),
            "match_score": round(score * 100, 1)
        })
    return pd.DataFrame(records).sort_values("match_score", ascending=False).reset_index(drop=True)
