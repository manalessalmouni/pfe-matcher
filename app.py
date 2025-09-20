import streamlit as st
import pandas as pd
from utils import load_cvs_from_zip, match_cvs

st.set_page_config(page_title="Matching CVs", page_icon="📂", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4a5759;'> Matching CVs </h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Paramètres")
    uploaded_zip = st.file_uploader("Upload un fichier .zip avec des CVs", type=["zip"])
    job_desc = st.text_area("Description de poste :", height=200)
    keywords = st.text_input("Compétences recherchées (séparés par virgule)", "python, sql, html, css")
    launch = st.button("Lancer le matching")

if launch:
    if uploaded_zip and job_desc.strip():
        recruiter_keywords = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        cvs = load_cvs_from_zip(uploaded_zip)
        df = match_cvs(job_desc, cvs, recruiter_keywords=recruiter_keywords)

        st.success(f"✅ {len(df)} CV analysés")

        for _, row in df.iterrows():
            st.markdown(f"""
            <div style="background:#F4F6F6; padding:15px; border-radius:10px; margin-bottom:10px; box-shadow:0 2px 5px rgba(0,0,0,0.1)">
                <h4>👤 {row['cv_id']}</h4>
                <b>Score global :</b> {row['match_score']} % <br>
                <b>Compétences :</b> {row['skill_score']} <br>
                <b>TF-IDF :</b> {row['tfidf_sim']} | <b>LSA :</b> {row['lsa_sim']}
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Résultats détaillés")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger résultats (CSV)", csv, "resultats.csv", "text/csv")
    else:
        st.error("Veuillez uploader un zip et entrer une description de poste.")





