# src/app.py
import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from dotenv import load_dotenv

# Optional LLM
import openai

load_dotenv()  # loads .env if present

# --- Config ---
st.set_page_config(layout="wide", page_title="Drug Consumption Dashboard")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Drug_Consumption.csv")

# --- Utility functions ---
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    return df

def encode_drug_cols(df, drug_cols):
    mapping = {f"CL{i}": i for i in range(7)}
    df_enc = df.copy()
    df_enc[drug_cols] = df_enc[drug_cols].replace(mapping)
    return df_enc

# Simple LLM wrapper (OpenAI ChatCompletion)
def llm_analyze(prompt: str, engine="gpt-4o-mini", max_tokens=400):
    # The user must provide OPENAI_API_KEY in environment or .env
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return "OPENAI_API_KEY not set. Set it as environment variable or in a .env file."
    openai.api_key = key
    try:
        # Using OpenAI ChatCompletion (adjust to your installed SDK version)
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # change to model you have access to
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM call error: {e}"

# --- Load Data ---
st.title("Drug Consumption — Exploratory & Multivariate Dashboard")

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Place Drug_Consumption.csv into data/")
    st.stop()

# Identify columns
drug_cols = ["Alcohol","Amphet","Amyl","Benzos","Caff","Cannabis","Choc","Coke",
             "Crack","Ecstasy","Heroin","Ketamine","Legalh","LSD","Meth","Mushrooms","Nicotine","Semer","VSA"]
traits = ["Nscore","Escore","Oscore","AScore","Cscore","Impulsive","SS"]

# Sidebar filters
st.sidebar.header("Filters")
age_sel = st.sidebar.multiselect("Age", options=df['Age'].unique(), default=list(df['Age'].unique()))
gender_sel = st.sidebar.multiselect("Gender", options=df['Gender'].unique(), default=list(df['Gender'].unique()))

df_f = df[df['Age'].isin(age_sel) & df['Gender'].isin(gender_sel)]

# Navigation
page = st.sidebar.radio("Page", ["Overview","Univariate","Bivariate","Multivariate","LLM Insights","Export"])

# --- Page: Overview ---
if page == "Overview":
    st.header("Overview")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Sample of data")
        st.dataframe(df_f.head(200))
    with col2:
        st.subheader("Counts")
        st.write("Respondents:", len(df_f))
        st.write("Countries:", df_f['Country'].nunique())
        st.write("Education levels:", df_f['Education'].nunique())

    st.markdown("### Top Drugs by recent use (daily/weekly/monthly)")
    # compute % with CL4-CL6
    recent_use = {}
    for c in drug_cols:
        vals = df_f[c].value_counts(normalize=True)
        recent = sum(vals.get(x,0) for x in ["CL4","CL5","CL6"])
        recent_use[c] = recent
    ru = pd.Series(recent_use).sort_values(ascending=False)
    fig = px.bar(ru.head(10), labels={'index':'Drug','value':'Proportion recent use'}, title="Top 10 Drugs: Proportion used in last month/week/day")
    st.plotly_chart(fig, use_container_width=True)

# --- Page: Univariate ---
elif page == "Univariate":
    st.header("Univariate Analysis")
    st.subheader("Demographics")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Age distribution")
        fig = px.histogram(df_f, x="Age", category_orders={"Age": sorted(df_f['Age'].unique())})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.write("Gender")
        fig = px.pie(df_f, names="Gender")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Personality traits")
    trait = st.selectbox("Select trait", traits)
    fig = px.histogram(df_f, x=trait, nbins=30, marginal="box")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Drug distributions (counts)")
    drug = st.selectbox("Select drug", drug_cols)
    order = ["CL0","CL1","CL2","CL3","CL4","CL5","CL6"]
    ct = df_f[drug].value_counts().reindex(order).fillna(0)
    fig = px.bar(ct, labels={'index':drug,'value':'count'})
    st.plotly_chart(fig, use_container_width=True)

# --- Page: Bivariate ---
elif page == "Bivariate":
    st.header("Bivariate Analysis")
    col1, col2 = st.columns(2)
    with col1:
        x = st.selectbox("X variable (categorical)", ["Age","Gender","Education"])
    with col2:
        y_drug = st.selectbox("Drug (categorical)", drug_cols)
    st.write("Countplot of", y_drug, "by", x)
    fig = px.histogram(df_f, x=x, color=y_drug, barmode='group', category_orders={y_drug:["CL0","CL1","CL2","CL3","CL4","CL5","CL6"]})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Trait vs Drug (boxplot)")
    trait = st.selectbox("Trait for boxplot", traits, key="bivar_trait")
    fig2 = px.box(df_f, x=y_drug, y=trait, category_orders={y_drug:["CL0","CL1","CL2","CL3","CL4","CL5","CL6"]})
    st.plotly_chart(fig2, use_container_width=True)

# --- Page: Multivariate ---
elif page == "Multivariate":
    st.header("Multivariate Analysis")
    st.markdown("### PCA on personality traits (colored by a selected drug)")
    drug_for_color = st.selectbox("Color by drug", drug_cols, index=5)
    df_enc = encode_drug_cols(df_f, drug_cols)
    X = df_enc[traits].dropna()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(Xs)
    df_pca = pd.DataFrame(pcs, columns=["PC1","PC2"])
    # align index
    df_pca[drug_for_color] = df_enc.loc[X.index, drug_for_color].astype(str).values
    fig = px.scatter(df_pca, x="PC1", y="PC2", color=drug_for_color, title=f"PCA colored by {drug_for_color}")
    st.plotly_chart(fig, use_container_width=True)
    st.write("Explained variance:", pca.explained_variance_ratio_)

    st.markdown("### Clustering (KMeans) on traits + drug encoding")
    n_clusters = st.slider("Number of clusters", 2, 8, 4)
    Xk = pd.concat([df_enc[traits], df_enc[drug_cols]], axis=1).dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xk)
    Xk["cluster"] = kmeans.labels_
    fig2 = px.scatter(Xk, x="Nscore", y="Impulsive", color="cluster", title="Nscore vs Impulsive colored by cluster")
    st.plotly_chart(fig2, use_container_width=True)

# --- Page: LLM Insights ---
elif page == "LLM Insights":
    st.header("LLM-driven Insights")

    st.markdown("You can ask the LLM a question about the dataset. Examples:")
    st.write("- `Summarize the risk factors associated with cannabis use in this dataset.`")
    st.write("- `Give me a short list of top 5 personality features associated with high nicotine use.`")
    st.write("- `Create a one paragraph summary of drug usage patterns for 18-24 year old males.`")

    user_q = st.text_area("Enter prompt for LLM", value="Summarize the key findings about cannabis users vs non-users in plain English (3-5 bullet points).")
    max_tokens = st.slider("Max tokens for response", 100, 1000, 300)

    if st.button("Ask LLM"):
        # prepare a compact context to send (small sample + statistics)
        # Build a short context: counts / means
        df_enc = encode_drug_cols(df_f, drug_cols)
        # get basic stats for cannabis
        def short_stats_for(drug):
            counts = df[drug].value_counts(normalize=True).to_dict()
            means = df_enc[traits].mean().to_dict()
            return counts, means
        counts, means = short_stats_for("Cannabis")
        # small context (very limited token usage)
        context = f"Dataset: {len(df_f)} respondents. Alcohol daily/weekly proportion: {df_f['Alcohol'].isin(['CL5','CL6']).mean():.2f}. " \
                  f"Cannabis recent use (month/week/day): {df_f['Cannabis'].isin(['CL4','CL5','CL6']).mean():.2f}. " \
                  f"Averages: " + ", ".join([f"{k}:{v:.2f}" for k,v in df_enc[traits].mean().to_dict().items()]) + "."
        prompt = f"{context}\n\nUser question:\n{user_q}\n\nAnswer concisely and in bullet points."
        with st.spinner("Querying LLM..."):
            reply = llm_analyze(prompt, max_tokens=max_tokens)
        st.markdown("**LLM Response**")
        st.write(reply)

# --- Page: Export ---
elif page == "Export":
    st.header("Export Data / Models")
    st.write("You can export filtered data to CSV or download PCA/clustering labels.")

    csv = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered data (.csv)", csv, file_name="drug_filtered.csv", mime="text/csv")

    st.write("Done.")
