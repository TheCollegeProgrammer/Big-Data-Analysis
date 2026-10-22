import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Spotify Data Lab", page_icon=None, layout="wide", initial_sidebar_state="expanded")

OUTPUT_DIR = "spotify_model_outputs"
CSV_PATH = os.path.join(OUTPUT_DIR, "processed_songs.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "popularity_model.pkl")
FEATURES_PATH = os.path.join(OUTPUT_DIR, "model_features.pkl")
MEDIANS_PATH = os.path.join(OUTPUT_DIR, "audio_feature_medians.pkl")

AUDIO_CANDIDATES = [
    "danceability", "energy", "key", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms", "mode", "time_signature"
]
ENGAGEMENT_INPUTS = ["likes", "comments"]
ORANGE = "#FF6A00"
BG = "#090909"
PANEL = "#171717"
TEXT = "#F3F4F6"
MUTED = "#9CA3AF"
GRID = "#333333"

st.markdown(f"""
<style>
:root {{ color-scheme: dark; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: rgba(9,9,9,0.96); }}
[data-testid="stSidebar"] {{ background: #111111; border-right: 1px solid #292929; }}
.block-container {{ max-width: 1440px; padding-top: 1.5rem; padding-bottom: 2.5rem; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.025em; }}
h1 {{ font-size: 2.05rem !important; font-weight: 750 !important; }}
h2 {{ font-size: 1.35rem !important; font-weight: 700 !important; }}
p, label, [data-testid="stMarkdownContainer"] {{ color: {TEXT}; }}
small, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}
[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #2A2A2A; padding: 16px 18px; border-radius: 12px; }}
[data-testid="stMetricLabel"] {{ color: {MUTED} !important; }}
[data-testid="stMetricValue"] {{ color: {TEXT} !important; }}
.stButton > button, .stDownloadButton > button {{ background: {ORANGE}; color: #111111; border: 1px solid {ORANGE}; border-radius: 8px; font-weight: 700; min-height: 2.65rem; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: #FF8533; border-color: #FF8533; color: #111111; }}
div[data-baseweb="tab-list"] {{ gap: 8px; border-bottom: 1px solid #303030; }}
button[data-baseweb="tab"] {{ color: #B8B8B8; padding: 12px 14px; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {ORANGE}; border-bottom-color: {ORANGE}; }}
[data-testid="stDataFrame"] {{ border: 1px solid #303030; border-radius: 10px; overflow: hidden; }}
div[data-baseweb="select"] > div, .stTextInput input {{ background: #171717; border-color: #383838; }}
hr {{ border-color: #303030; }}
.hero {{ padding: 1.15rem 1.4rem; border: 1px solid #303030; border-left: 4px solid {ORANGE}; border-radius: 12px; background: linear-gradient(110deg, #171717 0%, #101010 70%); margin-bottom: 1.2rem; }}
.brand {{ color: {ORANGE}; font-size: .78rem; letter-spacing: .19em; font-weight: 800; }}
.hero-sub {{ color: {MUTED}; margin-top: .35rem; font-size: .98rem; }}
.section-note {{ color: {MUTED}; font-size: .9rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="brand">SPOTIFY DATA LAB</div>
  <h1 style="margin:.35rem 0 0 0;">Song Popularity & Recommendation</h1>
  <div class="hero-sub">Explore listening signals, compare audio characteristics, and discover songs by artist.</div>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH), f"Processed CSV: {CSV_PATH}"
    return pd.DataFrame(), "Processed CSV not found"

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH)):
        return None, None, {}
    with open(MODEL_PATH, "rb") as f:
        saved_model = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        saved_features = pickle.load(f)
    medians = {}
    if os.path.exists(MEDIANS_PATH):
        with open(MEDIANS_PATH, "rb") as f:
            medians = pickle.load(f)
    return saved_model, saved_features, medians

df, data_source = load_data()
model, model_features, saved_medians = load_model_artifacts()
if df.empty:
    st.error("Processed song data was not found. Run the notebook's final save-results cell first.")
    st.code("python -m streamlit run app.py")
    st.stop()

df.columns = [str(c).strip().lower() for c in df.columns]
for c in AUDIO_CANDIDATES + ENGAGEMENT_INPUTS + ["popularity_score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

track_col = next((c for c in ["track", "title", "track_name"] if c in df.columns), None)
artist_col = next((c for c in ["artist", "artist_name"] if c in df.columns), None)
audio_features = [c for c in AUDIO_CANDIDATES if c in df.columns and df[c].notna().any()]
df["_track_display"] = df[track_col].fillna("").astype(str).str.strip() if track_col else ""
df["_artist_display"] = df[artist_col].fillna("").astype(str).str.strip() if artist_col else ""

def chart_layout(fig, height=340, title=None):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title=title,
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(family="Inter, Arial, sans-serif", color=TEXT, size=12),
        margin=dict(l=28, r=24, t=58 if title else 24, b=35),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#222222", font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig

st.sidebar.markdown('<div class="brand">DATASET STATUS</div>', unsafe_allow_html=True)
st.sidebar.caption(data_source)
st.sidebar.metric("Songs", f"{len(df):,}")
st.sidebar.metric("Artists", f"{df['_artist_display'].replace('', np.nan).nunique():,}")
st.sidebar.metric("Audio features", len(audio_features))
st.sidebar.metric("Model status", "Loaded" if model is not None else "Not loaded")

overview_tab, explore_tab, recommend_tab, demo_tab, weights_tab = st.tabs([
    "Overview", "Explore Songs", "Recommendations", "Model Demo", "Feature Weights"
])

with overview_tab:
    st.subheader("Dataset overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total songs", f"{len(df):,}")
    c2.metric("Unique artists", f"{df['_artist_display'].replace('', np.nan).nunique():,}")
    c3.metric("Audio variables", len(audio_features))
    score = df["popularity_score"].dropna() if "popularity_score" in df.columns else pd.Series(dtype=float)
    c4.metric("Average popularity", f"{score.mean():.2f}" if len(score) else "N/A")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Popularity distribution")
        if len(score):
            fig = px.histogram(score, nbins=32, labels={"value": "Popularity score", "count": "Songs"})
            fig.update_traces(marker_color=ORANGE, marker_line_color=BG, marker_line_width=0.5)
            fig.update_layout(bargap=0.08)
            st.plotly_chart(chart_layout(fig, 320, "Custom Popularity Score"), use_container_width=True)
        else:
            st.info("Popularity scores are not available in the processed data.")
    with right:
        st.subheader("Feature distribution")
        dist_cols = [c for c in audio_features + ENGAGEMENT_INPUTS if c in df.columns and df[c].notna().any()]
        if dist_cols:
            chosen = st.selectbox("Select a variable", dist_cols)
            fig = px.histogram(df, x=chosen, nbins=30)
            fig.update_traces(marker_color=ORANGE, marker_line_color=BG, marker_line_width=0.5)
            st.plotly_chart(chart_layout(fig, 320, f"Distribution of {chosen.replace('_', ' ').title()}"), use_container_width=True)
        else:
            st.info("No numeric features available for plotting.")

    if len(audio_features) >= 2:
        st.subheader("Audio feature correlation")
        corr = df[audio_features].corr(numeric_only=True)
        fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale=[
            [0, "#363636"], [0.5, "#171717"], [1, ORANGE]
        ], aspect="auto", text_auto=".2f")
        st.plotly_chart(chart_layout(fig, max(350, min(600, 26 * len(audio_features))), "Audio Feature Correlation Matrix"), use_container_width=True)

with explore_tab:
    st.subheader("Explore the catalogue")
    search = st.text_input("Search by song or artist", placeholder="Enter a song or artist name")
    filtered = df.copy()
    if search:
        mask = filtered["_track_display"].str.contains(search, case=False, na=False, regex=False)
        mask |= filtered["_artist_display"].str.contains(search, case=False, na=False, regex=False)
        filtered = filtered[mask]
    show_cols = [c for c in [track_col, artist_col, "album", "popularity_score"] + audio_features + ENGAGEMENT_INPUTS if c and c in filtered.columns]
    st.dataframe(filtered[show_cols].head(500), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(len(filtered), 500):,} of {len(filtered):,} matching songs.")

with recommend_tab:
    st.subheader("Artist-based recommendations")
    st.markdown('<div class="section-note">Choose an artist and one reference track. Other songs by that artist are ranked by audio similarity and custom popularity.</div>', unsafe_allow_html=True)
    if not artist_col or not track_col:
        st.warning("Artist and track columns are required in the dataset.")
    elif "popularity_score" not in df.columns:
        st.warning("Popularity score is missing. Re-run the notebook's processing and save cell.")
    else:
        artists = sorted(a for a in df["_artist_display"].unique() if a)
        if not artists:
            st.warning("No artist names were found.")
        else:
            a_col, b_col, c_col = st.columns([1, 1.5, 1])
            with a_col:
                selected_artist = st.selectbox("Artist", artists, key="rec_artist")
            artist_df = df[df["_artist_display"].str.casefold() == selected_artist.casefold()].copy()
            tracks = sorted(t for t in artist_df["_track_display"].unique() if t)
            with b_col:
                selected_track = st.selectbox("Reference song", tracks, key="rec_track") if tracks else None
            with c_col:
                top_n = st.select_slider("Number of recommendations", options=[5, 6, 7, 8, 9, 10], value=5)
            popularity_weight = st.slider("Popularity influence", 0, 100, 60, 5, help="0 prioritizes audio similarity; 100 prioritizes popularity.")
            if st.button("Generate recommendations", type="primary", disabled=not bool(tracks)):
                candidates = artist_df[
                    (artist_df["_track_display"].str.casefold() != selected_track.casefold())
                    & artist_df["popularity_score"].notna()
                ].copy()
                if candidates.empty:
                    st.info("No other scored songs were found for this artist.")
                else:
                    source = artist_df[artist_df["_track_display"].str.casefold() == selected_track.casefold()].iloc[0]
                    if len(audio_features) >= 2:
                        candidate_x = candidates[audio_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
                        medians = candidate_x.median().fillna(0)
                        candidate_x = candidate_x.fillna(medians)
                        source_x = pd.DataFrame([{f: pd.to_numeric(source.get(f), errors="coerce") for f in audio_features}]).replace([np.inf, -np.inf], np.nan).fillna(medians)
                        scaler = StandardScaler()
                        candidate_scaled = scaler.fit_transform(candidate_x)
                        source_scaled = scaler.transform(source_x)
                        similarities = cosine_similarity(candidate_scaled, source_scaled).ravel()
                        candidates["_audio_similarity"] = np.clip((similarities + 1) / 2, 0, 1)
                    else:
                        candidates["_audio_similarity"] = 0.5
                    pop = pd.to_numeric(candidates["popularity_score"], errors="coerce")
                    pmin, pmax = pop.min(), pop.max()
                    candidates["_popularity_norm"] = ((pop - pmin) / (pmax - pmin)) if pmax > pmin else 1.0
                    w = popularity_weight / 100
                    candidates["_rank_score"] = (1 - w) * candidates["_audio_similarity"] + w * candidates["_popularity_norm"]
                    candidates = candidates.sort_values(["_rank_score", "popularity_score"], ascending=False).head(top_n)
                    result_cols = [c for c in [track_col, artist_col, "album", "popularity_score", "_audio_similarity", "_rank_score"] if c and c in candidates.columns]
                    result = candidates[result_cols].rename(columns={
                        "popularity_score": "Popularity score",
                        "_audio_similarity": "Audio similarity",
                        "_rank_score": "Recommendation rank"
                    }).reset_index(drop=True)
                    result.index += 1
                    st.success(f"Found {len(result)} recommendations for {selected_artist}.")
                    st.dataframe(result, use_container_width=True)
                    bar = px.bar(result.reset_index(names="Rank"), x="Rank", y="Popularity score",
                                 hover_name=track_col if track_col in result.columns else None)
                    bar.update_traces(marker_color=ORANGE)
                    st.plotly_chart(chart_layout(bar, 300, "Popularity of Recommended Songs"), use_container_width=True)
                    st.caption("Popularity is the project's custom score, not Spotify's official popularity score.")

with demo_tab:
    st.subheader("Popularity prediction")
    if model is None or not model_features:
        st.warning("Saved model files were not found. Run the updated model-training notebook cell first.")
    else:
        st.markdown('<div class="section-note">Adjust the sliders, then predict using the saved model inputs.</div>', unsafe_allow_html=True)
        inputs = {}
        columns = st.columns(3)
        for i, feature in enumerate(model_features):
            box = columns[i % 3]
            series = pd.to_numeric(df[feature], errors="coerce").dropna() if feature in df.columns else pd.Series(dtype=float)
            if series.empty:
                default = float(saved_medians.get(feature, 0.0))
                lo, hi = default - 1.0, default + 1.0
            else:
                lo = float(series.min())
                hi = float(series.quantile(0.99))
                if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                    lo, hi = float(series.min()), float(series.max()) + 1.0
                default = float(series.median())
            if feature in ["key", "mode", "time_signature"]:
                lo_i, hi_i = int(np.floor(lo)), max(int(np.ceil(hi)), int(np.floor(lo)) + 1)
                inputs[feature] = box.slider(feature.replace("_", " ").title(), lo_i, hi_i,
                                             int(np.clip(round(default), lo_i, hi_i)), step=1, key=f"demo_{feature}")
            else:
                step = max((hi - lo) / 100, 0.01)
                inputs[feature] = box.slider(feature.replace("_", " ").title(), float(lo), float(hi),
                                             float(np.clip(default, lo, hi)), step=float(step),
                                             key=f"demo_{feature}", format="%.2f")
        if st.button("Predict popularity", type="primary"):
            input_df = pd.DataFrame([inputs], columns=model_features).replace([np.inf, -np.inf], np.nan)
            input_df = input_df.fillna(pd.Series(saved_medians)).fillna(0)
            prediction = float(model.predict(input_df)[0])
            st.metric("Predicted custom popularity score", f"{prediction:.2f} / 100")
            st.caption("Prediction of the project's custom score, not Spotify's official popularity score.")

with weights_tab:
    st.subheader("Feature importance and relative weights")
    if model is not None and model_features and hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"Feature": model_features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
        total = importance["Importance"].sum()
        importance["Weight (%)"] = ((importance["Importance"] / total) * 100).round(2) if total > 0 else 0.0
        st.dataframe(importance, use_container_width=True, hide_index=True)
        plot_df = importance.sort_values("Importance", ascending=True)
        fig = px.bar(plot_df, x="Importance", y="Feature", orientation="h", text="Weight (%)")
        fig.update_traces(marker_color=ORANGE, texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        st.plotly_chart(chart_layout(fig, max(320, min(700, 34 * len(plot_df))), "Model Feature Importance"), use_container_width=True)
        st.download_button("Download feature weights CSV", importance.to_csv(index=False).encode("utf-8"),
                           file_name="feature_importance.csv", mime="text/csv")
    else:
        st.info("Feature importance is unavailable. Train/load the updated model first.")

st.divider()
st.caption("Feature importance shows relative model contribution, not causality. Likes/comments in both predictors and target construction can cause target leakage.")
