import streamlit as st
import requests
import os

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="FakeNews | Détection de fake news climatiques",
    page_icon="🌍",
    layout="centered"
)

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"]{
  background: #f4fbf7 !important;
}
[data-testid="stAppViewContainer"] > .main{
  background: transparent !important;
}

.block-container{
  background: rgba(255,255,255,0.82);
  border: 2px solid rgba(0,0,0,0.14) !important;
  border-radius: 22px !important;
  padding: 0rem 2rem 2rem 2rem;
  outline: 1px solid rgba(255,255,255,0.70);
  margin-top: 2rem;
  margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# CSS (hero + cards + big indicator)
# ---------------------------
st.markdown("""
<style>

.fn-title { font-size: 1.85rem; font-weight: 850; Margin: 0 0 0 0; }
.fn-sub { color: rgba(0,0,0,0.72); line-height: 1.35; Margin: 0 0 0 0; }
            
.fn-badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 650;
  border: 1px solid rgba(0,0,0,0.08);
  background: rgba(0,0,0,0.02);
}

.small-note { font-size: 0.88rem; color: rgba(0,0,0,0.60); }

.fn-indicator-wrap{
  display:flex;
  align-items:center;
  gap:14px;
  margin: 6px 0 10px 0;
}
.fn-indicator{
  font-size: 1.05rem;
  backdrop-filter: blur(6px);
  width:86px;
  height:86px;
  border-radius:999px;
  display:flex;
  align-items:center;
  justify-content:center;
  font-weight:900;
  letter-spacing:0.5px;
  border: 1px solid rgba(0,0,0,0.08);
  box-shadow: 0 10px 28px rgba(0,0,0,0.08);
  user-select:none;
}
.fn-indicator-text{
  display:flex;
  flex-direction:column;
  gap:3px;
}
.fn-kicker{ font-size: 0.9rem; color: rgba(0,0,0,0.62); }
.fn-main{ font-size: 1.15rem; font-weight: 850; }
.fn-conf{ font-size: 0.95rem; color: rgba(0,0,0,0.68); }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header / Hero
# ---------------------------
st.markdown('<p class="fn-title">🌍 FakeNews</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="fn-sub">Collez un texte <b>en français</b> sur le changement climatique et obtenez une estimation : '
    '<b>fiable</b>, <b>faux</b> ou <b>biaisé</b>.</p>',
    unsafe_allow_html=True
)

# ---------------------------
# Session state
# ---------------------------
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

# ---------------------------
# Input card
# ---------------------------

st.session_state.user_text = st.text_area(
    label="",
    value=st.session_state.user_text,
    placeholder="Collez ici un texte en français (idéalement 2–3 phrases ou plus)…",
    height=220
)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Exemple fiable", use_container_width=True):
        st.session_state.user_text = (
            "Le réchauffement climatique est principalement causé par l’augmentation des gaz à effet de serre "
            "d’origine humaine, ce qui influence les températures moyennes et certains extrêmes climatiques."
        )
with c2:
    if st.button("Exemple douteux", use_container_width=True):
        st.session_state.user_text = (
            "Les scientifiques admettent que le réchauffement climatique est une invention : "
            "tout cela n’est qu’un complot et les températures n’ont jamais changé."
        )
with c3:
    if st.button("Effacer", use_container_width=True):
        st.session_state.user_text = ""

st.markdown('<p class="small-note">Astuce: plus le texte est long, plus la classification est stable.</p>', unsafe_allow_html=True)

# ---------------------------
# Predict (API call)
# ---------------------------
# API_URL comporte l'adresse de l'API pour la prédiction du modèle
# exemple: https://[your_HF_name]-[your_space_name].hf.space/predict
API_URL = os.environ["API_URL"] + '/predict'

verify = st.button("Vérifier la nouvelle", type="primary", use_container_width=True)

def render_indicator(kind: str, main_label: str, conf: float | None):
    """
    kind: 'ok' | 'bad' | 'warn' | 'info'
    """
    # Soft background tints (not screaming)
    styles = {
        "ok":   ("background: rgba(46, 204, 113, 0.18);", "FIABLE"),
        "bad":  ("background: rgba(231, 76, 60, 0.18);",  "FAUX"),
        "warn": ("background: rgba(241, 196, 15, 0.22);", "BIAISÉ"),
        "info": ("background: rgba(52, 152, 219, 0.16);", "INFO"),
    }
    bg, short = styles.get(kind, styles["info"])

    conf_txt = ""
    if isinstance(conf, (float, int)):
        conf = max(0.0, min(1.0, float(conf)))
        conf_txt = f"Confiance: {int(conf*100)}%"

    st.markdown(f"""
    <div class="fn-indicator-wrap">
      <div class="fn-indicator" style="{bg}">{short}</div>
      <div class="fn-indicator-text">
        <div class="fn-kicker">Résultat de l’analyse</div>
        <div class="fn-main">{main_label}</div>
        <div class="fn-conf">{conf_txt}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def normalize_api_response(result):
    # Cas scikit-learn : {"prediction": 0}
    if isinstance(result.get("prediction"), int):
        prediction = result["prediction"]
        conf = result.get("confidence", None)
        reasons = result.get("reasons", None)

    # Cas transformers : {"prediction": "LABEL_2", "score": 0.95}
    elif isinstance(result.get("prediction"), str):
        label_to_int = {
            "LABEL_0": 0,
            "LABEL_1": 1,
            "LABEL_2": 2,
        }
        prediction = label_to_int.get(result["prediction"], -1)
        conf = result.get("score", None)
        reasons = result.get("reasons", None)

    else:
        prediction = -1  # Valeur par défaut pour "non classé"
        conf = None
        reasons = None

    return prediction, conf, reasons

if verify:
    text = (st.session_state.user_text or "").strip()

    if not text:
        st.warning("Veuillez saisir un texte avant de vérifier.")
    else:
        with st.spinner("Analyse de l'article en cours..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"text": text},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()

                # Normalise la réponse
                prediction, conf, reasons = normalize_api_response(result)

                # Big indicator + message
                if prediction == 2:
                    render_indicator("ok", "Probablement vrai", conf)
                    st.success("L'article est probablement vrai.")
                elif prediction == 1:
                    render_indicator("bad", "Probablement faux", conf)
                    st.error("L'article est probablement faux.")
                elif prediction == 0:
                    render_indicator("warn", "Biaisé", conf)
                    st.warning("L'article est probablement vrai, mais biaisé.")
                else:
                    render_indicator("info", "Non classé", conf)
                    st.info("L'article n'a pas pu être classé.")

                # Confidence bar if available
                if isinstance(conf, (float, int)):
                    conf = max(0.0, min(1.0, float(conf)))
                    st.progress(int(conf * 100))

                # Reasons if available (optional)
                if reasons:
                    with st.expander("Pourquoi ? (détails)"):
                        if isinstance(reasons, list):
                            for r in reasons:
                                st.write("• " + str(r))
                        else:
                            st.write(reasons)

                st.markdown("</div>", unsafe_allow_html=True)

            except requests.exceptions.RequestException as e:
                st.error(f"Une erreur est survenue lors de la vérification de l'article: {e}")

st.caption("⚠️ Cet outil fournit une estimation basée sur un modèle d’apprentissage automatique et peut se tromper.")