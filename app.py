import os
import io
import time
import csv
import pickle
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from tqdm import tqdm
from datetime import datetime

from sentence_transformers import SentenceTransformer
import faiss

from utils import list_images, load_image, pil_from_bytes
from exif_utils import get_exif_dict, parse_datetime_original, camera_make_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

st.set_page_config(page_title="AI Photo Search", page_icon="🔎", layout="wide")

ROOT = os.path.dirname(__file__)
PHOTOS_DIR = os.path.join(ROOT, "photos")
UPLOADS_DIR = os.path.join(PHOTOS_DIR, "uploads")
CACHE_DIR = os.path.join(ROOT, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

INDEX_PATH = os.path.join(CACHE_DIR, "faiss.index")
EMB_PATH = os.path.join(CACHE_DIR, "embeddings.npy")
NAMES_PATH = os.path.join(CACHE_DIR, "names.pkl")
EXIF_PATH = os.path.join(CACHE_DIR, "exif.pkl")

# ---------------------- Sidebar ----------------------
st.sidebar.title("Impostazioni")
top_k = st.sidebar.slider("Numero risultati (k)", 4, 100, 24)
thumb_size = st.sidebar.slider("Dimensione anteprima", 120, 380, 200, step=10)
rebuild = st.sidebar.button("🔄 Ricalcola indice")

mode = st.sidebar.radio("Modalità ricerca", ["Testo → Immagine", "Immagine → Immagine"])

st.sidebar.markdown("---")
st.sidebar.subheader("Pesi re-ranking")
w_sem = st.sidebar.slider("Peso semantico (CLIP)", 0.0, 1.0, 0.70, 0.01)
w_kw  = st.sidebar.slider("Peso keyword (nome file)", 0.0, 1.0, 0.20, 0.01)
w_exif= st.sidebar.slider("Peso EXIF", 0.0, 1.0, 0.10, 0.01)
min_score = st.sidebar.slider("Soglia minima similarità", 0.0, 1.0, 0.25, 0.01)

# ---------------------- Dataset upload ----------------------
with st.sidebar.expander("📤 Upload dataset"):
    uploaded_files = st.file_uploader("Carica immagini (multi-file)", type=["jpg","jpeg","png","webp"], accept_multiple_files=True)
    if uploaded_files:
        saved = 0
        for uf in uploaded_files:
            try:
                b = uf.read()
                im = pil_from_bytes(b)
                # Normalize name
                safe_name = uf.name.replace("\\", "_").replace("/", "_")
                path = os.path.join(UPLOADS_DIR, safe_name)
                im.save(path)
                saved += 1
            except Exception as e:
                st.warning(f"Errore con {uf.name}: {e}")
        if saved:
            st.success(f"Salvate {saved} immagini in 'photos/uploads/'. Premi **Ricalcola indice**.")
            # Suggest rebuild
            rebuild = True

# ---------------------- Model load ----------------------
@st.cache_resource(show_spinner=True)
def load_model():
    model = SentenceTransformer("clip-ViT-B-32")
    return model

model = load_model()

def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-10
    return x / norms

def _build_index(emb: np.ndarray) -> faiss.Index:
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)
    return idx

def _save_cache(emb: np.ndarray, names: list, index: faiss.Index, exif_map: dict):
    np.save(EMB_PATH, emb)
    with open(NAMES_PATH, "wb") as f:
        pickle.dump(names, f)
    with open(EXIF_PATH, "wb") as f:
        pickle.dump(exif_map, f)
    faiss.write_index(index, INDEX_PATH)

def _load_cache():
    if not (os.path.exists(EMB_PATH) and os.path.exists(NAMES_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(EXIF_PATH)):
        return None, None, None, None
    emb = np.load(EMB_PATH)
    with open(NAMES_PATH, "rb") as f:
        names = pickle.load(f)
    with open(EXIF_PATH, "rb") as f:
        exif_map = pickle.load(f)
    index = faiss.read_index(INDEX_PATH)
    return emb, names, index, exif_map

# ---------------------- Indexing with EXIF ----------------------
def index_photos() -> tuple:
    paths, names = list_images(PHOTOS_DIR)
    if len(paths) == 0:
        st.warning("Nessuna immagine nella cartella 'photos/'. Aggiungine alcune e ricarica l'app.")
        return None, None, None, None

    st.info(f"Trovate {len(paths)} immagini. Calcolo embedding & EXIF...")
    images = []
    for p in tqdm(paths):
        images.append(load_image(p, max_side=768))
    emb = model.encode(images, batch_size=32, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    emb = _normalize(emb)
    index = _build_index(emb)

    # Build EXIF map for filters
    exif_map = {}
    for p, rel in tqdm(zip(paths, names), total=len(paths)):
        exif = get_exif_dict(p)
        date_iso = parse_datetime_original(exif)
        make, model_cam = camera_make_model(exif)
        exif_map[rel] = {
            "Date": date_iso,
            "Make": make,
            "Model": model_cam
        }

    return emb, names, index, exif_map

def ensure_index(force: bool = False):
    emb, names, index, exif_map = _load_cache()
    paths, _ = list_images(PHOTOS_DIR)
    if force or emb is None or len(names) != len(paths):
        with st.spinner("Costruzione/aggiornamento indice..."):
            res = index_photos()
            if res[0] is not None:
                emb, names, index, exif_map = res
                _save_cache(emb, names, index, exif_map)
    return emb, names, index, exif_map

embeddings, names, index = None, None, None
exif_map = None

embeddings, names, index, exif_map = ensure_index(force=rebuild)

st.title("🔎 AI Photo Search")
st.caption("Cerca nel tuo pool di foto usando **testo** o **immagini**. CLIP ViT-B/32 + FAISS, filtri EXIF, ricerca booleana e re-ranking.")

if embeddings is None or index is None or names is None:
    st.stop()

# ---------------------- Filters (filename, EXIF) ----------------------
# Build camera lists
all_makes = sorted({ (exif_map[n].get("Make") or "").strip() for n in names if exif_map.get(n) })
all_makes = [m for m in all_makes if m]
make_sel = st.multiselect("Marca fotocamera (EXIF)", options=all_makes, default=[])

# Once make selected, suggest models
models_opts = sorted({ (exif_map[n].get("Model") or "").strip() for n in names if exif_map.get(n) and (not make_sel or (exif_map[n].get('Make') or '').strip() in make_sel)})
models_opts = [m for m in models_opts if m]
model_sel = st.multiselect("Modello fotocamera (EXIF)", options=models_opts, default=[])

fname_contains = st.text_input("Nome file contiene (case-insensitive, può essere regex semplice):", value="")

# Date filter bounds
dates_available = [exif_map[n].get("Date") for n in names if exif_map.get(n) and exif_map[n].get("Date")]
def _to_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except Exception:
        return None
dates_parsed = [ _to_date(d) for d in dates_available ]
dates_parsed = [d for d in dates_parsed if d]

date_min = min(dates_parsed).isoformat() if dates_parsed else ""
date_max = max(dates_parsed).isoformat() if dates_parsed else ""

c1, c2 = st.columns(2)
with c1:
    start_date = st.text_input("Data inizio (YYYY-MM-DD)", value=date_min)
with c2:
    end_date = st.text_input("Data fine (YYYY-MM-DD)", value=date_max)

def _date_in_range(dstr, start_date, end_date):
    if not dstr:
        return True
    try:
        d = datetime.strptime(dstr, "%Y-%m-%d").date()
    except Exception:
        return True
    ok_start = True
    ok_end = True
    if start_date:
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d").date()
            ok_start = d >= s
        except Exception:
            ok_start = True
    if end_date:
        try:
            e = datetime.strptime(end_date, "%Y-%m-%d").date()
            ok_end = d <= e
        except Exception:
            ok_end = True
    return ok_start and ok_end

# Precompute filter mask
def apply_filters(names):
    mask = np.ones(len(names), dtype=bool)
    for i, rel in enumerate(names):
        info = exif_map.get(rel, {})
        # make/model
        if make_sel and (info.get("Make") or "").strip() not in make_sel:
            mask[i] = False
            continue
        if model_sel and (info.get("Model") or "").strip() not in model_sel:
            mask[i] = False
            continue
        # filename contains
        if fname_contains:
            if fname_contains.lower() not in rel.lower():
                mask[i] = False
                continue
        # date range
        if not _date_in_range(info.get("Date"), start_date, end_date):
            mask[i] = False
            continue
    return mask

# Utility for grid display
def show_results(idx_list, score_list, names, top_k, thumb_size, min_score=0.25):
    # filtra risultati troppo bassi
    filtered = [(i, s) for i, s in zip(idx_list, score_list) if s >= min_score]
    n = min(len(filtered), top_k)
    if n == 0:
        st.warning(f"Nessuna immagine supera la soglia di similarità ({min_score:.2f})")
        return
    cols = st.columns(4, gap="large")
    for i in range(n):
        idx, sc = filtered[i]
        col = cols[i % 4]
        with col:
            rel_path = names[idx]
            abs_path = os.path.join(PHOTOS_DIR, rel_path)
            img = load_image(abs_path, max_side=thumb_size)
            st.image(img, caption=f"{rel_path}\nscore: {sc:.3f}", use_container_width=True)

# ---------------------- Boolean query parsing ----------------------
def parse_boolean(query: str):
    # Very simple parser: returns (pos_terms_groups, neg_terms)
    # Supports: AND, OR, NOT
    # Example: "cavallo AND spiaggia NOT notte OR corsa"
    # We split by OR into groups, inside each group AND is enforced.
    if not query:
        return [[]], []
    q = query.replace(" and ", " AND ").replace(" or ", " OR ").replace(" not ", " NOT ")
    tokens = q.split()
    groups = []
    current = []
    neg = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "OR":
            if current:
                groups.append(current)
                current = []
        elif t == "AND":
            pass
        elif t == "NOT":
            i += 1
            if i < len(tokens):
                neg.append(tokens[i])
        else:
            current.append(t)
        i += 1
    if current:
        groups.append(current)
    if not groups:
        groups = [[]]
    return groups, neg

def embed_text_logic(groups, neg_terms):
    # For OR groups: compute embedding per group (avg of AND terms), then average groups
    group_vecs = []
    for g in groups:
        if not g:
            continue
        if len(g) == 1:
            v = model.encode([g[0]], convert_to_numpy=True, normalize_embeddings=True)[0]
        else:
            vs = model.encode(g, convert_to_numpy=True, normalize_embeddings=True)
            v = vs.mean(axis=0)
        group_vecs.append(v)
    if group_vecs:
        qv = np.stack(group_vecs, axis=0).mean(axis=0, keepdims=True)
    else:
        qv = model.encode([""], convert_to_numpy=True, normalize_embeddings=True)
    # Apply NOT terms by subtracting
    if neg_terms:
        neg_vecs = model.encode(neg_terms, convert_to_numpy=True, normalize_embeddings=True).mean(axis=0, keepdims=True)
        qv = qv - 0.5 * neg_vecs
    return _normalize(qv)

# ---------------------- Search & re-ranking ----------------------
def keyword_score(name: str, query: str) -> float:
    if not query:
        return 0.0
    # Count matches of positive words minus negatives
    groups, neg = parse_boolean(query)
    pos_words = set([w for g in groups for w in g])
    score = 0.0
    low = name.lower()
    for w in pos_words:
        if w.lower() in low:
            score += 1.0
    for w in neg:
        if w.lower() in low:
            score -= 1.0
    return max(0.0, score)

def exif_score(info: dict, make_sel: list, model_sel: list, start_date: str, end_date: str) -> float:
    s = 0.0
    if make_sel and (info.get("Make") or "").strip() in make_sel:
        s += 1.0
    if model_sel and (info.get("Model") or "").strip() in model_sel:
        s += 1.0
    if _date_in_range(info.get("Date"), start_date, end_date):
        s += 0.5
    return s

# Export helpers
def export_csv(idx_list, score_list, names):
    rows = []
    for i, idx in enumerate(idx_list):
        rel = names[idx]
        info = exif_map.get(rel, {})
        rows.append({
            "rank": i+1,
            "path": rel,
            "score": float(score_list[i]),
            "date": info.get("Date"),
            "make": (info.get("Make") or ""),
            "model": (info.get("Model") or ""),
        })
    fn = os.path.join(CACHE_DIR, "results.csv")
    with open(fn, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return fn

def export_zip(idx_list, names):
    zpath = os.path.join(CACHE_DIR, "results.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, idx in enumerate(idx_list):
            rel = names[idx]
            src = os.path.join(PHOTOS_DIR, rel)
            # Keep folder structure inside zip
            zf.write(src, arcname=f"{i+1:03d}_{rel}")
    return zpath

# ---------------------- UI execution ----------------------
results_idxs = None
results_scores = None

if mode == "Testo → Immagine":
    query = st.text_input("Descrivi cosa cerchi (usa AND / OR / NOT):", value="cavallo AND spiaggia NOT notte")
    if st.button("Cerca (testo)") and query.strip():
        with st.spinner("Cerco..."):
            groups, neg_terms = parse_boolean(query)
            q_emb = embed_text_logic(groups, neg_terms)
            # broader retrieval
            topN = max(top_k*5, 200)
            scores, idxs = index.search(q_emb.astype('float32'), topN)
            idxs = idxs[0].tolist()
            scores = scores[0].tolist()

            # Apply filters mask
            mask = apply_filters(names)
            filtered = [(i, s) for i, s in zip(idxs, scores) if mask[i]]

            # Compute auxiliary scores for re-ranking
            aux = []
            for i, s in filtered:
                kw = keyword_score(names[i], query)
                ex = exif_score(exif_map.get(names[i], {}), make_sel, model_sel, start_date, end_date)
                blended = w_sem * s + w_kw * kw + w_exif * ex
                aux.append((i, s, kw, ex, blended))

            aux.sort(key=lambda x: x[4], reverse=True)
            results_idxs = [a[0] for a in aux]
            results_scores = [a[4] for a in aux]

elif mode == "Immagine → Immagine":
    uploaded = st.file_uploader("Carica un'immagine (jpg/png/webp)...", type=["jpg","jpeg","png","webp"])
    if uploaded is not None:
        img = pil_from_bytes(uploaded.read())
        st.image(img, caption="Query", width=thumb_size*2)
        if st.button("Trova simili"):
            with st.spinner("Cerco immagini simili..."):
                q_emb = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
                q_emb = _normalize(q_emb)
                topN = max(top_k*5, 200)
                scores, idxs = index.search(q_emb.astype('float32'), topN)
                idxs = idxs[0].tolist()
                scores = scores[0].tolist()

                # Apply filters
                mask = apply_filters(names)
                filtered = [(i, s) for i, s in zip(idxs, scores) if mask[i]]

                # Re-ranking (no keyword component; still include EXIF and semantic)
                aux = []
                for i, s in filtered:
                    ex = exif_score(exif_map.get(names[i], {}), make_sel, model_sel, start_date, end_date)
                    blended = w_sem * s + w_exif * ex
                    aux.append((i, s, ex, blended))

                aux.sort(key=lambda x: x[3], reverse=True)
                results_idxs = [a[0] for a in aux]
                results_scores = [a[3] for a in aux]

# Show & Export
if results_idxs is not None and results_scores is not None:
    show_results(results_idxs, results_scores, names, top_k, thumb_size,min_score)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬇️ Esporta CSV risultati"):
            path = export_csv(results_idxs[:top_k], results_scores[:top_k], names)
            st.success(f"CSV pronto: {path}")
            st.download_button("Download CSV", data=open(path, "rb").read(), file_name="results.csv")
    with c2:
        if st.button("⬇️ Esporta ZIP immagini"):
            zpath = export_zip(results_idxs[:top_k], names)
            st.success(f"ZIP pronto: {zpath}")
            st.download_button("Download ZIP", data=open(zpath, "rb").read(), file_name="results.zip")
