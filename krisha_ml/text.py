"""Text NLP track (lecture 10): ad descriptions as an extra information source.

Three kinds of blocks, all evaluated with the shared harness of ``core`` (train rows only, Endterm folds):

1. TF-IDF (word 1-2 grams + char 3-5 grams) + Ridge as a **residual correction** of the base model,
   cross-fitted inside every fold (``tfidf_corrector``); the Ridge penalty is chosen by efficient
   leave-one-out (RidgeCV, GCV) on the training fold only, so no alpha is picked on the CV folds.
2. Sentence-transformer embeddings (``embed_texts`` / ``load_embeddings``, cached in
   ``cache/text_emb_<model>.npy``), used as a Ridge residual correction (``embedding_corrector``) and as
   fold-fitted PCA(32) extra features for the boosting model (``embedding_pca_features``).
3. Red flags / attributes from a regex lexicon (``extract_flags``) plus an embedding-similarity score
   per flag (``flag_similarity``).

Leakage guards: TF-IDF sees ``raw.sanitize_text`` (no price sentences, digits or contacts). The
transformers see ``clean_for_embedding``: the same removals, but case and punctuation are kept because
they carry sentence structure for the encoder.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from . import core, raw

ALPHAS = np.logspace(-1, 6, 22)   # Ridge penalties searched by GCV inside each training fold (dense embeddings need > 1e4)

# --------------------------------------------------------------------------- text cleaning
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️‍]")


def clean_for_embedding(text: str | None) -> str:
    """Transformer input: contacts, URLs, price/money sentences and all numbers removed (same rules as
    ``raw.sanitize_text``), but case and punctuation kept; emojis dropped; whitespace collapsed."""
    text = str(text or "")
    for rx in (raw._URL, raw._EMAIL, raw._HANDLE, raw._PHONE, raw._CONTACT_MASK):
        text = rx.sub(" ", text)
    parts = re.split(r"(?<=[.!?\n])", text)
    text = "".join(p for p in parts if not raw._MONEY.search(p))
    text = raw._NUMBER.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- 1. TF-IDF residual model
def make_tfidf(word: bool = True, char: bool = True, max_char_features: int = 60000) -> FeatureUnion:
    """Word 1-2 gram + char_wb 3-5 gram TF-IDF on sanitized text (sublinear tf, L2 rows)."""
    parts = []
    if word:
        parts.append(("word", TfidfVectorizer(preprocessor=raw.sanitize_text, ngram_range=(1, 2), min_df=3,
                                              max_df=0.9, sublinear_tf=True, max_features=40000, dtype=np.float32)))
    if char:
        parts.append(("char", TfidfVectorizer(preprocessor=raw.sanitize_text, analyzer="char_wb", ngram_range=(3, 5),
                                              min_df=5, max_df=0.9, sublinear_tf=True, max_features=max_char_features,
                                              dtype=np.float32)))
    return FeatureUnion(parts)


def _ridge_cv(X, y, df_tr: pd.DataFrame | None = None, weighted: bool = False) -> RidgeCV:
    """RidgeCV (GCV over ALPHAS) on the training fold. ``weighted``: sample weight proportional to the
    base model's predicted price, so the log-scale fit leans towards the ads that dominate MAE in KZT."""
    w = None
    if weighted:
        w = np.exp(df_tr["base_log_pred"].to_numpy() - df_tr["base_log_pred"].mean())
    return RidgeCV(alphas=ALPHAS).fit(X, y, sample_weight=w)


@dataclass
class FitLog:
    """Ridge alphas chosen inside the folds, appended to a text file (works across joblib processes)."""
    path: str

    def add(self, a: float) -> None:
        with open(self.path, "a") as fh:
            fh.write(f"{a:.6g}\n")

    def read(self) -> list[float]:
        try:
            return [float(x) for x in open(self.path).read().split()]
        except FileNotFoundError:
            return []

    def reset(self) -> "FitLog":
        Path(self.path).unlink(missing_ok=True)
        return self


def tfidf_corrector(word: bool = True, char: bool = True, weighted: bool = False, log: FitLog | None = None) -> core.Corrector:
    """Residual corrector: TF-IDF -> RidgeCV on the honest training residuals of the fold."""
    def corr(df_tr, resid_tr, df_va):
        vec = make_tfidf(word, char)
        Xtr = vec.fit_transform(df_tr["description"].fillna(""))
        model = _ridge_cv(Xtr, resid_tr, df_tr, weighted)
        if log is not None:
            log.add(model.alpha_)
        return model.predict(vec.transform(df_va["description"].fillna("")))
    return corr


# --------------------------------------------------------------------------- 2. transformer embeddings
# key -> (HF id, passage prefix, query prefix)
MODELS = {
    "rubert-tiny2": ("cointegrated/rubert-tiny2", "", ""),
    "user-base": ("deepvk/USER-base", "query: ", "query: "),   # model card: "query: " for features
    "e5-base": ("intfloat/multilingual-e5-base", "passage: ", "query: "),
}


def emb_path(key: str):
    return core.CACHE_DIR / f"text_emb_{key}.npy"


def _device() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=3)
def load_encoder(key: str):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODELS[key][0], device=_device())
    model.max_seq_length = 512
    return model


def embed_texts(key: str, texts: list[str], prefix: str | None = None, batch_size: int = 16) -> np.ndarray:
    """L2-normalised sentence embeddings (mean pooling as configured by each model)."""
    prefix = MODELS[key][1] if prefix is None else prefix
    model = load_encoder(key)
    # prompt="" disables a model default prompt (USER-base adds "query: " itself); we add the prefix explicitly.
    # longest first: less padding per batch on MPS
    order = np.argsort([-len(t) for t in texts])
    emb = model.encode([prefix + texts[i] for i in order], prompt="", batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    out = np.empty_like(emb)
    out[order] = emb
    return out.astype(np.float32)


def build_embeddings(key: str, force: bool = False) -> np.ndarray:
    """Embed every clean ad once (rows = ``core.load_clean()`` order, ids saved alongside)."""
    path = emb_path(key)
    if path.exists() and not force:
        return np.load(path)
    df = core.load_clean()
    emb = embed_texts(key, [clean_for_embedding(t) for t in df["description"]])
    core.CACHE_DIR.mkdir(exist_ok=True)
    np.save(core.CACHE_DIR / "text_emb_ids.npy", df[core.ID].to_numpy("int64"))
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
    np.save(tmp, emb)
    os.replace(tmp, path)
    return emb


def load_embeddings(key: str) -> pd.DataFrame:
    """Cached embeddings as a frame indexed by ad id (columns e0..e{d-1})."""
    emb = np.load(emb_path(key))
    ids = np.load(core.CACHE_DIR / "text_emb_ids.npy")
    return pd.DataFrame(emb, index=pd.Index(ids, name=core.ID), columns=[f"e{i}" for i in range(emb.shape[1])])


def _emb_rows(table: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
    return table.reindex(df[core.ID].to_numpy("int64")).to_numpy(np.float32)


def embedding_corrector(key: str, weighted: bool = False, log: FitLog | None = None) -> core.Corrector:
    """Residual corrector: standardised embedding -> RidgeCV on the honest fold residuals."""
    table = load_embeddings(key)

    def corr(df_tr, resid_tr, df_va):
        sc = StandardScaler().fit(_emb_rows(table, df_tr))
        model = _ridge_cv(sc.transform(_emb_rows(table, df_tr)), resid_tr, df_tr, weighted)
        if log is not None:
            log.add(model.alpha_)
        return model.predict(sc.transform(_emb_rows(table, df_va)))
    return corr


def combined_corrector(keys: str | tuple[str, ...], flag_table: pd.DataFrame | None = None, emb_weight: float = 1.0,
                       weighted: bool = False, log: FitLog | None = None) -> core.Corrector:
    """TF-IDF + one or more embeddings (+ optional 0/1 flags) in ONE Ridge on the fold residuals.
    Blocks are sparse-stacked; each dense block is standardised and scaled by 1/sqrt(dim), so every block
    has a row norm of about 1 like the L2-normalised TF-IDF rows (fixed weights, not tuned on the folds)."""
    keys = (keys,) if isinstance(keys, str) else tuple(keys)
    tables = [load_embeddings(k) for k in keys]
    F = flag_table.set_index("id") if flag_table is not None else None

    def corr(df_tr, resid_tr, df_va):
        vec = make_tfidf()
        dense_tr = [_emb_rows(t, df_tr) for t in tables]
        dense_va = [_emb_rows(t, df_va) for t in tables]
        if F is not None:
            g = lambda d: F.reindex(d[core.ID].to_numpy("int64")).fillna(0).to_numpy(np.float32)
            dense_tr.append(g(df_tr)); dense_va.append(g(df_va))
        scalers = [StandardScaler().fit(b) for b in dense_tr]
        scale = lambda blocks: [sparse.csr_matrix(sc.transform(b) * (emb_weight / np.sqrt(b.shape[1])))
                                for sc, b in zip(scalers, blocks)]
        Xtr = sparse.hstack([vec.fit_transform(df_tr["description"].fillna(""))] + scale(dense_tr)).tocsr()
        Xva = sparse.hstack([vec.transform(df_va["description"].fillna(""))] + scale(dense_va)).tocsr()
        model = _ridge_cv(Xtr, resid_tr, df_tr, weighted)
        if log is not None:
            log.add(model.alpha_)
        return model.predict(Xva)
    return corr


def embedding_pca_features(key: str, n_components: int = 32, prefix: str = "txt_pc") -> core.FeatureFn:
    """feature_fn for ``core.cv_evaluate``: PCA fitted on the training fold's embeddings only."""
    table = load_embeddings(key)

    def fn(df_tr, y_tr, df_va):
        pca = PCA(n_components, random_state=core.SEED).fit(_emb_rows(table, df_tr))
        cols = [f"{prefix}{i}" for i in range(n_components)]
        mk = lambda d: pd.DataFrame(pca.transform(_emb_rows(table, d)), columns=cols)
        return mk(df_tr), mk(df_va)
    return fn


def oof_text_feature(corrector: core.Corrector, name: str = "txt_resid_oof") -> core.FeatureFn:
    """feature_fn: the corrector's cross-fitted residual prediction as ONE extra feature for the HGB
    (stacking as a feature instead of an additive correction). Training rows get inner-fold OOF values
    against inner-fold OOF base residuals; validation rows get the corrector fitted on the whole fold."""
    from sklearn.model_selection import KFold

    def fn(df_tr, y_tr, df_va):
        X_tr = core.encode_features(df_tr)
        resid = np.log(y_tr) - core.oof_log_pred(X_tr, y_tr)
        oof = np.empty(len(df_tr))
        for itr, iva in KFold(5, shuffle=True, random_state=core.SEED).split(df_tr):
            oof[iva] = corrector(df_tr.iloc[itr].reset_index(drop=True), resid[itr], df_tr.iloc[iva].reset_index(drop=True))
        va = corrector(df_tr, resid, df_va)
        return pd.DataFrame({name: oof}), pd.DataFrame({name: va})
    return fn


# --------------------------------------------------------------------------- 3. red flags / attributes
# Negations removed before the encumbrance search ("без обременений", "обременений нет", "не в залоге").
_NEG_ENC = re.compile(r"(?:без|нет|никаких|отсутству\w*|не)\s+(?:\w+\s+){0,2}?(?:обремен\w*|залог\w*|арест\w*)"
                      r"|(?:обремен\w*|залог\w*|арест\w*)\s+(?:\w+\s+){0,1}?(?:нет|отсутству\w*)"
                      r"|кепілде\s+емес")
FLAGS: dict[str, str] = {
    "assignment_of_rights": r"переуступ|уступк\w* прав|договор\w* уступки|цесси",
    "cash_only": r"только\s+(?:за\s+)?наличн|без\s+ипотек|ипотек\w*\s+не\s+(?:рассматр|подход|прох)|не\s+под\s+ипотек"
                 r"|не\s+подходит\s+под\s+ипотек|только\s+налич",
    "rough_finish": r"черновая|черновой|черновую|предчистов|white\s*box|вайт\s*бокс|без\s+отделки|без\s+ремонта",
    # "не требует ремонта" (needs no repair) is the opposite, hence the negative look-behinds
    "needs_repair": r"(?<!не )(?:требует|требуется)\s+(?:\w+\s+)?ремонт|(?<!не )нуждается\s+в\s+(?:\w+\s+)?ремонт"
                    r"|под\s+ремонт|нужен\s+ремонт|старый\s+ремонт|косметическ\w+\s+ремонт\w*\s+(?:не\s+помешает|требуется)",
    "urgent": r"срочн|шұғыл|тез\s+сатыл",
    "bargain": r"\bторг\b|саудалас",
    "encumbrance": r"обремен|залог|арест|кепіл",
    "no_encumbrance": r"без\s+обремен|обременени\w*\s+нет|не\s+в\s+залоге|без\s+залог",
    "former_dorm_text": r"общежит|общаг",
    "designer_renovation": r"дизайнерск|дизайн[\s-]*проект|авторск\w*\s+ремонт|ремонт\s+по\s+дизайн",
    "panoramic_view": r"панорам",
    "two_level": r"двух\s*-?\s*уровнев|двухуровн|в\s+двух\s+уровнях|пентхаус",
}
_FLAG_RX = {k: re.compile(v) for k, v in FLAGS.items()}


def _flag_text(text: str | None) -> str:
    return str(text or "").lower().replace("ё", "е")


def extract_flags(df: pd.DataFrame, col: str = "description") -> pd.DataFrame:
    """0/1 flag per ad (``id`` + one column per FLAGS key, prefixed ``flag_``)."""
    texts = df[col].map(_flag_text)
    out = {"id": df[core.ID].to_numpy("int64")}
    for k, rx in _FLAG_RX.items():
        src = texts.map(lambda t: _NEG_ENC.sub(" ", t)) if k == "encumbrance" else texts
        out[f"flag_{k}"] = src.map(lambda t, rx=rx: int(bool(rx.search(t)))).to_numpy("int8")
    return pd.DataFrame(out)


# Prototype phrases for the embedding-similarity check of each flag (encoded as queries).
FLAG_PROTOTYPES = {
    "assignment_of_rights": "Продажа квартиры по переуступке права требования в строящемся доме",
    "cash_only": "Продажа только за наличный расчёт, ипотека не рассматривается",
    "rough_finish": "Квартира в черновой или предчистовой отделке, без ремонта",
    "needs_repair": "Квартира требует ремонта, старое состояние",
    "urgent": "Срочная продажа, торг уместен",
    "bargain": "Цена обсуждаема, реальному покупателю хороший торг",
    "encumbrance": "Квартира в залоге у банка, есть обременение",
    "no_encumbrance": "Документы в порядке, без обременений",
    "former_dorm_text": "Квартира в бывшем общежитии",
    "designer_renovation": "Дизайнерский ремонт, квартира сделана по дизайн-проекту",
    "panoramic_view": "Панорамные окна с видом на город",
    "two_level": "Двухуровневая квартира, пентхаус на последнем этаже",
}


def flag_similarity(key: str) -> pd.DataFrame:
    """Cosine similarity of each ad's embedding to each flag prototype (static, no fitting)."""
    table = load_embeddings(key)
    names = list(FLAG_PROTOTYPES)
    protos = embed_texts(key, [FLAG_PROTOTYPES[n] for n in names], prefix=MODELS[key][2])
    sims = table.to_numpy(np.float32) @ protos.T
    return pd.DataFrame(sims, index=table.index, columns=[f"sim_{n}" for n in names]).reset_index()


def flag_corrector(flag_table: pd.DataFrame, log: FitLog | None = None) -> core.Corrector:
    """Residual corrector: RidgeCV on the 0/1 flags (a small linear model on the base residual)."""
    T = flag_table.set_index("id")

    def corr(df_tr, resid_tr, df_va):
        g = lambda d: T.reindex(d[core.ID].to_numpy("int64")).fillna(0).to_numpy(float)
        model = _ridge_cv(g(df_tr), resid_tr)
        if log is not None:
            log.add(model.alpha_)
        return model.predict(g(df_va))
    return corr


# --------------------------------------------------------------------------- helpers for reports
def residual_delta(res: core.CVResult) -> np.ndarray:
    """Out-of-fold log-scale correction implied by a residual CVResult (log(pred) - base OOF log pred)."""
    return np.log(res.oof_pred) - core.base_residual_inputs()["outer"]


def top_terms(texts: pd.Series, resid: np.ndarray, n: int = 25) -> pd.DataFrame:
    """Word 1-2 gram TF-IDF Ridge on residuals -> the n terms with the largest +/- coefficients."""
    vec = TfidfVectorizer(preprocessor=raw.sanitize_text, ngram_range=(1, 2), min_df=10, max_df=0.9, sublinear_tf=True)
    X = vec.fit_transform(texts.fillna(""))
    model = _ridge_cv(X, resid)
    coef, terms = model.coef_, np.asarray(vec.get_feature_names_out())
    df_freq = np.asarray((X > 0).sum(axis=0)).ravel()
    idx = np.r_[np.argsort(coef)[:n], np.argsort(coef)[::-1][:n]]
    return pd.DataFrame({"term": terms[idx], "coef": coef[idx], "n_ads": df_freq[idx],
                         "direction": np.where(coef[idx] > 0, "up", "down"), "alpha": model.alpha_})


# --------------------------------------------------------------------------- 4. optional fine-tuning
def finetune_corrector(hf_id: str = "cointegrated/rubert-tiny2", epochs: int = 3, lr: float = 5e-5,
                       max_len: int = 256, batch_size: int = 16, log: list | None = None) -> core.Corrector:
    """Residual corrector: fine-tune a small encoder (mean pooling + linear head) on the fold's honest
    log residuals (train fold only; target standardised, Smooth-L1 loss), then predict the validation ads.
    Fixed schedule (no early stopping on validation rows), seed 42; MPS kernels are not bit-exact."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = torch.device(_device())

    def encode(model, tok, texts):
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def corr(df_tr, resid_tr, df_va):
        torch.manual_seed(core.SEED)
        rng = np.random.default_rng(core.SEED)
        tok = AutoTokenizer.from_pretrained(hf_id)
        enc_model = AutoModel.from_pretrained(hf_id).to(dev)
        head = torch.nn.Linear(enc_model.config.hidden_size, 1).to(dev)
        mu, sd = float(np.mean(resid_tr)), float(np.std(resid_tr))
        t_tr = [clean_for_embedding(t) for t in df_tr["description"]]
        t_va = [clean_for_embedding(t) for t in df_va["description"]]
        y = torch.tensor((resid_tr - mu) / sd, dtype=torch.float32)
        params = [{"params": enc_model.parameters(), "lr": lr}, {"params": head.parameters(), "lr": lr * 20}]
        opt = torch.optim.AdamW(params, weight_decay=0.01)
        steps = epochs * int(np.ceil(len(t_tr) / batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 20], total_steps=steps, pct_start=0.1)
        loss_fn = torch.nn.SmoothL1Loss()
        enc_model.train()
        for ep in range(epochs):
            order, total = rng.permutation(len(t_tr)), 0.0
            for s in range(0, len(order), batch_size):
                idx = order[s:s + batch_size]
                pred = head(encode(enc_model, tok, [t_tr[i] for i in idx])).squeeze(-1)
                loss = loss_fn(pred, y[idx].to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                total += float(loss) * len(idx)
            if log is not None:
                log.append({"epoch": ep + 1, "train_loss": total / len(t_tr)})
        enc_model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(t_va), 64):
                out.append(head(encode(enc_model, tok, t_va[s:s + 64])).squeeze(-1).float().cpu().numpy())
        del enc_model, head
        if dev.type == "mps":
            torch.mps.empty_cache()
        return np.concatenate(out) * sd + mu
    return corr
