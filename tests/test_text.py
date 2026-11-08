"""Fast checks of the text track (krisha_ml/text.py), no transformer download needed (a few seconds)."""
import numpy as np
import pandas as pd
import pytest

from krisha_ml import core, text


def test_clean_for_embedding_drops_prices_digits_contacts():
    raw_text = ("Продаётся светлая квартира. Цена 45 млн тенге, торг!\n"
                "Звоните +7 701 123 45 67 или пишите @agent_kz. Площадь 85,5 м2, 3 комнаты.")
    out = text.clean_for_embedding(raw_text)
    assert "Продаётся светлая квартира." in out          # case and punctuation kept
    assert "млн" not in out and "Цена" not in out          # the price sentence is gone
    assert not any(ch.isdigit() for ch in out)
    assert "@agent" not in out and "701" not in out


@pytest.mark.parametrize("desc, flag, expected", [
    ("Квартира не требует ремонта, заезжай и живи", "needs_repair", 0),
    ("Квартира требует ремонта", "needs_repair", 1),
    ("Документы готовы, без обременений", "encumbrance", 0),
    ("Документы готовы, без обременений", "no_encumbrance", 1),
    ("Квартира находится в залоге у банка", "encumbrance", 1),
    ("Продажа по переуступке права требования", "assignment_of_rights", 1),
    ("Только наличный расчёт", "cash_only", 1),
    ("Рассмотрим ипотеку и наличные", "cash_only", 0),
    ("Рядом торговый центр", "bargain", 0),
    ("Торг уместен", "bargain", 1),
    ("Предчистовая отделка, white box", "rough_finish", 1),
    ("Двухуровневая квартира с панорамным видом", "two_level", 1),
    ("Двухуровневая квартира с панорамным видом", "panoramic_view", 1),
    ("Шұғыл сатылады", "urgent", 1),
])
def test_flag_lexicon(desc, flag, expected):
    f = text.extract_flags(pd.DataFrame({"id": [1], "description": [desc]}))
    assert int(f[f"flag_{flag}"].iloc[0]) == expected


def test_flags_cover_every_clean_ad():
    clean = core.load_clean()
    f = text.extract_flags(clean)
    assert len(f) == len(clean) and f["id"].is_unique
    assert set(f.columns) == {"id"} | {f"flag_{k}" for k in text.FLAGS}
    assert f.drop(columns="id").isin([0, 1]).all().all()


def test_tfidf_corrector_contract():
    train = core.load_train()
    df_tr, df_va = train.df.iloc[:400].reset_index(drop=True), train.df.iloc[400:500].reset_index(drop=True)
    resid = np.random.default_rng(0).normal(0, 0.1, len(df_tr))
    df_tr = df_tr.assign(base_log_pred=np.log(train.y[:400]))
    for weighted in (False, True):
        delta = text.tfidf_corrector(weighted=weighted)(df_tr, resid, df_va)
        assert delta.shape == (100,) and np.isfinite(delta).all()
