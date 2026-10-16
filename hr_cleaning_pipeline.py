"""
HR Analytics — Job Change of Data Scientists
Exploration, analyse et nettoyage des données.

Ce script reprend l'intégralité du notebook, découpé en fonctions :
    1. Chargement des données
    2. Qualité des données (doublons, valeurs manquantes, modalités, cohérence)
    3. Analyse univariée
    4. Analyse bivariée vs la cible
    5. Valeurs manquantes informatives
    6. Nettoyage et transformation (classe HRCleaner)
    7. Vérifications, corrélations, multicolinéarité, jeux de variables
    8. Contrôle final du nettoyage
    9. Export et synthèse

Utilisation (terminal ou cellule Kaggle) :
    python hr_cleaning_pipeline.py
    python hr_cleaning_pipeline.py --data-dir chemin/vers/csv --out-dir resultats --no-plots
"""

import argparse
import glob
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # pas d'affichage : les graphiques sont enregistrés en fichiers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 100


# =============================================================================
# CONFIGURATION
# =============================================================================
TARGET = "target"
ID_COL = "enrollee_id"
NUM_COLS = ["city_development_index", "training_hours"]
PLOT_CATS = ["gender", "relevent_experience", "enrolled_university", "education_level",
             "major_discipline", "experience", "company_size", "company_type", "last_new_job"]
ORDER_MAP = {
    "experience": ["<1"] + [str(i) for i in range(1, 21)] + [">20"],
    "company_size": ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
    "last_new_job": ["never", "1", "2", "3", "4", ">4"],
    "education_level": ["Primary School", "High School", "Graduate", "Masters", "Phd"],
}

EXPERIENCE_MAP = {"<1": 0, **{str(i): i for i in range(1, 21)}, ">20": 21}
LAST_NEW_JOB_MAP = {"never": 0, "1": 1, "2": 2, "3": 3, "4": 4, ">4": 5}
COMPANY_SIZE_MAP = {"<10": 0, "10-49": 1, "50-99": 2, "100-500": 3, "500-999": 4,
                    "1000-4999": 5, "5000-9999": 6, "10000+": 7}
EDUCATION_MAP = {"Primary School": 0, "High School": 1, "Graduate": 2, "Masters": 3, "Phd": 4}
ENROLLED_MAP = {"no_enrollment": 0, "Part time course": 1, "Full time course": 2}
REL_EXP_MAP = {"Has relevent experience": 1, "No relevent experience": 0}

ORDINAL_MAPS = {"experience": EXPERIENCE_MAP, "last_new_job": LAST_NEW_JOB_MAP,
                "company_size": COMPANY_SIZE_MAP, "education_level": EDUCATION_MAP,
                "enrolled_university": ENROLLED_MAP}
NOMINAL_COLS = ["gender", "major_discipline", "company_type"]
# Ordinales dont la relation à la cible n'est pas monotone → version one-hot en plus
NON_MONOTONIC_COLS = ["company_size", "education_level"]
# Indicateurs de manque uniquement pour les manques informatifs (p < 0,05)
MISSING_FLAG_COLS = ["gender", "major_discipline", "company_size", "company_type",
                     "enrolled_university", "last_new_job"]
ONE_HOT_PREFIXES = ["gender_", "major_discipline_", "company_type_", "company_size_cat_", "education_level_cat_"]
RARE_CITY_THRESHOLD = 20
LOW_CDI_THRESHOLD = 0.625

COLOR_POS, COLOR_NEG, COLOR_MAIN = "#e07a5f", "#81b29a", "#3d405b"


# =============================================================================
# OUTILS
# =============================================================================
def section(title):
    """Affiche un titre de section dans la console."""
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def save_fig(fig, out_dir, name):
    """Enregistre une figure dans out_dir/figures et la ferme."""
    if out_dir is None:
        plt.close(fig)
        return
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)


def ordered_levels(series, col, fallback_index):
    """Ordre d'affichage des modalités : ordre métier si connu, NaN en dernier."""
    order = [o for o in ORDER_MAP.get(col, fallback_index) if o in series.unique()]
    if "NaN" in series.unique() and "NaN" not in order:
        order.append("NaN")
    return order


# =============================================================================
# 1. CHARGEMENT
# =============================================================================
def find_file(name, data_dir=None):
    """Cherche un fichier CSV dans data_dir, puis dans /kaggle/input et le dossier courant."""
    patterns = [os.path.join(data_dir, "**", name)] if data_dir else []
    patterns += [f"/kaggle/input/**/{name}", f"./**/{name}"]
    for p in patterns:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} introuvable — vérifie le chemin des données.")


def load_data(data_dir=None):
    """Charge train, test et sample_submission."""
    section("1. CHARGEMENT ET APERÇU DES DONNÉES")
    train = pd.read_csv(find_file("aug_train.csv", data_dir))
    test = pd.read_csv(find_file("aug_test.csv", data_dir))
    sample_sub = pd.read_csv(find_file("sample_submission.csv", data_dir))
    print(f"Train : {train.shape} | Test : {test.shape} | Sample submission : {sample_sub.shape}")
    return train, test, sample_sub


def overview(train):
    """Types, valeurs non nulles et statistiques descriptives."""
    print("\n--- Types et valeurs non nulles ---")
    train.info()
    print("\n--- Statistiques descriptives (numériques) ---")
    print(train.describe().T)
    print("\n--- Statistiques descriptives (catégorielles) ---")
    print(train.describe(include="object").T)


# =============================================================================
# 2. QUALITÉ DES DONNÉES
# =============================================================================
def check_ids_and_duplicates(train, test, sample_sub):
    """Unicité des ID, recouvrement train/test, doublons et cibles contradictoires."""
    section("2.1 IDENTIFIANTS ET DOUBLONS")
    print("enrollee_id unique dans train :", train[ID_COL].is_unique)
    print("enrollee_id unique dans test  :", test[ID_COL].is_unique)
    print("Recouvrement d'ID train/test  :", len(set(train[ID_COL]) & set(test[ID_COL])))
    print("Test et sample_submission alignés :", set(test[ID_COL]) == set(sample_sub[ID_COL]))

    feat_cols = [c for c in train.columns if c not in [ID_COL, TARGET]]
    dup_profiles = train[train.duplicated(subset=feat_cols, keep=False)]
    conflicts = dup_profiles.groupby(feat_cols, dropna=False)[TARGET].nunique()
    result = {"dup_full": int(train.duplicated().sum()),
              "dup_features": int(train.duplicated(subset=feat_cols).sum()),
              "dup_conflicts": int((conflicts > 1).sum())}
    print(f"\nLignes dupliquées (toutes colonnes)                 : {result['dup_full']}")
    print(f"Lignes dupliquées (mêmes features, ID différent)    : {result['dup_features']}")
    print(f"Groupes de doublons avec des cibles contradictoires : {result['dup_conflicts']}")
    print("→ Profils identiques = personnes différentes : conservés.")
    return result


def missing_table(df):
    """Nombre et pourcentage de valeurs manquantes par colonne."""
    m = df.isna().sum()
    return (pd.DataFrame({"nb_manquants": m, "pct_manquants": (m / len(df) * 100).round(2)})
            .query("nb_manquants > 0").sort_values("pct_manquants", ascending=False))


def missing_report(train, test, out_dir=None):
    """Tableau des valeurs manquantes train/test et graphiques associés."""
    section("2.2 VALEURS MANQUANTES")
    miss = missing_table(train).join(missing_table(test), rsuffix="_test", how="outer").fillna(0)
    miss = miss.sort_values("pct_manquants", ascending=False)
    print(miss)
    rows_with_na = train.isna().any(axis=1).mean()
    print(f"\nLignes avec au moins une valeur manquante : {rows_with_na:.1%}")
    print("→ Supprimer ces lignes ferait perdre trop de données : on imputera.")

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    miss["pct_manquants"].sort_values().plot.barh(ax=axes[0], color=COLOR_POS)
    axes[0].set_title("% de valeurs manquantes (train)")
    axes[0].set_xlabel("%")
    sns.heatmap(train.isna().sample(min(1000, len(train)), random_state=0), cbar=False, ax=axes[1], cmap="viridis")
    axes[1].set_title("Carte des valeurs manquantes (échantillon de 1000 lignes)")
    save_fig(fig, out_dir, "02_valeurs_manquantes")

    fig = plt.figure(figsize=(7, 5))
    sns.heatmap(train[miss.index].isna().corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Corrélation entre les indicateurs de valeurs manquantes")
    save_fig(fig, out_dir, "02_correlation_manques")
    return miss, rows_with_na


def categorical_report(train, test):
    """Modalités de chaque variable catégorielle et modalités absentes du train."""
    section("2.3 MODALITÉS DES VARIABLES CATÉGORIELLES")
    cat_cols = train.select_dtypes("object").columns.tolist()
    for c in cat_cols:
        if c == "city":
            continue
        print(f"\n▶ {c} ({train[c].nunique()} modalités)")
        print(train[c].value_counts(dropna=False).to_string())
    print(f"\n▶ city : {train['city'].nunique()} villes distinctes (haute cardinalité)")

    for c in cat_cols:
        only_test = set(test[c].dropna()) - set(train[c].dropna())
        if only_test:
            print(f"⚠ {c} : modalités présentes uniquement dans le test → {only_test}")


def consistency_checks(train):
    """Une ville = un indice ; croisement éducation × spécialité."""
    section("2.4 COHÉRENCE LOGIQUE")
    cdi_per_city = train.groupby("city")["city_development_index"].nunique()
    print("Villes ayant plusieurs city_development_index :", int((cdi_per_city > 1).sum()))
    print("\nÉducation × spécialité :")
    print(pd.crosstab(train["education_level"].fillna("NaN"), train["major_discipline"].fillna("NaN")))


# =============================================================================
# 3. ANALYSE UNIVARIÉE
# =============================================================================
def target_analysis(train, out_dir=None):
    """Distribution de la cible et ratio de déséquilibre."""
    section("3.1 VARIABLE CIBLE")
    counts = train[TARGET].value_counts().sort_index()
    ratio = counts[0] / counts[1]
    print(counts.to_string())
    print(f"Taux de target=1 : {train[TARGET].mean():.1%} | Ratio de déséquilibre : {ratio:.2f} pour 1")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["0 — ne cherche pas", "1 — cherche un emploi"], counts.values, color=[COLOR_NEG, COLOR_POS])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v}\n({v / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_title("Distribution de la cible")
    ax.set_ylim(0, counts.max() * 1.2)
    save_fig(fig, out_dir, "03_cible")
    return counts, ratio


def iqr_outliers(s):
    """Nombre de valeurs hors de [Q1 - 1,5·IQR ; Q3 + 1,5·IQR]."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def numeric_analysis(train, out_dir=None):
    """Distributions, asymétrie et valeurs extrêmes des variables numériques."""
    section("3.2 VARIABLES NUMÉRIQUES")
    for c in NUM_COLS:
        n = iqr_outliers(train[c])
        print(f"{c:25s} | asymétrie = {train[c].skew():+.2f} | outliers IQR : {n:5d} ({n / len(train):.1%}) "
              f"| min = {train[c].min()} | max = {train[c].max()}")
    print("→ Valeurs extrêmes réalistes (pas des erreurs) : conservées, training_hours sera passé en log1p.")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for i, c in enumerate(NUM_COLS):
        sns.histplot(train[c], kde=True, ax=axes[0, i], color=COLOR_MAIN)
        axes[0, i].set_title(f"{c} — asymétrie = {train[c].skew():.2f}")
        sns.boxplot(x=train[c], ax=axes[1, i], color="#f2cc8f")
    save_fig(fig, out_dir, "03_numeriques")


def categorical_distribution(train, out_dir=None):
    """Effectifs des variables catégorielles et concentration des villes."""
    section("3.3 VARIABLES CATÉGORIELLES")
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, c in zip(axes.flat, PLOT_CATS):
        s = train[c].fillna("NaN")
        sns.countplot(y=s, order=ordered_levels(s, c, s.value_counts().index), ax=ax, color=COLOR_MAIN)
        ax.set_title(c)
        ax.set_xlabel("")
        ax.set_ylabel("")
    save_fig(fig, out_dir, "03_categorielles")

    city_counts = train["city"].value_counts()
    print(f"Top 10 villes = {city_counts.head(10).sum() / len(train):.1%} des candidats")
    print(f"Villes avec moins de {RARE_CITY_THRESHOLD} candidats : "
          f"{(city_counts < RARE_CITY_THRESHOLD).sum()} / {len(city_counts)}")
    fig, ax = plt.subplots(figsize=(12, 4))
    city_counts.head(15).plot.bar(ax=ax, color=COLOR_MAIN, title="Top 15 villes")
    save_fig(fig, out_dir, "03_villes")


# =============================================================================
# 4. ANALYSE BIVARIÉE
# =============================================================================
def target_rate_by_category(train, out_dir=None):
    """Taux de target=1 par modalité de chaque variable catégorielle."""
    section("4.1 TAUX DE RECHERCHE D'EMPLOI PAR MODALITÉ")
    global_rate = train[TARGET].mean()
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, c in zip(axes.flat, PLOT_CATS):
        s = train[c].fillna("NaN")
        g = train.assign(**{c: s}).groupby(c)[TARGET].agg(["mean", "size"])
        g = g.loc[ordered_levels(s, c, g.sort_values("mean").index)]
        print(f"\n▶ {c}\n" + g.rename(columns={"mean": "taux_target", "size": "effectif"})
              .round(3).to_string())
        ax.barh(g.index.astype(str), g["mean"],
                color=[COLOR_POS if m > global_rate else COLOR_NEG for m in g["mean"]])
        ax.axvline(global_rate, ls="--", c="k", lw=1)
        ax.set_title(f"{c} — taux de target=1")
        ax.set_xlim(0, max(0.7, g["mean"].max() * 1.1))
    fig.suptitle(f"Taux de recherche d'emploi par modalité (pointillés = moyenne {global_rate:.1%})",
                 y=1.005, fontsize=14)
    save_fig(fig, out_dir, "04_taux_par_modalite")
    return global_rate


def cramers_v(x, y):
    """V de Cramér corrigé du biais (Bergsma 2013), statistique du χ² et p-value."""
    ct = pd.crosstab(x, y)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    return np.sqrt(phi2 / max(1e-9, min(kc - 1, rc - 1))), chi2, p


def association_table(train, out_dir=None):
    """Test du χ² et V de Cramér entre chaque catégorielle et la cible."""
    section("4.2 FORCE DE L'ASSOCIATION : χ² ET V DE CRAMÉR")
    rows = []
    for c in PLOT_CATS + ["city"]:
        v, chi2, p = cramers_v(train[c].fillna("NaN"), train[TARGET])
        rows.append({"variable": c, "cramers_v": round(v, 3), "chi2": round(chi2, 1), "p_value": p})
    assoc = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    assoc["significatif (5%)"] = assoc["p_value"] < 0.05
    print(assoc.to_string())

    fig = plt.figure(figsize=(8, 4.5))
    sns.barplot(data=assoc, x="cramers_v", y="variable", color=COLOR_MAIN)
    plt.title("Association avec la cible (V de Cramér, NaN traité comme une modalité)")
    save_fig(fig, out_dir, "04_cramers_v")
    return assoc


def numeric_vs_target(train, out_dir=None):
    """Test de Mann-Whitney, médianes par classe et taux par décile de cdi."""
    section("4.3 VARIABLES NUMÉRIQUES VS CIBLE")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, c in zip(axes, NUM_COLS):
        _, p = stats.mannwhitneyu(train.loc[train[TARGET] == 0, c], train.loc[train[TARGET] == 1, c])
        print(f"{c:25s} | Mann-Whitney p = {p:.2e}")
        sns.kdeplot(data=train, x=c, hue=TARGET, common_norm=False, fill=True, ax=ax,
                    palette=[COLOR_NEG, COLOR_POS])
        ax.set_title(f"{c} — Mann-Whitney p = {p:.2e}")
    save_fig(fig, out_dir, "04_numeriques_vs_cible")
    print("\n" + train.groupby(TARGET)[NUM_COLS].median()
          .rename(index={0: "médiane target=0", 1: "médiane target=1"}).to_string())

    deciles = train.groupby(pd.qcut(train["city_development_index"], q=10, duplicates="drop"))[TARGET].mean()
    print("\nTaux de target=1 par décile de city_development_index :\n" + deciles.round(3).to_string())
    fig, ax = plt.subplots(figsize=(10, 4))
    deciles.plot(ax=ax, marker="o", color=COLOR_POS, title="Taux de target=1 par décile de city_development_index")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("taux target=1")
    save_fig(fig, out_dir, "04_cdi_deciles")


def interactions(train, out_dir=None):
    """Taux de target=1 selon deux variables croisées."""
    section("4.4 INTERACTIONS")
    exp_rel = pd.crosstab(train["experience"], train["relevent_experience"],
                          values=train[TARGET], aggfunc="mean").reindex(ORDER_MAP["experience"])
    edu_enr = pd.crosstab(train["education_level"], train["enrolled_university"],
                          values=train[TARGET], aggfunc="mean").reindex(ORDER_MAP["education_level"])
    print("Éducation × inscription université (taux target=1) :\n" + edu_enr.round(3).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    exp_rel.plot(ax=axes[0], marker="o", title="Taux target=1 : expérience × expérience pertinente")
    edu_enr.plot.bar(ax=axes[1], title="Taux target=1 : éducation × inscription université")
    save_fig(fig, out_dir, "04_interactions")


# =============================================================================
# 5. VALEURS MANQUANTES INFORMATIVES
# =============================================================================
def missing_informativeness(train, miss_cols):
    """Compare le taux de target=1 entre lignes manquantes et renseignées (test du χ²)."""
    section("5. LES VALEURS MANQUANTES SONT-ELLES INFORMATIVES ?")
    rows = []
    for c in miss_cols:
        is_na = train[c].isna()
        if is_na.sum() == 0:
            continue
        rows.append({"variable": c,
                     "taux_target_si_manquant": train.loc[is_na, TARGET].mean(),
                     "taux_target_si_renseigne": train.loc[~is_na, TARGET].mean(),
                     "p_value_chi2": stats.chi2_contingency(pd.crosstab(is_na, train[TARGET]))[1]})
    na_info = pd.DataFrame(rows)
    na_info["ecart"] = na_info["taux_target_si_manquant"] - na_info["taux_target_si_renseigne"]
    na_info = na_info.sort_values("ecart", key=abs, ascending=False).reset_index(drop=True)
    na_info["informatif (5%)"] = na_info["p_value_chi2"] < 0.05
    print(na_info.to_string(formatters={
        "taux_target_si_manquant": "{:.1%}".format, "taux_target_si_renseigne": "{:.1%}".format,
        "ecart": "{:+.1%}".format, "p_value_chi2": "{:.2e}".format}))
    return na_info


# =============================================================================
# 6. NETTOYAGE ET TRANSFORMATION
# =============================================================================
class HRCleaner:
    """Nettoyage reproductible : fit sur le train, transform sur train et test."""

    def _harmonize(self, df):
        df = df.copy()
        for c in df.select_dtypes("object").columns:
            df[c] = df[c].str.strip()
        df["company_size"] = df["company_size"].replace({"10/49": "10-49"})
        # NaN de spécialité pour les niveaux pré-universitaires = non applicable
        mask = df["major_discipline"].isna() & df["education_level"].isin(["Primary School", "High School"])
        df.loc[mask, "major_discipline"] = "Not applicable"
        return df

    def fit(self, df):
        df = self._harmonize(df)
        self.ord_medians_ = {c: df[c].map(m).median() for c, m in ORDINAL_MAPS.items()}
        city_counts = df["city"].value_counts()
        self.frequent_cities_ = set(city_counts[city_counts >= RARE_CITY_THRESHOLD].index)
        grouped = df["city"].where(df["city"].isin(self.frequent_cities_), "rare_city")
        self.city_freq_ = grouped.value_counts(normalize=True).to_dict()
        self.cdi_median_ = df["city_development_index"].median()
        self.th_median_ = df["training_hours"].median()
        self.nominal_levels_ = {c: sorted(df[c].fillna("Unknown").unique())
                                for c in NOMINAL_COLS + NON_MONOTONIC_COLS}
        return self

    def transform(self, df):
        df = self._harmonize(df)
        out = pd.DataFrame(index=df.index)
        out[ID_COL] = df[ID_COL]

        # Indicateurs de manque (avant imputation)
        for c in MISSING_FLAG_COLS:
            out[f"{c}_missing"] = df[c].isna().astype(int)
        out["no_company_info"] = (df["company_size"].isna() & df["company_type"].isna()).astype(int)

        # Numériques
        out["city_development_index"] = df["city_development_index"].fillna(self.cdi_median_)
        out["low_cdi"] = (out["city_development_index"] <= LOW_CDI_THRESHOLD).astype(int)
        out["cdi_bin"] = pd.cut(out["city_development_index"], bins=[0, 0.625, 0.775, 0.9, 1.0],
                                labels=False, include_lowest=True)
        out["training_hours"] = df["training_hours"].fillna(self.th_median_)
        out["training_hours_log"] = np.log1p(out["training_hours"])

        # Ville : regroupement des rares + fréquence
        city = df["city"].where(df["city"].isin(self.frequent_cities_), "rare_city")
        out["city_grouped"] = city
        out["city_freq"] = city.map(self.city_freq_).fillna(0)

        # Binaire
        out["relevent_experience"] = df["relevent_experience"].map(REL_EXP_MAP)

        # Ordinales (NaN → médiane du train)
        for c, m in ORDINAL_MAPS.items():
            out[c] = df[c].map(m).fillna(self.ord_medians_[c]).astype(int)

        # One-hot : nominales + ordinales non monotones
        for c in NOMINAL_COLS + NON_MONOTONIC_COLS:
            s = df[c].fillna("Unknown")
            s = s.where(s.isin(self.nominal_levels_[c]), "Unknown")
            prefix = f"{c}_cat" if c in NON_MONOTONIC_COLS else c
            for lvl in self.nominal_levels_[c]:
                out[f"{prefix}_{lvl}".replace(" ", "_")] = (s == lvl).astype(int)

        # Features dérivées
        out["exp_minus_last_job"] = out["experience"] - out["last_new_job"]
        out["is_student"] = (out["enrolled_university"] > 0).astype(int)
        out["n_missing"] = df[MISSING_FLAG_COLS].isna().sum(axis=1)

        if TARGET in df.columns:
            out[TARGET] = df[TARGET].astype(int)
        return out

    def readable(self, df):
        """Version lisible : catégories imputées mais non encodées (pour CatBoost)."""
        d = self._harmonize(df)
        for c in NOMINAL_COLS + list(ORDINAL_MAPS):
            d[c] = d[c].fillna("Unknown")
        d["city"] = d["city"].where(d["city"].isin(self.frequent_cities_), "rare_city")
        return d


def clean_data(train, test):
    """Apprend le nettoyage sur le train et l'applique au train et au test."""
    section("6. NETTOYAGE ET TRANSFORMATION")
    cleaner = HRCleaner().fit(train)
    train_clean, test_clean = cleaner.transform(train), cleaner.transform(test)
    train_readable, test_readable = cleaner.readable(train), cleaner.readable(test)
    sensitive_cols = [c for c in train_clean.columns if c.startswith("gender")]
    print(f"Train nettoyé : {train_clean.shape} | Test nettoyé : {test_clean.shape}")
    print("Colonnes sensibles à exclure de la modélisation :", sensitive_cols)
    return cleaner, train_clean, test_clean, train_readable, test_readable, sensitive_cols


# =============================================================================
# 7. VÉRIFICATIONS APRÈS NETTOYAGE
# =============================================================================
def validate_cleaning(train, test, train_clean, test_clean):
    """Plus de NaN, colonnes alignées, aucune ligne perdue."""
    section("7.1 VÉRIFICATIONS APRÈS NETTOYAGE")
    assert train_clean.drop(columns=["city_grouped"]).isna().sum().sum() == 0, "Il reste des NaN dans le train !"
    assert test_clean.drop(columns=["city_grouped"]).isna().sum().sum() == 0, "Il reste des NaN dans le test !"
    feature_cols = [c for c in train_clean.columns if c not in [ID_COL, TARGET, "city_grouped"]]
    assert list(test_clean[feature_cols].columns) == feature_cols, "Colonnes train/test non alignées"
    assert len(train_clean) == len(train) and len(test_clean) == len(test), "Des lignes ont été perdues"
    print("✔ Aucune valeur manquante, colonnes alignées, aucune ligne perdue.")
    print(f"✔ {len(feature_cols)} variables prêtes pour la modélisation.")
    return feature_cols


def spearman_analysis(train_clean, out_dir=None):
    """Corrélations de Spearman entre variables clés et avec la cible."""
    section("7.2 CORRÉLATIONS DE SPEARMAN")
    core = ["city_development_index", "low_cdi", "training_hours_log", "relevent_experience", "experience",
            "last_new_job", "company_size", "education_level", "enrolled_university", "city_freq",
            "no_company_info", "company_size_missing", "gender_missing", "last_new_job_missing",
            "n_missing", TARGET]
    corr = train_clean[core].corr(method="spearman")
    corr_target = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("Corrélation avec la cible :\n" + corr_target.round(3).to_string())

    fig = plt.figure(figsize=(12, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                mask=np.triu(np.ones_like(corr, dtype=bool), k=1))
    plt.title("Corrélations de Spearman (données nettoyées)")
    save_fig(fig, out_dir, "07_spearman_heatmap")

    fig, ax = plt.subplots(figsize=(8, 5))
    corr_target.plot.barh(ax=ax, color=[COLOR_POS if v > 0 else COLOR_NEG for v in corr_target],
                          title="Corrélation de Spearman avec target")
    ax.invert_yaxis()
    save_fig(fig, out_dir, "07_spearman_cible")
    return corr_target


def high_correlation_pairs(df, cols, threshold=0.7):
    """Paires de variables dont |corrélation de Pearson| dépasse le seuil."""
    c = df[cols].corr().abs()
    c = c.where(np.triu(np.ones(c.shape, dtype=bool), k=1)).stack()
    return c[c > threshold].sort_values(ascending=False)


def multicollinearity(train_clean, feature_cols):
    """Paires de variables très corrélées dans le jeu complet."""
    section("7.3 MULTICOLINÉARITÉ")
    pairs = high_correlation_pairs(train_clean, feature_cols)
    print(f"Paires avec |corrélation| > 0.7 : {len(pairs)}")
    print(pairs.to_string() if len(pairs) else "aucune")
    return pairs


def build_feature_sets(train_clean, feature_cols, sensitive_cols):
    """Jeu complet pour les arbres, jeu réduit sans redondances pour la logistique."""
    section("7.4 JEUX DE VARIABLES POUR LA MODÉLISATION")
    features_tree = [c for c in feature_cols if c not in sensitive_cols]
    drop_linear = (
        [c for c in feature_cols if c.endswith("_Unknown")]                  # doublons des *_missing
        + ["company_size", "education_level"]                                # remplacées par leur one-hot
        + ["company_size_missing", "company_type_missing", "n_missing"]      # résumées par no_company_info
        + ["exp_minus_last_job", "is_student", "cdi_bin", "training_hours"]  # dérivées ≈ source
        + ["company_size_cat_50-99", "company_type_Pvt_Ltd",
           "major_discipline_STEM", "education_level_cat_Graduate"]          # modalités de référence
        + ["major_discipline_Not_applicable"]                                # = High School + Primary School
    )
    features_linear = [c for c in features_tree if c not in drop_linear]
    print(f"Arbres : {len(features_tree)} variables | Logistique : {len(features_linear)} variables")
    remaining = high_correlation_pairs(train_clean, features_linear)
    print("Paires restantes avec |r| > 0.7 :", remaining.round(3).to_dict() or "aucune")
    return features_tree, features_linear


def compute_vif(train_clean, features_linear):
    """Facteur d'inflation de la variance (nécessite statsmodels)."""
    section("7.5 VIF DU JEU LINÉAIRE")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        print("statsmodels non installé : VIF ignoré (pip install statsmodels).")
        return None
    X = train_clean[features_linear].astype(float).assign(const=1.0)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1] - 1)],
                    index=features_linear).sort_values(ascending=False)
    print(vif.head(10).round(2).to_string())
    if vif.max() > 10:
        print("⚠ VIF > 10 :", vif[vif > 10].index.tolist())
    else:
        print("✔ Tous les VIF < 10 : multicolinéarité acceptable pour la régression logistique.")
    return vif


# =============================================================================
# 8. CONTRÔLE FINAL
# =============================================================================
def final_quality_checks(train, train_clean, test_clean, cleaner):
    """Aucune valeur manquante ni aberrante dans les jeux nettoyés."""
    section("8. CONTRÔLE FINAL DU NETTOYAGE")
    checks = {}
    ranges = {"city_development_index": (0, 1), "experience": (0, 21), "last_new_job": (0, 5),
              "company_size": (0, 7), "education_level": (0, 4), "enrolled_university": (0, 2),
              "n_missing": (0, len(MISSING_FLAG_COLS))}

    for name, d in [("train", train_clean), ("test", test_clean)]:
        checks[f"{name} : aucune valeur manquante"] = d.isna().sum().sum() == 0
        checks[f"{name} : training_hours > 0"] = (d["training_hours"] > 0).all()
        for col, (lo, hi) in ranges.items():
            checks[f"{name} : {col} dans [{lo}, {hi}]"] = d[col].between(lo, hi).all()

        binaries = [c for c in d.columns if (c.endswith("_missing") and c != "n_missing")
                    or c.startswith(tuple(ONE_HOT_PREFIXES))] \
            + ["relevent_experience", "low_cdi", "no_company_info", "is_student"]
        checks[f"{name} : variables binaires uniquement 0/1"] = d[binaries].isin([0, 1]).all().all()

        for prefix in ONE_HOT_PREFIXES:
            cols = [c for c in d.columns if c.startswith(prefix) and not c.endswith("_missing")]
            checks[f"{name} : one-hot {prefix[:-1]} cohérent"] = (d[cols].sum(axis=1) == 1).all()

    checks["'10/49' supprimé de company_size"] = "10/49" not in cleaner._harmonize(train)["company_size"].unique()

    for k, v in checks.items():
        print(("✔ " if v else "✘ ") + k)
    ok = all(checks.values())
    print("\n→ NETTOYAGE VALIDÉ" if ok else "\n→ ⚠ des contrôles ont échoué")

    q1, q3 = train_clean["training_hours"].quantile([0.25, 0.75])
    n_ext = int((train_clean["training_hours"] > q3 + 1.5 * (q3 - q1)).sum())
    print(f"\nInfo : {n_ext} valeurs extrêmes (IQR) de training_hours conservées volontairement "
          f"(valeurs réelles, max = {train_clean['training_hours'].max()} h)")
    return ok


# =============================================================================
# 9. EXPORT ET SYNTHÈSE
# =============================================================================
def export_data(out_dir, train_clean, test_clean, train_readable, test_readable, features_tree, features_linear):
    """Enregistre les jeux nettoyés et les listes de variables."""
    section("9.1 EXPORT DES DONNÉES NETTOYÉES")
    os.makedirs(out_dir, exist_ok=True)
    train_clean.to_csv(os.path.join(out_dir, "train_clean_encoded.csv"), index=False)
    test_clean.to_csv(os.path.join(out_dir, "test_clean_encoded.csv"), index=False)
    train_readable.to_csv(os.path.join(out_dir, "train_clean_readable.csv"), index=False)
    test_readable.to_csv(os.path.join(out_dir, "test_clean_readable.csv"), index=False)
    for name, cols in [("features_tree.txt", features_tree), ("features_linear.txt", features_linear)]:
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
            f.write("\n".join(cols))
    print("Fichiers exportés dans", os.path.abspath(out_dir))
    print(" - train/test_clean_encoded.csv  : tout numérique (sklearn / XGBoost / LightGBM)")
    print(" - train/test_clean_readable.csv : catégories en clair (CatBoost)")
    print(" - features_tree.txt / features_linear.txt : listes de variables par type de modèle")


def print_summary(train, test, counts, rows_with_na, miss, assoc, corr_target, na_info):
    """Synthèse automatique des principaux résultats."""
    section("9.2 SYNTHÈSE AUTOMATIQUE")
    print(f"• {len(train)} candidats en train, {len(test)} en test.")
    print(f"• {train[TARGET].mean():.1%} des candidats cherchent un nouvel emploi "
          f"→ jeu déséquilibré ({counts[0] / counts[1]:.1f}:1).")
    print(f"• {rows_with_na:.0%} des lignes ont au moins une valeur manquante ; variable la plus incomplète : "
          f"{miss['pct_manquants'].idxmax()} ({miss['pct_manquants'].max():.1f} %).")
    print("\n• Variables catégorielles les plus associées à la cible (V de Cramér) :")
    for _, r in assoc.head(5).iterrows():
        print(f"    - {r['variable']:22s} V = {r['cramers_v']:.3f}")
    print("\n• Variables nettoyées les plus corrélées à la cible (Spearman) :")
    for k, v in corr_target.head(5).items():
        print(f"    - {k:22s} ρ = {v:+.3f}")
    sig_na = na_info.loc[na_info["p_value_chi2"] < 0.05, "variable"].tolist()
    print(f"\n• Valeurs manquantes informatives (p < 0.05) : {', '.join(sorted(sig_na)) or 'aucune'}")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================
def parse_args():
    default_out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "outputs"
    parser = argparse.ArgumentParser(description="Exploration, analyse et nettoyage — HR Analytics")
    parser.add_argument("--data-dir", default=None, help="Dossier contenant les CSV (recherche auto sinon)")
    parser.add_argument("--out-dir", default=default_out, help="Dossier de sortie (CSV et figures)")
    parser.add_argument("--no-plots", action="store_true", help="Ne pas enregistrer les graphiques")
    args, _ = parser.parse_known_args()  # tolère les arguments ajoutés par Jupyter
    return args


def main():
    args = parse_args()
    fig_dir = None if args.no_plots else args.out_dir

    # 1. Chargement
    train, test, sample_sub = load_data(args.data_dir)
    overview(train)

    # 2. Qualité
    check_ids_and_duplicates(train, test, sample_sub)
    miss, rows_with_na = missing_report(train, test, fig_dir)
    categorical_report(train, test)
    consistency_checks(train)

    # 3. Univarié
    counts, _ = target_analysis(train, fig_dir)
    numeric_analysis(train, fig_dir)
    categorical_distribution(train, fig_dir)

    # 4. Bivarié
    target_rate_by_category(train, fig_dir)
    assoc = association_table(train, fig_dir)
    numeric_vs_target(train, fig_dir)
    interactions(train, fig_dir)

    # 5. Manques informatifs
    na_info = missing_informativeness(train, miss.index.tolist())

    # 6. Nettoyage
    cleaner, train_clean, test_clean, train_readable, test_readable, sensitive_cols = clean_data(train, test)

    # 7. Vérifications
    feature_cols = validate_cleaning(train, test, train_clean, test_clean)
    corr_target = spearman_analysis(train_clean, fig_dir)
    multicollinearity(train_clean, feature_cols)
    features_tree, features_linear = build_feature_sets(train_clean, feature_cols, sensitive_cols)
    compute_vif(train_clean, features_linear)

    # 8. Contrôle final
    if not final_quality_checks(train, train_clean, test_clean, cleaner):
        raise RuntimeError("Le contrôle final a échoué : données non exportées.")

    # 9. Export et synthèse
    export_data(args.out_dir, train_clean, test_clean, train_readable, test_readable,
                features_tree, features_linear)
    print_summary(train, test, counts, rows_with_na, miss, assoc, corr_target, na_info)
    if fig_dir:
        print(f"\nGraphiques enregistrés dans {os.path.abspath(os.path.join(fig_dir, 'figures'))}")

    return train_clean, test_clean, features_tree, features_linear


if __name__ == "__main__":
    main()
