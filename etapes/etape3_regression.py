import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_DONNEES = DOSSIER_PROJET / "data" / "Fil rouge donnees.csv.csv"
DOSSIER_FIGURES = Path(__file__).resolve().parent / "figures" / "etape3"

STATION = "07015"
PAS_DE_TEMPS = "3h"
KELVIN_ZERO = 273.15

COLONNES_UTILES = ["numer_sta", "date", "t", "pres", "pmer", "u", "dd", "ff", "rr3", "td"]

BORNES_PLAUSIBLES = {
    "t_celsius": (-30.0, 45.0),
    "pres": (90000, 106000),
    "u": (0, 100),
    "ff": (0, 60),
}

VARIABLES_MODELE = ["u", "pres", "ff", "mois", "heure"]
CIBLE = "t_celsius"
PROPORTION_TEST = 0.2
GRAINE = 42
MIN_LIGNES_MODELE = 50


def charger_donnees(chemin):
    chemin = Path(chemin)
    if not chemin.is_file():
        raise FileNotFoundError(
            f"Fichier de données introuvable : {chemin}\n"
            "Vérifiez le nom du fichier et qu'il se trouve bien dans le dossier data/."
        )
    try:
        df = pd.read_csv(chemin, sep=";", na_values=["mq"], dtype={"numer_sta": str})
    except pd.errors.EmptyDataError:
        raise ValueError(f"Le fichier {chemin.name} est vide.") from None

    if df.shape[1] < 2:
        raise ValueError(
            f"Le fichier {chemin.name} n'a qu'une colonne : le séparateur n'est "
            "probablement pas « ; »."
        )
    if df.empty:
        raise ValueError(f"Le fichier {chemin.name} ne contient aucune ligne de données.")
    return df


def verifier_chargement(df):
    dates = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S")
    grille = pd.date_range(dates.min().normalize(),
                           dates.max().normalize() + pd.Timedelta("21h"),
                           freq=PAS_DE_TEMPS)
    absents = grille.difference(dates)

    print(f"Dimensions du fichier brut : {df.shape}")
    print(f"Lignes attendues (pas de 3 h sur {dates.min():%d/%m/%Y} - "
          f"{dates.max():%d/%m/%Y}) : {len(grille)}, chargées : {len(df)}, "
          f"créneaux absents : {len(absents)}, doublons de date : {dates.duplicated().sum()}")
    if len(absents):
        print("Créneaux absents par mois :",
              pd.Series(absents.month).value_counts().sort_index().to_dict())
    vides = [c for c in df.columns if df[c].isna().all()]
    print(f"Colonnes entièrement vides (ignorées) : {len(vides)}")
    return {"lignes": len(df), "lignes_attendues": len(grille), "creneaux_absents": len(absents)}


def selectionner_colonnes(df, colonnes=COLONNES_UTILES):
    absentes = [c for c in colonnes if c not in df.columns]
    if absentes:
        raise ValueError(f"Colonnes attendues absentes du fichier : {absentes}")
    return df[colonnes].copy()


def convertir_unites(df):
    df = df.copy()
    try:
        df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S")
    except ValueError as erreur:
        raise ValueError(f"Format de date inattendu (AAAAMMJJHHMMSS attendu) : {erreur}") from None
    df["t_celsius"] = df["t"] - KELVIN_ZERO
    df["trace_pluie"] = df["rr3"] == -0.1
    df["rr3_mm"] = df["rr3"].where(df["rr3"] >= 0, 0.0).where(df["rr3"].notna())
    df["mois"] = df["date"].dt.month
    df["heure"] = df["date"].dt.hour
    df["jour"] = df["date"].dt.date
    return df.sort_values("date").reset_index(drop=True)


def verifier_plausibilite(df, bornes=BORNES_PLAUSIBLES):
    problemes = []
    for colonne, (bas, haut) in bornes.items():
        serie = df[colonne].dropna()
        print(f"  {colonne:<10} min = {serie.min():>9.2f}   max = {serie.max():>9.2f}")
        hors = df.loc[(df[colonne] < bas) | (df[colonne] > haut), ["date", colonne]]
        if not hors.empty:
            problemes.append(f"{colonne} hors de [{bas}, {haut}] : {len(hors)} valeur(s), "
                             f"ex. {hors.iloc[0].to_dict()}")
    if problemes:
        raise ValueError("Valeurs non plausibles détectées :\n  " + "\n  ".join(problemes))


def resume_valeurs_manquantes(df, colonnes):
    nb = df[colonnes].isna().sum()
    return pd.DataFrame({"manquantes": nb, "proportion_%": (100 * nb / len(df)).round(2)})


def traiter_valeurs_manquantes(df):
    df = df.dropna(subset=["t_celsius"]).copy()
    for colonne in ["pres", "u", "ff"]:
        df[colonne] = df[colonne].interpolate(limit=2)
    return df


def agregation_mensuelle(df):
    return df.groupby("mois").agg(
        t_moyenne=("t_celsius", "mean"),
        t_min=("t_celsius", "min"),
        t_max=("t_celsius", "max"),
        humidite_moy=("u", "mean"),
        pluie_totale_mm=("rr3_mm", "sum"),
        nb_releves=("t_celsius", "size"),
    ).round(2)


def agregation_horaire(df):
    return df.groupby("heure").agg(
        t_moyenne=("t_celsius", "mean"),
        humidite_moy=("u", "mean"),
    ).round(2)


def enregistrer_figure(fig, nom, dossier=DOSSIER_FIGURES):
    dossier.mkdir(parents=True, exist_ok=True)
    chemin = dossier / nom
    fig.savefig(chemin, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if chemin.stat().st_size == 0:
        raise OSError(f"La figure {chemin} est vide.")
    return chemin


def figure_evolution_temperature(df, mensuel):
    serie = df.set_index("date")["t_celsius"].asfreq(PAS_DE_TEMPS)
    journaliere = serie.resample("D").mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    ax1.plot(serie.index, serie, lw=0.5, color="tab:blue", alpha=0.6, label="Relevé (pas de 3 h)")
    ax1.plot(journaliere.index, journaliere, lw=1.5, color="tab:red", label="Moyenne journalière")
    ax1.set_title("Température à Lille-Lesquin en 2023 (trous = relevés absents)")
    ax1.set_ylabel("Température (°C)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(mensuel.index, mensuel["t_moyenne"], marker="o", color="tab:red", label="Moyenne")
    ax2.fill_between(mensuel.index, mensuel["t_min"], mensuel["t_max"],
                     color="tab:orange", alpha=0.25, label="Min - max du mois")
    ax2.set_title("Agrégation mensuelle de la température")
    ax2.set_xlabel("Mois")
    ax2.set_ylabel("Température (°C)")
    ax2.set_xticks(range(1, 13))
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return enregistrer_figure(fig, "01_evolution_temperature.png")


def resume_precipitations(df):
    rr3 = df["rr3"].dropna()
    return {
        "min_brut": rr3.min(),
        "max": rr3.max(),
        "nb_traces_-0.1": int((rr3 == -0.1).sum()),
        "part_zero_%": round(100 * (rr3 == 0).mean(), 1),
        "part_trace_%": round(100 * (rr3 == -0.1).mean(), 1),
        "part_pluie_%": round(100 * (rr3 > 0).mean(), 1),
        "manquantes": int(df["rr3"].isna().sum()),
    }


def figures_precipitations(df, mensuel):
    pluie = df.loc[df["rr3_mm"] > 0, "rr3_mm"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.hist(df["rr3_mm"].dropna(), bins=40, color="grey")
    ax1.set_title("Toutes les valeurs : la barre « 0 mm » domine")
    ax1.set_xlabel("Précipitations sur 3 h (mm)")
    ax1.set_ylabel("Nombre de relevés")
    ax2.hist(pluie, bins=40, color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_title(f"Relevés pluvieux uniquement (> 0 mm, n = {len(pluie)})")
    ax2.set_xlabel("Précipitations sur 3 h (mm)")
    ax2.set_ylabel("Nombre de relevés (échelle log)")
    fig.tight_layout()
    chemin1 = enregistrer_figure(fig, "02_precipitations_histogramme.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(mensuel.index, mensuel["pluie_totale_mm"], color="tab:blue")
    ax.set_title("Cumul mensuel des précipitations — Lille-Lesquin 2023")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Cumul (mm)")
    ax.set_xticks(range(1, 13))
    ax.grid(axis="y", alpha=0.3)
    chemin2 = enregistrer_figure(fig, "03_precipitations_mensuelles.png")
    return [chemin1, chemin2]


def matrice_correlation(df, colonnes):
    return df[colonnes].corr().round(3)


def figure_correlations(correlations):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlations)), correlations.columns, rotation=45)
    ax.set_yticks(range(len(correlations)), correlations.index)
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Coefficient de corrélation")
    ax.set_title("Corrélations entre variables météo")
    return enregistrer_figure(fig, "04_matrice_correlation.png")


def figure_nuages_points(df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    nuage = axes[0].scatter(df["u"], df["t_celsius"], c=df["mois"], cmap="twilight", s=6)
    axes[0].set_xlabel("Humidité relative (%)")
    axes[0].set_title("Température vs humidité (couleur = mois)")
    axes[1].scatter(df["u"], df["t_celsius"], c=df["heure"], cmap="viridis", s=6)
    axes[1].set_xlabel("Humidité relative (%)")
    axes[1].set_title("Température vs humidité (couleur = heure)")
    axes[2].scatter(df["pres"] / 100, df["t_celsius"], c=df["mois"], cmap="twilight", s=6)
    axes[2].set_xlabel("Pression station (hPa)")
    axes[2].set_title("Température vs pression (couleur = mois)")
    for ax in axes:
        ax.set_ylabel("Température (°C)")
        ax.grid(alpha=0.3)
    fig.colorbar(nuage, ax=axes[0], label="Mois")
    fig.tight_layout()
    return enregistrer_figure(fig, "05_nuages_points.png")


def figures_complementaires(df, mensuel, horaire):
    chemins = []

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.boxplot([df.loc[df["mois"] == m, "t_celsius"] for m in range(1, 13)])
    ax.set_xticks(range(1, 13), range(1, 13))
    ax.set_title("Distribution de la température par mois")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Température (°C)")
    ax.grid(axis="y", alpha=0.3)
    chemins.append(enregistrer_figure(fig, "06_boites_temperature_mois.png"))

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(mensuel.index, mensuel["t_moyenne"], marker="o", color="tab:red")
    ax1.set_ylabel("Température moyenne (°C)", color="tab:red")
    ax1.set_xlabel("Mois")
    ax1.set_xticks(range(1, 13))
    ax2 = ax1.twinx()
    ax2.plot(mensuel.index, mensuel["humidite_moy"], marker="s", color="tab:blue")
    ax2.set_ylabel("Humidité moyenne (%)", color="tab:blue")
    ax1.set_title("Température et humidité mensuelles moyennes")
    chemins.append(enregistrer_figure(fig, "07_temperature_humidite_double_axe.png"))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(horaire.index, horaire["t_moyenne"], marker="o")
    ax.set_xticks(horaire.index)
    ax.set_title("Cycle diurne : température moyenne par heure de relevé (UTC)")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Température moyenne (°C)")
    ax.grid(alpha=0.3)
    chemins.append(enregistrer_figure(fig, "08_cycle_diurne.png"))

    attendus = pd.Series(pd.date_range("2023-01-01", "2023-12-31 21:00", freq=PAS_DE_TEMPS))
    par_mois_attendu = attendus.dt.month.value_counts().sort_index()
    taux = 100 * (1 - mensuel["nb_releves"] / par_mois_attendu)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(taux.index, taux, color="tab:purple")
    ax.set_xticks(range(1, 13))
    ax.set_title("Proportion de relevés absents par mois")
    ax.set_xlabel("Mois")
    ax.set_ylabel("Relevés absents (%)")
    chemins.append(enregistrer_figure(fig, "09_releves_absents_par_mois.png"))
    return chemins


def ajouter_variables_cycliques(df):
    df = df.copy()
    jour_annee = df["date"].dt.day_of_year
    df["mois_sin"] = np.sin(2 * np.pi * jour_annee / 365)
    df["mois_cos"] = np.cos(2 * np.pi * jour_annee / 365)
    df["heure_sin"] = np.sin(2 * np.pi * df["heure"] / 24)
    df["heure_cos"] = np.cos(2 * np.pi * df["heure"] / 24)
    return df


def preparer_variables(df, variables, cible):
    colonnes = variables + [cible]
    absentes = [c for c in colonnes if c not in df.columns]
    if absentes:
        raise ValueError(f"Colonnes absentes pour le modèle : {absentes}")
    non_numeriques = [c for c in colonnes if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeriques:
        raise ValueError(f"Colonnes non numériques pour le modèle : {non_numeriques}")

    donnees = df[colonnes + ["date"]].dropna(subset=colonnes)
    perdues = len(df) - len(donnees)
    print(f"  Lignes avant/après suppression des NaN : {len(df)} -> {len(donnees)} "
          f"({perdues} perdue(s), {100 * perdues / len(df):.2f} %)")
    if len(donnees) < MIN_LIGNES_MODELE:
        raise ValueError(f"Trop peu de lignes exploitables ({len(donnees)}) pour entraîner un modèle.")
    return donnees[variables], donnees[cible], donnees["date"]


def separer(X, y, dates, chronologique=True, proportion_test=PROPORTION_TEST, graine=GRAINE):
    if chronologique:
        return train_test_split(X, y, dates, test_size=proportion_test, shuffle=False)
    return train_test_split(X, y, dates, test_size=proportion_test, random_state=graine)


def entrainer_et_evaluer(X_train, X_test, y_train, y_test):
    modele = LinearRegression().fit(X_train, y_train)
    y_pred = modele.predict(X_test)
    y_pred_train = modele.predict(X_train)
    reference = DummyRegressor(strategy="mean").fit(X_train, y_train)
    y_ref = reference.predict(X_test)
    metriques = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "r2_train": r2_score(y_train, y_pred_train),
        "mae_reference": mean_absolute_error(y_test, y_ref),
        "r2_reference": r2_score(y_test, y_ref),
    }
    return modele, y_pred, metriques


def afficher_resultats(nom, modele, variables, metriques, dates_train, dates_test):
    print(f"\n--- {nom} ---")
    print(f"  Entraînement : {len(dates_train)} lignes, du {dates_train.min():%d/%m/%Y} "
          f"au {dates_train.max():%d/%m/%Y}")
    print(f"  Test         : {len(dates_test)} lignes, du {dates_test.min():%d/%m/%Y} "
          f"au {dates_test.max():%d/%m/%Y}")
    print(f"  Ordonnée à l'origine : {modele.intercept_:.3f} °C")
    for variable, coef in zip(variables, modele.coef_):
        print(f"  Coefficient {variable:<10}: {coef:+.5f} °C par unité")
    print(f"  MAE test  = {metriques['mae']:.2f} °C   (référence moyenne : "
          f"{metriques['mae_reference']:.2f} °C)")
    print(f"  RMSE test = {metriques['rmse']:.2f} °C")
    print(f"  R² test   = {metriques['r2']:.3f}   (référence : {metriques['r2_reference']:.3f})")
    print(f"  MAE train = {metriques['mae_train']:.2f} °C   R² train = {metriques['r2_train']:.3f}")


def figure_predictions(y_test, y_pred, dates_test, titre, nom_fichier):
    erreurs = pd.Series(np.asarray(y_pred) - np.asarray(y_test), index=dates_test.values).sort_index()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.scatter(y_pred, y_test, s=6, alpha=0.5)
    bornes = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(bornes, bornes, color="red", lw=1, label="Prédiction parfaite")
    ax.set_xlabel("Température prédite (°C)")
    ax.set_ylabel("Température réelle (°C)")
    ax.set_title("Prédit vs réel (jeu de test)")
    ax.legend()

    ax = axes[0, 1]
    heures = erreurs.index.hour
    ax.boxplot([erreurs[heures == h] for h in sorted(set(heures))])
    ax.set_xticks(range(1, len(set(heures)) + 1), sorted(set(heures)))
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Erreur prédit - réel (°C)")
    ax.set_title("Erreurs par heure de la journée")

    ax = axes[1, 0]
    mois = erreurs.index.month
    ax.boxplot([erreurs[mois == m] for m in sorted(set(mois))])
    ax.set_xticks(range(1, len(set(mois)) + 1), sorted(set(mois)))
    ax.axhline(0, color="red", lw=1)
    ax.set_xlabel("Mois")
    ax.set_ylabel("Erreur prédit - réel (°C)")
    ax.set_title("Erreurs par mois")

    ax = axes[1, 1]
    ax.scatter(erreurs.index, erreurs.values, s=3, alpha=0.4, label="Erreur (relevé)")
    ax.plot(erreurs.resample("W").mean(), color="tab:red", lw=1.5, label="Moyenne hebdomadaire")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Erreur prédit - réel (°C)")
    ax.set_title("Erreurs au cours du temps")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)

    fig.suptitle(titre)
    fig.tight_layout()
    return enregistrer_figure(fig, nom_fichier)


def comparer_variantes(df):
    variantes = [
        ("Base, découpage chronologique", VARIABLES_MODELE, True),
        ("Base, découpage aléatoire", VARIABLES_MODELE, False),
        ("Sans variables calendaires (u, pres, ff)", ["u", "pres", "ff"], True),
        ("Humidité seule", ["u"], True),
        ("Cyclique sin/cos, chronologique", ["u", "pres", "ff", "mois_sin", "mois_cos",
                                             "heure_sin", "heure_cos"], True),
        ("Cyclique sin/cos, aléatoire", ["u", "pres", "ff", "mois_sin", "mois_cos",
                                         "heure_sin", "heure_cos"], False),
        ("FUITE : avec td (point de rosée)", VARIABLES_MODELE + ["td"], True),
    ]
    lignes = []
    for nom, variables, chrono in variantes:
        print(f"\n  Variante : {nom}")
        X, y, dates = preparer_variables(df, variables, CIBLE)
        X_tr, X_te, y_tr, y_te, _, _ = separer(X, y, dates, chronologique=chrono)
        _, _, m = entrainer_et_evaluer(X_tr, X_te, y_tr, y_te)
        lignes.append({"variante": nom, "mae_test": m["mae"], "rmse_test": m["rmse"],
                       "r2_test": m["r2"], "mae_train": m["mae_train"],
                       "mae_reference": m["mae_reference"]})
    return pd.DataFrame(lignes).set_index("variante").round(3)


def main(chemin=FICHIER_DONNEES):
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    print("=" * 70, "\nSÉANCE 2 — Chargement et nettoyage\n" + "=" * 70)
    brut = charger_donnees(chemin)
    verifier_chargement(brut)
    df = selectionner_colonnes(brut)
    print("\nTypes des colonnes utiles dans le fichier brut :")
    print(df.dtypes.to_string())
    if not (df["numer_sta"] == STATION).all():
        print(f"Attention : le fichier contient d'autres stations que {STATION}.")
    df = convertir_unites(df)

    print("\nValeurs manquantes avant nettoyage :")
    colonnes_interet = ["t_celsius", "pres", "u", "ff", "rr3_mm"]
    print(resume_valeurs_manquantes(df, colonnes_interet).to_string())
    print(f"Relevés « trace de pluie » (rr3 = -0.1) convertis en 0 mm : {df['trace_pluie'].sum()}")

    print("\nContrôle de plausibilité (après conversion) :")
    verifier_plausibilite(df)

    df = traiter_valeurs_manquantes(df)
    print("\nValeurs manquantes après nettoyage :")
    print(resume_valeurs_manquantes(df, colonnes_interet).to_string())
    print(f"Dimensions du DataFrame nettoyé : {df.shape}")

    mensuel = agregation_mensuelle(df)
    horaire = agregation_horaire(df)
    print("\nAgrégation mensuelle :")
    print(mensuel.to_string())
    print(f"Mois le plus chaud : {mensuel['t_moyenne'].idxmax()}, "
          f"le plus froid : {mensuel['t_moyenne'].idxmin()}")
    print("\nAgrégation horaire (cycle diurne, heures UTC) :")
    print(horaire.to_string())

    print("\n" + "=" * 70, "\nSÉANCE 3 — Exploration visuelle\n" + "=" * 70)
    figures = [figure_evolution_temperature(df, mensuel)]
    print("Précipitations :", resume_precipitations(df))
    print(f"Mois le plus pluvieux (cumul) : {mensuel['pluie_totale_mm'].idxmax()} "
          f"({mensuel['pluie_totale_mm'].max():.1f} mm)")
    figures += figures_precipitations(df, mensuel)

    correlations = matrice_correlation(df, ["t_celsius", "pres", "pmer", "u", "ff"])
    print("\nMatrice de corrélation :")
    print(correlations.to_string())
    print("\nCorrélation avec la température, triée par force :")
    print(correlations["t_celsius"].drop("t_celsius").abs().sort_values(ascending=False).to_string())
    figures.append(figure_correlations(correlations))
    figures.append(figure_nuages_points(df))
    figures += figures_complementaires(df, mensuel, horaire)

    print("\n" + "=" * 70, "\nSÉANCE 4 — Régression linéaire\n" + "=" * 70)
    df = ajouter_variables_cycliques(df)
    print(f"Cible : {CIBLE} ; variables : {VARIABLES_MODELE}")
    X, y, dates = preparer_variables(df, VARIABLES_MODELE, CIBLE)
    print(f"  Dimensions : X = {X.shape}, y = {y.shape}, "
          f"index alignés : {X.index.equals(y.index)}")
    print(f"  Amplitude de la température sur l'année : {y.min():.1f} à {y.max():.1f} °C "
          f"(écart-type {y.std():.2f} °C)")

    resultats = {}
    for nom, chrono, fichier in [
        ("Modèle principal — découpage chronologique", True, "10_erreurs_modele_chronologique.png"),
        ("Variante — découpage aléatoire", False, "11_erreurs_modele_aleatoire.png"),
    ]:
        X_tr, X_te, y_tr, y_te, d_tr, d_te = separer(X, y, dates, chronologique=chrono)
        modele, y_pred, metriques = entrainer_et_evaluer(X_tr, X_te, y_tr, y_te)
        afficher_resultats(nom, modele, VARIABLES_MODELE, metriques, d_tr, d_te)
        figures.append(figure_predictions(y_te, y_pred, d_te, nom, fichier))
        resultats[nom] = metriques

    variables_cycl = ["u", "pres", "ff", "mois_sin", "mois_cos", "heure_sin", "heure_cos"]
    Xc, yc, dc = preparer_variables(df, variables_cycl, CIBLE)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = separer(Xc, yc, dc, chronologique=True)
    modele_c, y_pred_c, metriques_c = entrainer_et_evaluer(X_tr, X_te, y_tr, y_te)
    afficher_resultats("Variante — encodage cyclique, chronologique", modele_c,
                       variables_cycl, metriques_c, d_tr, d_te)
    figures.append(figure_predictions(y_te, y_pred_c, d_te,
                                      "Variante cyclique sin/cos — découpage chronologique",
                                      "12_erreurs_modele_cyclique.png"))

    print("\nComparaison des variantes :")
    recap = comparer_variantes(df)
    print("\n" + recap.to_string())

    print("\nFigures générées :")
    for chemin_figure in figures:
        print("  ", chemin_figure.relative_to(DOSSIER_PROJET))


if __name__ == "__main__":
    fichier = sys.argv[1] if len(sys.argv) > 1 else FICHIER_DONNEES
    try:
        main(fichier)
    except (FileNotFoundError, ValueError) as erreur:
        print(f"\nERREUR : {erreur}", file=sys.stderr)
        sys.exit(1)
