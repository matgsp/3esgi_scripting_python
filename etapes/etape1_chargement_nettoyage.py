from pathlib import Path

import pandas as pd

FICHIER_DONNEES = Path(__file__).resolve().parent.parent / "data" / "Fil rouge donnees.csv.csv"

pd.set_option("display.width", 140)

df_brut = pd.read_csv(FICHIER_DONNEES, sep=";", na_values=["mq"], dtype={"numer_sta": str})

print("Dimensions du fichier brut :", df_brut.shape)
lignes_attendues = 365 * 8
print(f"Lignes attendues : {lignes_attendues}, lignes chargées : {len(df_brut)}, "
      f"écart : {lignes_attendues - len(df_brut)}")

dates_brutes = pd.to_datetime(df_brut["date"].astype(str), format="%Y%m%d%H%M%S")
grille = pd.date_range("2023-01-01 00:00", "2023-12-31 21:00", freq="3h")
absents = grille.difference(dates_brutes)
print("Créneaux absents :", len(absents))
print("Créneaux absents par mois :", pd.Series(absents.month).value_counts().sort_index().to_dict())
print("Doublons de date :", dates_brutes.duplicated().sum())

colonnes_utiles = ["numer_sta", "date", "t", "pres", "pmer", "u", "dd", "ff", "rr3"]
df = df_brut[colonnes_utiles].copy()
print("\nTypes des colonnes utiles :")
print(df.dtypes.to_string())

df["t_celsius"] = df["t"] - 273.15
print(f"\nTempérature (°C) : min = {df['t_celsius'].min():.2f}, max = {df['t_celsius'].max():.2f}")
print(f"Pression (hPa) : min = {df['pres'].min() / 100:.1f}, max = {df['pres'].max() / 100:.1f}")
print(f"Humidité (%) : min = {df['u'].min()}, max = {df['u'].max()}")
print(f"Vent (m/s) : min = {df['ff'].min()}, max = {df['ff'].max()}")

print("Traces de pluie (rr3 = -0.1) :", (df["rr3"] == -0.1).sum())
df["rr3_mm"] = df["rr3"].where(df["rr3"] >= 0, 0.0).where(df["rr3"].notna())

colonnes_interet = ["t_celsius", "pres", "u", "ff", "rr3_mm"]
manquantes = df[colonnes_interet].isna().sum()
print("\nValeurs manquantes avant nettoyage :")
print(pd.DataFrame({"manquantes": manquantes,
                    "proportion_%": (100 * manquantes / len(df)).round(2)}).to_string())

df = df.dropna(subset=["t_celsius"])
for colonne in ["pres", "u", "ff"]:
    df[colonne] = df[colonne].interpolate(limit=2)

print("\nValeurs manquantes après nettoyage :")
print(df[colonnes_interet].isna().sum().to_string())

df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d%H%M%S")
df["mois"] = df["date"].dt.month
df["heure"] = df["date"].dt.hour

stats_mois = df.groupby("mois").agg(
    t_moyenne=("t_celsius", "mean"),
    t_min=("t_celsius", "min"),
    t_max=("t_celsius", "max"),
    pluie_totale_mm=("rr3_mm", "sum"),
    nb_releves=("t_celsius", "size"),
).round(2)
print("\nAgrégation mensuelle :")
print(stats_mois.to_string())
print("Mois le plus chaud :", stats_mois["t_moyenne"].idxmax())
print("Mois le plus froid :", stats_mois["t_moyenne"].idxmin())

stats_heure = df.groupby("heure").agg(
    t_moyenne=("t_celsius", "mean"),
    humidite_moy=("u", "mean"),
).round(2)
print("\nAgrégation horaire (heures UTC) :")
print(stats_heure.to_string())

releves_attendus_mois = pd.Series(grille.month).value_counts().sort_index()
taux_absents = (100 * (1 - stats_mois["nb_releves"] / releves_attendus_mois)).round(1)
print("\nProportion de relevés absents par mois (%) :")
print(taux_absents.to_string())
