# Projet fil rouge

Du fichier brut Météo-France (station 07015) à une régression linéaire de la température.

## Organisation et exécution

```
projet_meteo.py                  script complet commenté (séances 2 à 4)
data/Fil rouge donnees.csv.csv   données brutes (« ; », « mq » = manquant)
figures/                         12 figures régénérées à chaque exécution
etapes/                          les 3 étapes sans commentaires (figures dans etapes/figures/)
sortie_execution.txt             sortie complète d'une exécution
```

```bash
pip install -r requirements.txt
```

```bash
python projet_meteo.py
```


---

## Séance 2 — Chargement et nettoyage

- **Lignes :** 2 864 chargées pour 2 920 attendues (365 × 8). 56 créneaux sont absents,
  dont 37 en août (trou de près de 5 jours). Aucun doublon.
- **Types :** tout est numérique grâce à `na_values=["mq"]`. Sans ce paramètre, une colonne
  contenant `mq` serait lue en texte. La 60ᵉ colonne, vide, vient du `;` final des lignes.
- **Colonnes gardées :** date, `t`, `pres`, `pmer`, `u`, `dd`, `ff`, `rr3` (plus `td`, pour
  démontrer la fuite d'information). Le DataFrame est plus lisible et le script échoue
  clairement si une colonne manque.
- **Conversion :** nouvelle colonne `t_celsius = t - 273.15`. Valeurs de -5,0 à 33,9 °C,
  plausibles. Les `rr3 = -0.1` sont des **traces de pluie** (< 0,1 mm), mises à 0 mm.
- **Valeurs manquantes :** aucune sur `t`, `pres`, `u`, `ff` ; 5 sur `rr3` (0,17 %).
  Règles prévues : `dropna` sur la température (on n'invente pas la cible) ; interpolation
  de 6 h maximum sur `pres`, `u`, `ff` ; NaN laissés sur la pluie (un 0 ou une moyenne la
  fausserait). Le vrai manque, ce sont les 56 lignes absentes : elles ne sont pas inventées.
- **Agrégation mensuelle :** le cycle saisonnier est cohérent. Janvier est le plus froid
  (5,7 °C), juin le plus chaud (20,4 °C), novembre le plus pluvieux (126,7 mm).
- **Cycle diurne :** minimum à 3 h UTC (9,9 °C), maximum à 15 h UTC (15,6 °C).

## Séance 3 — Exploration et structuration

- **Température (fig. 01) :** les relevés bruts sont peu lisibles (oscillation jour/nuit) ;
  la moyenne mensuelle montre la saison mais masque les vagues de chaleur et de froid.
  Le trou d'août est laissé visible sur la courbe.
- **Précipitations (fig. 02–03) :** 72,5 % de zéros, 8 % de traces, maximum 19 mm en 3 h.
  Un histogramme limité aux relevés pluvieux montre la répartition. Pour le mois le plus
  pluvieux, on prend le **cumul** (somme), pas la moyenne.
- **Corrélations (fig. 04–05) :** humidité -0,61 ; pression -0,06 ; vent +0,02. Le nuage
  température/humidité forme des nappes par saison : le lien mélange le cycle jour/nuit et
  le cycle saisonnier.
- **Note pour la séance 4 :** garder l'humidité (lien le plus fort), le mois et l'heure
  (les deux cycles), plus la pression et le vent (sans fuite). Exclure `td` et `t`.
- **Figures :** `savefig` est appelé avant `plt.close`, sinon le fichier est blanc.
- **Structure :** fonctions à responsabilité unique avec docstring, appelées par `main()` ;
  constantes définies une seule fois ; les fonctions travaillent sur une copie du DataFrame.
- **Erreurs testées :** fichier absent, colonne manquante, valeur aberrante, fichier vide,
  en-tête seul, mauvais séparateur. Chaque cas affiche un message clair et sort avec le
  code 1 (`try` courts, exceptions précises).

## Séance 4 — Régression linéaire

- **Problème :** cible `t_celsius` ; variables `u`, `pres`, `ff`, `mois`, `heure`. Avec
  `mois` et `heure`, un modèle linéaire ne voit qu'une pente, pas un cycle.
- **Fuite d'information :** `t`, `td`, `tn12`, `tx12`… contiennent déjà la température.
- **X et y :** un seul `dropna` sur variables + cible (lignes alignées), 0 ligne perdue,
  `ValueError` si une colonne manque ou s'il reste moins de 50 lignes.
- **Découpage :** 20 % de test, **chronologique** (entraînement du 01/01 au 21/10, test du
  21/10 au 31/12) pour juger une période jamais vue. Le découpage aléatoire (graine 42)
  est plus flatteur, car chaque relevé de test a ses voisins à ±3 h dans l'entraînement.
- **Entraînement :** sur le seul jeu d'entraînement ; utiliser le test fausserait le score.

| Variante | MAE test | R² test | MAE train | MAE référence |
|----------|---------:|--------:|----------:|--------------:|
| **Base, chronologique (principal)** | **12,32 °C** | -12,39 | 3,04 | 5,12 |
| Base, aléatoire | 4,06 °C | 0,47 | 3,94 | 5,66 |
| Sans mois/heure, chronologique | 3,63 °C | -0,88 | 4,34 | 5,12 |
| Humidité seule, chronologique | 2,97 °C | -0,32 | 4,66 | 5,12 |
| Sin/cos, chronologique | 3,14 °C | -0,26 | 2,52 | 5,12 |
| Sin/cos, aléatoire | **2,66 °C** | **0,78** | 2,57 | 5,66 |
| Fuite (+ `td`) | 0,66 °C | 0,95 | 0,49 | 5,12 |

La référence prédit toujours la moyenne de l'entraînement.

- **Évaluation :** le modèle principal fait **pire que la référence**. Il a appris
  « +1,47 °C par mois » sur janvier–octobre et prolonge cette pente en novembre–décembre.
  Ce n'est pas du sur-apprentissage, c'est une extrapolation.
- **R² :** un R² proche de 1 (0,95 avec `td`) signale une fuite d'information. Les R²
  négatifs en chronologique s'expliquent par un test court (2,5 mois) et peu dispersé :
  il vaut mieux comparer la MAE à la référence.
- **Coefficients :** `u` -0,19 °C par % (signe attendu), `pres` -0,08 °C par hPa, `ff`
  instable (+0,11 ou -0,32 selon le découpage). Les unités diffèrent : on ne compare pas
  les coefficients entre eux.
- **Erreurs (fig. 10–12) :** systématiques, pas aléatoires. Biais saisonnier avec `mois`
  brut ; avec sin/cos, il reste une surestimation l'après-midi (+2,5 °C) et lors des coups
  de froid.
- **Meilleure piste :** l'encodage sin/cos du jour et de l'heure fait passer la MAE de
  12,32 à 3,14 °C (chronologique) et de 4,06 à 2,66 °C (aléatoire).
- **Limites :** le modèle ne prévoit rien, car ses variables sont mesurées au même instant
  que la température. Il ignore les masses d'air, ne couvre qu'une station et une année,
  et la relation réelle n'est pas linéaire.

## Questions de l'oral

- **Variables :** l'humidité est la plus corrélée et le mois et l'heure portent les cycles.
  Pas `td` : R² de 0,95 = fuite.
- **Pourquoi séparer :** pour mesurer la performance sur des données non vues.
- **Pourquoi chronologique :** c'est une série temporelle ; l'aléatoire surestime la qualité.
- **Erreur moyenne :** écart moyen en °C. 12,3 °C pour le modèle principal, 3,1 °C avec
  sin/cos, contre 5,1 °C pour la référence.
- **Si on modifie une ligne :** avec `shuffle` activé, la MAE passe de 12,3 à 4,1 °C ;
  ajouter `td` donne R² ≈ 0,95.
- **Limite principale :** `mois` et `heure` bruts ne représentent pas un cycle.

