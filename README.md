# HR Analytics — Job Change of Data Scientists

Exploration, analyse et nettoyage des données du jeu Kaggle
[HR Analytics: Job Change of Data Scientists](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists).

**Objectif :** comprendre quels facteurs (démographie, formation, expérience, entreprise actuelle) poussent un candidat à chercher un nouvel emploi (`target = 1`) après sa formation, et préparer des données propres pour la modélisation.

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `hr_cleaning.ipynb` | Notebook commenté : exploration, analyses statistiques, nettoyage et interprétations |
| `hr_cleaning_pipeline.py` | Script Python reprenant tout le notebook, découpé en fonctions |
| `requirements.txt` | Bibliothèques nécessaires |
| `outputs/figures/` | Graphiques générés par le script |
| `outputs/features_*.txt` | Listes de variables retenues pour chaque type de modèle |

## Démarche

1. **Qualité des données** : identifiants, doublons, valeurs manquantes, modalités, cohérence train/test.
2. **Analyse univariée** : distribution de la cible, des variables numériques et catégorielles.
3. **Analyse bivariée** : taux de changement d'emploi par modalité, χ², V de Cramér, Mann-Whitney.
4. **Valeurs manquantes informatives** : test du lien entre l'absence d'une information et la cible.
5. **Nettoyage** : harmonisation, encodage ordinal et one-hot, imputation, indicateurs de manque, regroupement des villes rares, variables dérivées. Les règles sont apprises sur le train puis appliquées au test.
6. **Vérifications** : corrélations de Spearman, multicolinéarité, VIF, contrôle final (aucune valeur manquante ni aberrante).

## Principaux résultats

- **24,9 %** des candidats cherchent un nouvel emploi : la cible est déséquilibrée (3 pour 1).
- **53 %** des lignes ont au moins une valeur manquante. Pour 6 variables sur 8, le manque est lié à la cible : il a été conservé sous forme d'indicateurs.
- **Facteurs les plus liés au changement d'emploi :**
  - le faible développement de la ville (59 % de candidats en recherche dans les villes les moins développées) ;
  - l'absence d'information sur l'entreprise actuelle (40 % contre 18 %) ;
  - un profil junior ou étudiant (peu d'expérience, inscription à l'université).
- Le nombre d'heures de formation n'a pas d'effet significatif.
- Après nettoyage : 19 158 lignes conservées, aucune valeur manquante ni aberrante, 51 variables pour les modèles à arbres et 33 pour la régression logistique (tous les VIF < 10). Les variables liées au genre sont exclues de la modélisation pour des raisons éthiques.

## Exécution

1. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
2. Télécharger les données depuis [Kaggle](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists) (bouton **Download**, fichier `archive.zip`), puis extraire l'archive dans un dossier `archive/`, à côté du script.
3. Lancer le script :
   ```bash
   python hr_cleaning_pipeline.py --data-dir archive --out-dir outputs
   ```
   Les données nettoyées et les graphiques sont écrits dans `outputs/`. L'option `--no-plots` désactive les graphiques.
