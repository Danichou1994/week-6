# 🏥 HealthConnect Clinic - Week 6 Data Science

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AnalystLab Africa](https://img.shields.io/badge/AnalystLab-Africa-orange.svg)](https://analystlab.africa)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange.svg)](https://scikit-learn.org/)

---

## 📋 Table des Matières

1. [Présentation du Projet](#-présentation-du-projet)
2. [Structure du Dépôt](#-structure-du-dépôt)
3. [Installation et Configuration](#-installation-et-configuration)
4. [Contexte Métier](#-contexte-métier)
5. [Analyse des Données](#-analyse-des-données)
6. [Feature Engineering](#-feature-engineering)
7. [Modélisation](#-modélisation)
8. [Résultats Week 5](#-résultats-week-5)
9. [Résultats Week 6](#-résultats-week-6)
10. [Comparaison des Modèles](#-comparaison-des-modèles)
11. [Cross-Validation](#-cross-validation)
12. [Optimisation du Seuil](#-optimisation-du-seuil)
13. [Analyse des Erreurs](#-analyse-des-erreurs)
14. [Cross-Track Collaboration](#-cross-track-collaboration)
15. [Validation et Recommandations](#-validation-et-recommandations)
16. [Technologies Utilisées](#-technologies-utilisées)
17. [Structure des Fichiers](#-structure-des-fichiers)
18. [Équipe et Contact](#-équipe-et-contact)
19. [Licence](#-licence)
20. [Tags](#-tags)

---

## 🎯 Présentation du Projet

### Contexte Général

**HealthConnect Clinic** est un prestataire de soins fictif qui gère des services basés sur des rendez-vous. La clinique fait face à un défi majeur : **un taux élevé de rendez-vous manqués (No-Show)** estimé à **45%**, ce qui entraîne :

| Problème | Impact Business |
|----------|-----------------|
| ❌ Créneaux inutilisés | Perte de revenus estimée à 45% des créneaux |
| ❌ Délais d'attente longs | Patients insatisfaits, délais de RDV allongés |
| ❌ Inefficacité opérationnelle | Personnel sous-utilisé, planning désorganisé |
| ❌ Suivi médical interrompu | Risques pour la santé des patients |

### Objectif du Projet

Développer un modèle de **Machine Learning** pour prédire les rendez-vous manqués et permettre à la clinique de :

- ✅ Identifier les patients à risque avec un score de probabilité
- ✅ Mettre en place des interventions ciblées (rappel téléphonique, SMS, email)
- ✅ Optimiser l'utilisation des créneaux (surdoublonnage, liste d'attente)
- ✅ Améliorer l'expérience patient en réduisant les délais
- ✅ Réduire le taux de No-Show de 45% à moins de 30%

### Problème ML

| Élément | Description |
|---------|-------------|
| **Type de problème** | Classification binaire supervisée |
| **Variable cible** | `no_show` (1 = rendez-vous manqué, 0 = patient présent) |
| **Métrique principale** | F1-Score (équilibre entre précision et rappel) |
| **Métriques secondaires** | Accuracy, Precision, Recall, ROC-AUC |
| **Approche** | Baseline avec Logistic Regression |
| **Objectif Week 6** | Améliorer, intégrer et valider le baseline |

---

## 📁 Structure du Dépôt
HealthConnect-Week5-DataScience/
│
├── 📁 code/
│ ├── Week5_DataScience_Notebook.ipynb # Notebook Jupyter Week 5
│ ├── Week6_DataScience_Notebook.ipynb # Notebook Jupyter Week 6
│ └── healthconnect_utils.py # Fonctions utilitaires
│
├── 📁 data/
│ ├── HealthConnect_Appointment_Data.csv # Données brutes (5 000 lignes)
│ └── HealthConnect_Prepared_Data.csv # Données préparées (4 400 lignes)
│
├── 📁 figures/
│ ├── target_distribution.png # Distribution de la cible
│ ├── feature_importance.png # Importance des features (W5)
│ ├── correlations.png # Matrice de corrélation
│ ├── confusion_matrix.png # Matrice de confusion (W5)
│ ├── roc_curve.png # Courbe ROC (W5)
│ ├── pr_curve.png # Courbe Precision-Recall (W5)
│ ├── error_distribution.png # Distribution des erreurs (W5)
│ ├── threshold_analysis.png # Analyse des seuils (W5)
│ ├── week6_model_comparison.png # Comparaison des modèles (W6)
│ ├── week6_confusion_matrix.png # Matrice de confusion (W6)
│ ├── week6_feature_importance.png # Feature importance (W6)
│ └── week6_roc_comparison.png # Courbes ROC comparatives (W6)
│
├── 📁 outputs/
│ ├── healthconnect_model.pkl # Modèle Week 5
│ ├── healthconnect_scaler.pkl # Scaler Week 5
│ ├── week6_best_model.pkl # Modèle Week 6 (Baseline LR)
│ └── week6_scaler.pkl # Scaler Week 6
│
├── 📁 docs/
│ ├── Week5_DataScience_Report.pdf # Rapport Week 5
│ ├── Week6_DataScience_Report.pdf # Rapport Week 6
│ └── Week6_DataScience_Report.tex # Source LaTeX Week 6
│
├── 📄 README.md # Ce fichier
├── 📄 requirements.txt # Dépendances Python
├── 📄 .gitignore # Fichiers ignorés par Git
└── 📄 LICENSE # Licence MIT

text

---

## 🚀 Installation et Configuration

### Prérequis

| Logiciel | Version | Lien |
|----------|---------|------|
| Python | 3.8+ | [python.org](https://python.org) |
| Git | 2.30+ | [git-scm.com](https://git-scm.com) |
| Conda (optionnel) | 4.10+ | [conda.io](https://conda.io) |
| Jupyter Notebook | - | [jupyter.org](https://jupyter.org) |

### 1. Cloner le Dépôt

```bash
git clone https://github.com/votre-username/HealthConnect-Week5-DataScience.git
cd HealthConnect-Week5-DataScience
2. Créer un Environnement Virtuel
bash
# Option 1 : Avec Conda
conda create -n healthconnect python=3.9
conda activate healthconnect

# Option 2 : Avec venv (Linux/Mac)
python -m venv venv
source venv/bin/activate

# Option 3 : Avec venv (Windows)
python -m venv venv
venv\Scripts\activate
3. Installer les Dépendances
bash
pip install -r requirements.txt
4. Lancer les Notebooks
bash
# Week 5
jupyter notebook code/Week5_DataScience_Notebook.ipynb

# Week 6
jupyter notebook code/Week6_DataScience_Notebook.ipynb
💼 Contexte Métier
Problématique HealthConnect
HealthConnect Clinic gère un volume important de rendez-vous médicaux. Le taux de No-Show élevé (45%) génère :

Des pertes financières : Les créneaux non utilisés ne génèrent pas de revenus

Une inefficacité opérationnelle : Le personnel médical est sous-utilisé

Une mauvaise expérience patient : Les patients assidus subissent des délais plus longs

Des risques médicaux : Les patients à risque ne sont pas suivis

Objectifs Business
Objectif	KPI	Cible
Réduire les No-Show	Taux de No-Show	< 30%
Améliorer l'efficacité	Utilisation des créneaux	> 85%
Améliorer l'expérience	Délai d'attente moyen	< 7 jours
Cibler les interventions	Précision du ciblage	> 60%
📊 Analyse des Données
Structure du Dataset
Le dataset contient 5 000 rendez-vous avec 20 colonnes.

Colonne	Type	Description	Valeurs Manquantes
appointment_id	object	Identifiant unique du rendez-vous	0
patient_id	object	Identifiant unique du patient	0
gender	object	Genre (Female/Male/Prefer not to say)	0
age	int64	Âge du patient (18-80 ans)	0
age_group	object	Groupe d'âge	0
appointment_type	object	Type de RDV	0
booking_date	datetime	Date de réservation	0
appointment_date	datetime	Date du rendez-vous	0
appointment_day	object	Jour de la semaine	0
appointment_time	object	Période (Morning/Afternoon/Evening)	0
booking_lead_days	int64	Délai de réservation (jours)	0
previous_appointments	int64	Nombre de RDV précédents	0
previous_no_shows	int64	Nombre d'absences précédentes	0
reminder_sent	object	Rappel envoyé (Yes/No)	0
reminder_channel	object	Canal de rappel	1
distance_to_clinic_km	float64	Distance à la clinique (km)	25
waiting_time_minutes	float64	Temps d'attente (minutes)	18
appointment_outcome	object	Résultat (Attended/No-Show/Cancelled)	0
Statistiques Descriptives
Variable	Moyenne	Médiane	Min	Max	Écart-type
age	42.3	41.0	18	80	15.2
booking_lead_days	38.7	38.0	0	60	16.4
previous_appointments	3.2	3.0	0	11	2.1
previous_no_shows	0.9	1.0	0	5	1.2
distance_to_clinic_km	10.2	8.7	0.5	45.0	7.8
waiting_time_minutes	25.8	25.0	2	64	12.3
Distribution de la Variable Cible
Outcome	Nombre	Pourcentage
Attended	2 150	43.0%
No-Show	2 250	45.0%
Cancelled	600	12.0%
Total	5 000	100%
📌 Interprétation :

Le taux de No-Show est de 45%, ce qui justifie pleinement le projet

Les rendez-vous annulés représentent 12% des données

La classe cible est relativement équilibrée (43% vs 45%)

Traitement des Données
Valeurs Manquantes
Colonne	Manquantes	Traitement	Valeur
distance_to_clinic_km	25	Imputation par médiane	9.5 km
waiting_time_minutes	18	Imputation par médiane	25 min
Exclusion des Cancelled
Les rendez-vous annulés ont été exclus car ils ne représentent pas un No-Show.

✅ Lignes conservées : 4 400 (88%)

❌ Lignes exclues : 600 (12%)

🔧 Feature Engineering
Nouvelles Features Créées
Feature	Type	Description	Justification
is_weekend	Binaire	1 si RDV en weekend	Les weekends ont plus de No-Show
has_previous_no_show	Binaire	1 si patient a déjà manqué	Historique = meilleur prédicteur
distance_category	Catégorielle	Very Close/Close/Medium/Far	Distance = frein
hour_category	Catégorielle	Morning/Afternoon/Evening	Horaires influencent l'assiduité
appointment_month	Numérique	Mois du RDV (1-12)	Saisonnalité
appointment_quarter	Numérique	Trimestre (1-4)	Tendances trimestrielles
patient_reliability_score	Numérique	Score de fiabilité	Patients fiables vs non fiables
Importance des Features (Week 5)
Rang	Feature	Coefficient	Impact
1	previous_no_shows	+0.85	⬆️ Très fort
2	has_previous_no_show	+0.72	⬆️ Très fort
3	booking_lead_days	+0.45	⬆️ Modéré
4	distance_to_clinic_km	+0.32	⬆️ Modéré
5	patient_reliability_score	-0.28	⬇️ Modéré
6	waiting_time_minutes	+0.21	⬆️ Faible
7	reminder_sent_encoded	-0.18	⬇️ Faible
8	age	-0.15	⬇️ Faible
9	is_weekend	+0.12	⬆️ Faible
10	appointment_month	+0.08	⬆️ Très faible
🧠 Modélisation
Approche Générale
Étape	Description	Outil
1	Chargement et exploration	Pandas
2	Préparation des données	Scikit-learn
3	Feature Engineering	Pandas, NumPy
4	Split train/test (80/20)	Scikit-learn
5	Standardisation	StandardScaler
6	Entraînement Baseline (LR)	LogisticRegression
7	Test de modèles avancés	RandomForest, GradientBoosting
8	Évaluation	Métriques sklearn
Configuration du Baseline (Week 5)
Paramètre	Valeur
Modèle	Logistic Regression
max_iter	1000
random_state	42
class_weight	balanced
penalty	l2
Split des Données
Ensemble	Taille	Attended	No-Show	Taux No-Show
Train	3 520	2 042 (58.0%)	1 478 (42.0%)	42.0%
Test	880	510 (58.0%)	370 (42.0%)	42.0%
✅ Split stratifié : les proportions sont conservées

📈 Résultats Week 5
Métriques du Baseline
Métrique	Score	Interprétation
Accuracy	62.13%	Correct dans 62.13% des cas
Precision	62.07%	62.07% des No-Show prédits sont vrais
Recall	66.80%	Détecte 66.80% des vrais No-Show
F1-Score	64.35%	Bon équilibre Precision/Recall
ROC-AUC	66.02%	Capacité modeste de discrimination
Matrice de Confusion (Week 5)
Prédit Attended	Prédit No-Show	Total
Réel Attended	265 (TN)	198 (FP)	463
Réel No-Show	161 (FN)	324 (TP)	485
Total	426	522	948
📌 Interprétation :

✅ 480 Vrais Négatifs : patients présents correctement identifiés

❌ 161 Faux Négatifs : No-Show manqués (le plus problématique)

⚠️ 198 Faux Positifs : patients présents classés à tort

✅ 324 Vrais Positifs : No-Show correctement détectés

📊 Résultats Week 6
Objectifs Week 6
✅ Analyser les faiblesses du baseline

✅ Tester 3 modèles avancés

✅ Réaliser une cross-validation 5-fold

✅ Optimiser le seuil de décision

✅ Analyser les erreurs en détail

✅ Collaborer avec Data Analytics

Réalisations Week 6
Réalisation	Détail	Impact
Test 3 modèles	Random Forest, Gradient Boosting, SMOTE+RF	Comparaison rigoureuse
Cross-validation	5-fold	Validation de la robustesse
Optimisation du seuil	0.30 optimal	+5.1 points de F1
Analyse des erreurs	FN/FP détaillés	Priorisation business
Cross-track	Data Analytics	Validation des insights
📈 Comparaison des Modèles
Tableau Comparatif Global
Modèle	Accuracy	Precision	Recall	F1	ROC-AUC
Baseline (LR)	62.13%	62.07%	66.80%	64.35%	66.02%
Random Forest	62.34%	62.65%	65.36%	63.98%	65.56%
Gradient Boosting	61.92%	62.11%	65.57%	63.79%	66.75%
SMOTE + RF	61.08%	61.84%	62.47%	62.15%	65.60%
🏆 Meilleur modèle (F1) : Baseline (Logistic Regression)

Visualisation Comparative
https://figures/week6_model_comparison.png

Différences par rapport au Baseline
Modèle	Δ Accuracy	Δ Precision	Δ Recall	Δ F1	Δ ROC-AUC
Random Forest	+0.21	+0.58	-1.44	-0.37	-0.46
Gradient Boosting	-0.21	+0.04	-1.23	-0.56	+0.73
SMOTE + RF	-1.05	-0.23	-4.33	-2.20	-0.42
Interprétation des Résultats
Pourquoi le baseline reste le meilleur ?
Le baseline est déjà bien optimisé

class_weight='balanced' gère le déséquilibre

max_iter=1000 assure la convergence

Régularisation L2 par défaut

Les modèles complexes surapprennent

Random Forest avec max_depth=10 → trop complexe

Gradient Boosting → sensible aux hyperparamètres

SMOTE → introduit du bruit dans les données

Les relations sont majoritairement linéaires

Les features sont peu corrélées entre elles

Pas d'interactions complexes détectées

La taille du dataset est limitée

4 400 échantillons → suffisant pour LR, limite pour RF/GB

Les modèles complexes ont besoin de plus de données

🔄 Cross-Validation
Résultats de la Cross-Validation 5-fold
Modèle	F1 moyen	Écart-type	Interprétation
Baseline (LR)	0.6403	0.0102	🏆 Meilleur et plus stable
Random Forest	0.6236	0.0099	Stable mais moins bon
Gradient Boosting	0.6296	0.0232	Moins stable
SMOTE + RF	0.6265	0.0122	Moins bon
📌 Interprétation :

Le Baseline est le modèle le plus robuste

Le plus faible écart-type (0.0102) indique une bonne stabilité

Le Gradient Boosting a un écart-type élevé (0.0232) → moins fiable

⚡ Optimisation du Seuil de Décision
F1-Score selon le Seuil
Seuil	F1-Score	Amélioration vs 0.50
0.50 (défaut)	0.6435	-
0.45	0.6673	+2.38
0.40	0.6799	+3.64
0.35	0.6865	+4.30
0.30 (optimal)	0.6945	+5.10
0.25	0.6915	+4.80
0.20	0.6779	+3.44
📌 Découverte Majeure :

Le seuil optimal de 0.30 améliore le F1-Score de +5.1 points

Amélioration significative sans changer l'algorithme

Recommandation : utiliser ce seuil pour le déploiement

https://figures/threshold_analysis.png

🔍 Analyse des Erreurs
Matrice de Confusion Détaillée
Prédit Attended	Prédit No-Show	Total
Réel Attended	265 (TN)	198 (FP)	463
Réel No-Show	161 (FN)	324 (TP)	485
Total	426	522	948
https://figures/week6_confusion_matrix.png

Analyse des Faux Négatifs (161 cas)
Caractéristique	FN	Moyenne	Différence
Âge moyen	34 ans	42 ans	-8 ans ⬇️
Délai de réservation	47 jours	38 jours	+9 jours ⬆️
Absences précédentes	1.8	0.9	+0.9 ⬆️
Distance moyenne	15.3 km	10.2 km	+5.1 km ⬆️
📌 Interprétation :

Les patients jeunes (34 ans) sont plus souvent manqués

Les délais longs augmentent le risque non détecté

Les patients avec historique d'absence sont sous-détectés

Analyse des Faux Positifs (198 cas)
Caractéristique	FP	Moyenne	Différence
Âge moyen	45 ans	42 ans	+3 ans ⬆️
Délai de réservation	35 jours	38 jours	-3 jours ⬇️
Absences précédentes	0.5	0.9	-0.4 ⬇️
Distance moyenne	8.7 km	10.2 km	-1.5 km ⬇️
Synthèse des Erreurs
Type	Nombre	Impact	Action Recommandée
Faux Négatifs	161	🔴 Élevé (risque médical)	Améliorer le Recall
Faux Positifs	198	🟡 Modéré (coût opérationnel)	Réduire les rappels inutiles
🤝 Cross-Track Collaboration
Partenaire : Data Analytics
Aspect	Détail
Track collaboré	Data Analytics
Dépendance projet	Les insights business informent les décisions de modélisation
Justification	Les KPIs validés guident la sélection des features
Information Reçue
KPIs validés (taux de No-Show par segment)

Insights sur les features importantes

Analyse des segments de patients à risque

Information Fournie
Feature importance du modèle

Seuils optimaux pour l'aide à la décision

Probabilités prédites pour analyse

Activité d'Intégration
Utiliser les KPIs validés pour prioriser les features

Aligner les seuils de décision avec les objectifs business

Valider les conclusions du modèle avec les insights business

Changements Résultants
Changement	Description
✅ Validation du baseline	Les insights business ont confirmé la robustesse
✅ Priorisation des erreurs	Les Faux Négatifs ont été identifiés comme priorité
✅ Interprétabilité	Les features ont été validées par le business
Preuves de la Collaboration
Fichier d'échange (Google Drive)

Capture d'écran des messages

Commit GitHub

Document de synthèse

📋 Validation et Recommandations
Critères de Validation
Critère	Objectif	Réalisé	Statut
Accuracy	> 60%	62.13%	✅
Recall	> 60%	66.80%	✅
F1-Score	> 60%	64.35%	✅
F1 (seuil 0.30)	> 65%	69.45%	✅
ROC-AUC	> 60%	66.02%	✅
Faux Négatifs	< 150	161	⚠️
Limitations du Modèle
Limitation	Description
Données fictives	Généralisation limitée à des données réelles
Pas de données temporelles	Pas de saisonnalité ou de tendances
Faux Négatifs	161 cas non détectés
Modèles complexes	N'ont pas amélioré les performances
Recommandations pour la Week 7
#	Recommandation	Priorité
1	Utiliser le seuil optimal de 0.30	🔴 Haute
2	Optimiser les hyperparamètres du baseline	🔴 Haute
3	Réduire les Faux Négatifs (161 cas)	🔴 Haute
4	Intégrer avec le pipeline ML Engineering	🟡 Moyenne
5	Valider avec les parties prenantes	🟡 Moyenne
6	Tester sur données de validation externes	🟡 Moyenne
🛠️ Technologies Utilisées
Technologie	Version	Utilisation
Python	3.8+	Langage principal
Pandas	1.5.0	Manipulation des données
NumPy	1.23.0	Calculs numériques
Scikit-learn	1.2.0	Machine Learning
Imbalanced-learn	0.10.0	SMOTE
Matplotlib	3.6.0	Visualisation
Seaborn	0.12.0	Visualisation avancée
Jupyter	-	Notebook interactif
Joblib	1.2.0	Sauvegarde des modèles
📂 Structure des Fichiers
Code
Fichier	Description
Week5_DataScience_Notebook.ipynb	Notebook Week 5 (baseline)
Week6_DataScience_Notebook.ipynb	Notebook Week 6 (amélioration)
healthconnect_utils.py	Fonctions utilitaires
Données
Fichier	Description	Taille
HealthConnect_Appointment_Data.csv	Données brutes	5 000 lignes
HealthConnect_Prepared_Data.csv	Données préparées	4 400 lignes
Figures
Fichier	Description
week6_model_comparison.png	Comparaison des métriques
week6_confusion_matrix.png	Matrice de confusion
week6_feature_importance.png	Importance des features
week6_roc_comparison.png	Courbes ROC comparatives
Outputs
Fichier	Description
week6_best_model.pkl	Meilleur modèle (Baseline LR)
week6_scaler.pkl	Scaler sauvegardé
Documentation
Fichier	Description
Week6_DataScience_Report.pdf	Rapport complet
Week6_DataScience_Report.tex	Source LaTeX
👥 Équipe et Contact
Rôle	Nom	Contact
Data Scientist	SOGA Para	sparadodaniel@gmail.com
Programme	AnalystLab Africa	analystlab.africa
Projet	HealthConnect Experience Lab	-
Semaine	Week 5 & Week 6	-
📝 Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

txt
MIT License

Copyright (c) 2026 AnalystLab Africa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
📌 Tags
#DataScience #MachineLearning #Healthcare #AnalystLabAfrica
#HealthConnect #Python #Classification #LogisticRegression
#RandomForest #GradientBoosting #SMOTE #CrossValidation
#ModelEvaluation #AI #PredictiveModeling #NoShowPrediction

🙏 Remerciements
AnalystLab Africa pour l'opportunité et l'accompagnement

HealthConnect Clinic pour les données

L'équipe Data Analytics pour la collaboration fructueuse

Les mentors pour leurs conseils et retours

🔗 Liens Utiles
Notebook Week 5

Notebook Week 6

Rapport Week 6 (PDF)

Dataset

Fait avec ❤️ dans le cadre du programme AnalystLab Africa Experience Lab
