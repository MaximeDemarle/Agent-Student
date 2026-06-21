# 🏙️ Agent Student — Assistant IA Multi-Datasets (MCP & LangGraph)

Ce projet est une application décisionnelle et analytique basée sur un architecture d'Agent IA (ReAct). Il permet d'explorer, de croiser et d'analyser la vie étudiante à travers trois thématiques majeures : l'impact de l'IA, les performances scolaires, et la santé mentale liée aux réseaux sociaux.

## 📊 Architecture du Projet

L'application est propulsée par **Streamlit** pour l'interface et **LangGraph** pour la gestion des cycles de l'agent. Elle intègre :
- **Outils Locaux :** Analyses statistiques et corrélations graphiques (Plotly) sur les datasets IA et Performances.
- **Serveur MCP (Model Context Protocol) :** Un serveur autonome connecté à une base SQLite/CSV pour l'analyse isolée de la santé mentale.
- **Mémoire Persistante :** Utilisation de `MemorySaver` pour conserver le contexte d'une question à l'autre.



## 🚀 Installation et Lancement

### 1. Prérequis
- Python 3.10+
- Ollama installé avec le modèle `llama3.2` fonctionnel en local.

### 2. Installation des dépendances
```bash
pip install -r requirements.txt
```


## Description du projet 

Pour ce projet utilise 3 datasets:
- **Impact de l'IA :** 'ai_student_impact_dataset.csv' décrivant différents aspects de l'tuilisation et de l'impact de l'IA sur les étudiants.
- **Performences des étudiants :** 'StudentPerformanceFactors.csv' donnant différents facteurs pouvant impacter les performances des étudiants.
- **Réseaux sociaux et santé mentale :** 'Student Social Media And Mental Health Impact.csv' qui a été convertit en serveur MCP et permet de voir l'influence des réseaux sociaux sur la santé mentale des étudiants.

Le projet se décompose en différentes étapes :

### Partie prérequis:

Ici c'est la partie où l'on fait les différents imports et installations afin que le projet puisse fonctionner.

### Partie définition des outils et du serveur MCP

On définit les différents outils de base et liés aux différents datasets ainsi que le serveur MCP.

### Création du premier agent

On créer un premier agent ayant accès à tous les outils et datasets et on le test.

### Création de l'interface utilisateur

On créer une interface utilisateur permettant de poser des questions à l'agent disposant d'un système de mémoire sur les différents datasets.


## Conclusion sur le projet

On peut remarquer que toutes les fonctions voulus sont fonctionnelles même s'il  arrive que l'agent s'emmêle parfois les pinceaux lorsqu'il tente d'appeler des outils pour les mauvais dataset.
