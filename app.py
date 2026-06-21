# Étape 6 — Votre application Streamlit complète
import re, glob
import streamlit as st
import plotly.express as px
import asyncio, os, sys, json, re
import pandas as pd, plotly.graph_objects as go
from plotly.subplots import make_subplots
import nest_asyncio; nest_asyncio.apply()
import webbrowser

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import tool




#-----Redéfinition des fonctions-----
def fuzzy_col(df, col_name: str) -> str:
    """Trouve la vraie colonne dans df à partir d'un nom approximatif."""
    col_clean = re.sub(r'[^a-zA-Z0-9 ]', '', col_name).strip().lower()
    for c in df.columns:
        if c.lower() == col_clean:
            return c
    for c in df.columns:
        if col_clean in c.lower() or c.lower() in col_clean:
            return c
    words = [w for w in col_clean.split() if len(w) >= 4]
    for word in words:
        stems = [word,
                 word[:-1] if len(word) > 4 else '',
                 word[:-2] if len(word) > 5 else '']
        for c in df.columns:
            if any(s and s in c.lower() for s in stems):
                return c
    return col_name

def _col_not_found_msg(df, col_name, kind='catégorielle'):
    if kind == 'catégorielle':
        available = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    else:
        available = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return (f"Colonne '{col_name}' introuvable. "
            f"Utilise un de ces noms exacts : {available}")

def _clean_filepath(filepath: str) -> str:
    """Sanitise le filepath : JSON array, faux chemin absolu, typo de nom."""
    fp = filepath.strip()
    # Cas 1 : tableau JSON  ["file.csv"]
    if fp.startswith("["):
        fp = fp.strip("[]").split(",")[0].strip()
        fp = fp.strip('"').strip("'")
    # Cas 2 : faux chemin absolu inventé par le LLM (/path/to/file.csv)
    if not os.path.exists(fp) and "/" in fp:
        basename = fp.split("/")[-1]
        if os.path.exists(basename):
            fp = basename
    # Cas 3 : typo dans le nom (job_posting.csv → job_postings.csv)
    if not os.path.exists(fp):
        name = os.path.basename(fp)
        candidates = glob.glob("*.csv")
        # prefer exact prefix match
        for c in candidates:
            if c.startswith(name.split(".")[0][:8]):
                fp = c
                break
        else:
            if len(candidates) == 1:
                fp = candidates[0]
    return fp




    
#-----Redéfinition des tools-----
#-----Tools Basics------
@tool
def describe_dataset(filepath: str = "ai_student_impact_dataset.csv") -> str:
    """Charge un CSV et retourne ses dimensions et colonnes. Utiliser EN PREMIER.
    Args:
        filepath: Chemin vers le fichier CSV à analyser.
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable. Utilise 'ai_student_impact_dataset.csv'."
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return (f"{df.shape[0]} lignes, {df.shape[1]} colonnes.\n"
            f"Catégorielles (utilise bar_chart, noms exacts) : {cat_cols}\n"
            f"Numériques (utilise numeric_stats, noms exacts) : {num_cols}")

@tool
def bar_chart(filepath: str = "ai_student_impact_dataset.csv", column: str = "", title: str = '') -> str:
    """Génère un graphique à barres horizontal des 10 valeurs les plus fréquentes d'une colonne.
    Args:
        filepath: Chemin vers le fichier CSV.
        column: Nom exact ou partiel de la colonne catégorielle.
        title: Titre du graphique (optionnel).
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable."
        
    col = fuzzy_col(df, column)
    if col not in df.columns:
        return _col_not_found_msg(df, column, 'catégorielle')
        
    top = df[col].value_counts().head(10).reset_index()
    top.columns = [col, 'count']
    
    # Création du graphique Plotly
    fig = px.bar(top.sort_values('count'), x='count', y=col, orientation='h', title=title or f"Top 10 — {col}")
    
    # Stockage de la figure dans le session_state pour que Streamlit puisse l'afficher
    st.session_state["last_generated_chart"] = fig
    
    return f"✅ Le graphique pour la colonne '{col}' a été généré avec succès et est prêt à être affiché à l'écran."

@tool
def scatter_correlation(filepath: str, x_col: str, y_col: str, color_col: str = None,title: str = '') -> str:
    """Génère un graphique de corrélation (nuage de points) entre deux colonnes numériques. Sauvegarde en HTML.
    Args:
        filepath: Chemin vers le fichier CSV.
        x_col: Nom de la colonne numérique pour l'axe X.
        y_col: Nom de la colonne numérique pour l'axe Y.
        color_col: Colonne optionnelle pour colorer les points (catégorielle).
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable."

    x = fuzzy_col(df, x_col)
    y = fuzzy_col(df, y_col)
    c = fuzzy_col(df, color_col) if color_col else None
    
    if x not in df.columns or y not in df.columns:
        return f"Colonnes introuvables pour le scatter plot. X: '{x}', Y: '{y}'."

    fig = px.scatter(df, x=x, y=y, color=c, title=f"Corrélation : {x} vs {y}")
    
    # Sécuriser le nom du fichier HTML de sortie
    filename_x = x.replace(" ", "_").replace("/", "_")
    filename_y = y.replace(" ", "_").replace("/", "_")
    out = f'scatter_{filename_x}_vs_{filename_y}.html'
    
    fig.write_html(out)
    webbrowser.open(f'file://{os.path.abspath(out)}')
    return f"Graphique de corrélation sauvegardé et ouvert : {out}"


@tool
def pie_distribution(filepath: str, column: str, title: str ='') -> str:
    """Génère un graphique en camembert (Pie chart) pour analyser la répartition (top 5). Sauvegarde en HTML.
    Args:
        filepath: Chemin vers le fichier CSV.
        column: Nom de la colonne catégorielle à analyser.
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable."
        
    col = fuzzy_col(df, column)
    if col not in df.columns:
        return _col_not_found_msg(df, column, 'catégorielle')
    
    top = df[col].value_counts().head(5).reset_index()
    top.columns = [col, 'count']
    
    fig = px.pie(top, values='count', names=col, title=f"Répartition (Top 5) de {col}")
    
    # Sécuriser le nom du fichier HTML de sortie
    out = f'pie_{col.replace(" ", "_").replace("/", "_")}.html'
    
    fig.write_html(out)
    webbrowser.open(f'file://{os.path.abspath(out)}')
    return f"Graphique en camembert sauvegardé et ouvert : {out}"


@tool
def top_values(filepath: str = "ai_student_impact_dataset.csv", column: str = "") -> str:
    """Retourne les 10 valeurs (s'il y en a 10) les plus fréquentes d'une colonne catégorielle (texte).
    Args:
        filepath: Chemin vers le fichier CSV.
        column: Nom exact ou partiel de la colonne catégorielle.
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable. Utilise 'ai_student_impact_dataset.csv'."
    col = fuzzy_col(df, column)
    if col not in df.columns:
        return _col_not_found_msg(df, column, 'catégorielle')
    result = df[col].value_counts()
    n_items = len(result)
    if n_items>10:
        result=result.head(10)
        titre = f"Top 10 — {col} :\n"
    else:
        titre = f"Top {n_items} — {col} :\n"
    return titre + result.to_string()



@tool
def numeric_stats(filepath: str, column: str) -> str:
    """TODO : Retourne min, max, moyenne, médiane d'une colonne numérique
    Args:
        filepath: Chemin vers le fichier CSV.
        column: Nom exact ou partiel de la colonne numérique.
    """
    try:
        df = pd.read_csv(_clean_filepath(filepath))
    except FileNotFoundError:
        return f"Fichier '{filepath}' introuvable. Utilise 'ai_student_impact_dataset.csv'."
    col = fuzzy_col(df, column)
    if col not in df.columns:
        return _col_not_found_msg(df, column, 'numérique')
    serie = df[col].dropna()

    return (
    f"Statistiques — {col} :\n"
    f"Moyenne : {serie.mean():.2f}\n"
    f"Médiane : {serie.median():.2f}\n"
    f"Minimum : {serie.min():.2f}\n"
    f"Maximum : {serie.max():.2f}"
)
    pass


@tool
def make_dashboard(filepath: str, column1: str, column2: str, title: str) -> str:
    """Génère un dashboard HTML avec 2 bar charts côte à côte. Sauvegarde et ouvre dans le navigateur.
    IMPORTANT : utilise uniquement des noms de colonnes retournés par describe_dataset.
    Args:
        filepath: Chemin vers le fichier CSV.
        column1: Première colonne catégorielle (nom exact ou partiel).
        column2: Deuxième colonne catégorielle (nom exact ou partiel).
        title: Titre du dashboard.
    """
    df = pd.read_csv(_clean_filepath(filepath))

    # Normaliser et vérifier les deux colonnes
    col1 = fuzzy_col(df, column1)
    col2 = fuzzy_col(df, column2)

    # Si une colonne est introuvable → retourner la liste des colonnes disponibles
    # (le LLM lira ce message et relancera avec les bons noms)
    if col1 not in df.columns or col2 not in df.columns:
        cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        missing = [c for c, real in [(column1, col1), (column2, col2)] if real not in df.columns]
        return (f"Colonnes introuvables : {missing}. "
                f"Utilise ces noms exacts : {cat_cols}")

    # Préparer les données : top 10 pour chaque colonne
    def top10(col):
        t = df[col].value_counts().head(10).reset_index()
        t.columns = [col, 'count']
        return t.sort_values('count')

    # Créer la figure avec 2 sous-graphiques côte à côte
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[col1, col2],
                        horizontal_spacing=0.15)
    t1 = top10(col1)
    fig.add_trace(go.Bar(x=t1['count'], y=t1[col1], orientation='h', name=col1), row=1, col=1)
    t2 = top10(col2)
    fig.add_trace(go.Bar(x=t2['count'], y=t2[col2], orientation='h', name=col2), row=1, col=2)
    fig.update_layout(title_text=title, height=500, showlegend=False)

    out = 'dashboard.html'
    fig.write_html(out)
    webbrowser.open(f'file://{os.path.abspath(out)}')
    return f"Dashboard sauvegardé : {out} ({col1} + {col2})"

#------Tools Impact IA-----
@tool
def analyse_impact_ia_par_groupe(filepath: str,column: str = 'Primary_Use_Case') -> str:
    """
    Analyse l'impact de l'IA en regroupant par une colonne catégorielle spécifique.
    Colonnes valides recommandées : 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Major_Category'
    """
    colonnes_valides = ['Primary_Use_Case', 'Prompt_Engineering_Skill', 'Major_Category', 'Burnout_Risk_Level']
    if column not in colonnes_valides:
        return f"Erreur : La colonne de segmentation doit être l'une des suivantes : {colonnes_valides}"
    
    df_ia = pd.read_csv(_clean_filepath(filepath))
        
    # Groupement et calcul des moyennes clés
    analyse = df_ia.groupby(column).agg({
        'Pre_Semester_GPA': 'mean',
        'Post_Semester_GPA': 'mean',
        'Weekly_GenAI_Hours': 'mean',
        'Skill_Retention_Score': 'mean',
        'Anxiety_Level_During_Exams': 'mean'
    }).round(2).reset_index()
    
    # Ajout d'une colonne calculée pour voir l'évolution globale du GPA par groupe
    analyse["Evolution_GPA"] = (analyse["Post_Semester_GPA"] - analyse["Pre_Semester_GPA"]).round(2)
    
    rapport = f"### 🤖 ANALYSE DE L'IA SEGMENTÉE PAR : {column}\n\n"
    rapport += analyse.to_markdown(index=False)
    
    return rapport

@tool
def detecter_dependance_ia(filepath: str,seuil_heures: int = 15) -> str:
    """
    Filtre les étudiants ayant une utilisation intensive de l'IA (défini par le seuil d'heures)
    et compare leur profil à la moyenne globale de tous les étudiants.
    """
    df_ia = pd.read_csv(_clean_filepath(filepath))

    # Groupe "Intensif" vs Groupe "Global"
    df_intensif = df_ia[df_ia['Weekly_GenAI_Hours'] >= seuil_heures]
    
    metrics = [
        'Pre_Semester_GPA', 'Post_Semester_GPA', 'Weekly_GenAI_Hours', 
        'Traditional_Study_Hours', 'Anxiety_Level_During_Exams', 'Skill_Retention_Score'
    ]
    
    moyenne_intensif = df_intensif[metrics].mean().to_frame(name=f'Utilisateurs Intensifs (>= {seuil_heures}h)')
    moyenne_globale = df_ia[metrics].mean().to_frame(name='Moyenne Globale')
    
    # Fusion des deux analyses pour comparaison directe
    comparatif = moyenne_globale.join(moyenne_intensif).round(2).reset_index()
    comparatif.rename(columns={'index': 'Indicateurs'}, inplace=True)
    
    rapport = f"### ⚠️ PROFIL COMPARAISON : FORTE DÉPENDANCE À L'IA (Effectif : {len(df_intensif)} étudiants)\n\n"
    rapport += comparatif.to_markdown(index=False)
    
    return rapport


@tool
def impact_temps_utilisation(filepath: str) -> str:
    """
    Analyse l'impact du temps d'utilisation hebdomadaire de l'IA sur le succès académique 
    (GPA, rétention des compétences) et le bien-être (anxiété, burnout) des étudiants.
    """
    df_ia = pd.read_csv(_clean_filepath(filepath))
    df = df_ia.copy()
    
    # 1. Définition des tranches d'utilisation basées sur 'Weekly_GenAI_Hours'
    # Moins de 5h = Faible | Entre 5h et 15h = Modéré | Plus de 15h = Intensif
    def segmenter_heures(heures):
        if heures < 5:
            return "1. Faible (< 5h/semaine)"
        elif heures <= 15:
            return "2. Modéré (5h - 15h/semaine)"
        else:
            return "3. Intensif (> 15h/semaine)"
            
    df['Tranche_Utilisation'] = df['Weekly_GenAI_Hours'].apply(segmenter_heures)
    
    # 2. Agrégation des statistiques clés par tranche d'utilisation
    analyse_impact = df.groupby('Tranche_Utilisation').agg({
        'Student_ID': 'count',                 # Nombre d'étudiants dans le groupe
        'Pre_Semester_GPA': 'mean',
        'Post_Semester_GPA': 'mean',
        'Traditional_Study_Hours': 'mean',     # Permet de voir s'ils étudient moins "traditionnellement"
        'Skill_Retention_Score': 'mean',       # Impact sur l'apprentissage profond
        'Anxiety_Level_During_Exams': 'mean'   # Impact psychologique
    }).round(2).reset_index()
    
    # Renommer la colonne count pour plus de clarté
    analyse_impact.rename(columns={'Student_ID': 'Effectif_Etudiants'}, inplace=True)
    
    # 3. Calcul de l'évolution du GPA pour chaque groupe
    analyse_impact["Evolution_GPA"] = (analyse_impact["Post_Semester_GPA"] - analyse_impact["Pre_Semester_GPA"]).round(2)
    
    # Réorganisation des colonnes pour une lecture logique par l'agent
    colonnes_ordre = [
        'Tranche_Utilisation', 'Effectif_Etudiants', 'Traditional_Study_Hours',
        'Pre_Semester_GPA', 'Post_Semester_GPA', 'Evolution_GPA',
        'Skill_Retention_Score', 'Anxiety_Level_During_Exams'
    ]
    
    # Construction du rapport Markdown
    rapport = "### ⏱️ ANALYSE DE L'IMPACT DU TEMPS D'UTILISATION DE L'IA\n\n"
    rapport += "Ce tableau compare le profil des étudiants selon leur volume horaire hebdomadaire sur les outils d'IA :\n\n"
    rapport += analyse_impact[colonnes_ordre].to_markdown(index=False)
    
    return rapport


    
#------Tools Performances-----
@tool
def top_progression(filepath : str,top_n: int = 15) -> str:
    """
    Calcule la progression des étudiants (Exam_Score - Previous_Scores),
    génère le top N des meilleures progressions et fournit la moyenne de leur profil.
    """

    df_perf = pd.read_csv(_clean_filepath(filepath))
    df = df_perf.copy()
    
    # Calcul de la variable de progression
    df['Progression'] = df['Exam_Score'] - df['Previous_Scores']
    
    # Extraction du Top N
    top_students = df.sort_values(by='Progression', ascending=False).head(top_n)
    
    # Sélection des colonnes numériques clés pour l'analyse
    cols_numeriques = ['Progression', 'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Tutoring_Sessions']
    cols_cat = ['Motivation_Level', 'Teacher_Quality', 'Parental_Involvement']
    
    # 1. Calcul des moyennes du groupe d'élite
    moyennes = top_students[cols_numeriques].mean().to_frame().T
    moyennes.index = ['MOYENNE DU TOP']
    
    # 2. Préparation du tableau des étudiants
    affichage_students = top_students[cols_numeriques + cols_cat].copy()
    
    # Construction du rendu textuel final pour l'agent
    rapport = f"### 📈 TOP {top_n} DES MEILLEURES PROGRESSIONS APPRÉCIABLES\n\n"
    rapport += affichage_students.to_markdown(index=False)
    rapport += "\n\n### 📊 PROFIL STATISTIQUE MOYEN DE CE GROUPE\n\n"
    rapport += moyennes.to_markdown(index=False)
    
    return rapport

@tool
def analyse_par_type_ecole(filepath : str) -> str:
    """
    Compare les performances et l'accès aux ressources entre les écoles publiques et privées.
    """

    df_perf = pd.read_csv(_clean_filepath(filepath))

    # Agrégation des notes et variables critiques par School_Type
    stats_ecole = df_perf.groupby('School_Type').agg({
        'Exam_Score': 'mean',
        'Attendance': 'mean',
        'Hours_Studied': 'mean',
        'Previous_Scores': 'mean'
    }).round(2).reset_index()
    
    # Distribution des ressources en % (ex: Internet_Access)
    distribution_internet = pd.crosstab(df_perf['School_Type'], df_perf['Internet_Access'], normalize='index') * 100
    distribution_internet = distribution_internet.round(1).reset_index()
    
    rapport = "### 🏢 COMPARAISON PUBLIC VS PRIVÉ (MOYENNES)\n\n"
    rapport += stats_ecole.to_markdown(index=False)
    rapport += "\n\n### 🌐 ACCÈS À INTERNET PAR TYPE D'ÉCOLE (EN %)\n\n"
    rapport += distribution_internet.to_markdown(index=False)
    
    return rapport

@tool
def facteurs_echec_reussite(filepath : str) -> str:
    """
    Isole les 10% des meilleurs étudiants et les 10% des étudiants en difficulté 
    sur leur Exam_Score, puis compare leurs profils (habitudes d'étude, sommeil, environnement).
    """
    df_perf = pd.read_csv(_clean_filepath(filepath))
    df = df_perf.copy()
    
    # 1. Détermination des seuils pour les top 10% et bottom 10%
    seuil_haut = df['Exam_Score'].quantile(0.90)
    seuil_bas = df['Exam_Score'].quantile(0.10)
    
    # 2. Création des deux sous-groupes
    df_top = df[df['Exam_Score'] >= seuil_haut]
    df_bottom = df[df['Exam_Score'] <= seuil_bas]
    
    # 3. Métriques numériques clés à comparer
    metrics_num = [
        'Exam_Score', 'Hours_Studied', 'Attendance', 
        'Sleep_Hours', 'Tutoring_Sessions', 'Physical_Activity'
    ]
    
    moyennes_top = df_top[metrics_num].mean().to_frame(name='Top 10% (Réussite)')
    moyennes_bottom = df_bottom[metrics_num].mean().to_frame(name='Bottom 10% (Difficulté)')
    
    # Fusion des analyses numériques
    comparatif_num = moyennes_top.join(moyennes_bottom).round(2).reset_index()
    comparatif_num.rename(columns={'index': 'Indicateurs Numériques'}, inplace=True)
    
    # 4. Métriques catégorielles clés (Facteurs environnementaux)
    # On va calculer le taux d'accès ou de forte implication pour donner du contexte social
    facteurs_env = []
    
    # % Accès Internet Élevé
    facteurs_env.append({
        'Facteur Environnemental': 'Accès Internet (Oui %)',
        'Top 10% (Réussite)': f"{(df_top['Internet_Access'] == 'Yes').mean() * 100:.1f}%",
        'Bottom 10% (Difficulté)': f"{(df_bottom['Internet_Access'] == 'Yes').mean() * 100:.1f}%"
    })
    
    # % Implication Parentale Forte
    facteurs_env.append({
        'Facteur Environnemental': 'Implication Parentale (High %)',
        'Top 10% (Réussite)': f"{(df_top['Parental_Involvement'] == 'High').mean() * 100:.1f}%",
        'Bottom 10% (Difficulté)': f"{(df_bottom['Parental_Involvement'] == 'High').mean() * 100:.1f}%"
    })
    
    # % Motivation Élevée
    facteurs_env.append({
        'Facteur Environnemental': 'Motivation de l\'élève (High %)',
        'Top 10% (Réussite)': f"{(df_top['Motivation_Level'] == 'High').mean() * 100:.1f}%",
        'Bottom 10% (Difficulté)': f"{(df_bottom['Motivation_Level'] == 'High').mean() * 100:.1f}%"
    })
    
    # % Qualité Enseignement Excellente
    facteurs_env.append({
        'Facteur Environnemental': 'Qualité Enseignant (High %)',
        'Top 10% (Réussite)': f"{(df_top['Teacher_Quality'] == 'High').mean() * 100:.1f}%",
        'Bottom 10% (Difficulté)': f"{(df_bottom['Teacher_Quality'] == 'High').mean() * 100:.1f}%"
    })
    
    comparatif_cat = pd.DataFrame(facteurs_env)
    
    # 5. Construction du rapport final
    rapport = f"### 🔑 ANALYSE DES FACTEURS DE RÉUSSITE VS ÉCHEC\n"
    rapport += f"Échantillon basé sur les extrêmes du jeu de données (Seuil Réussite >= {seuil_haut} pts | Seuil Difficulté <= {seuil_bas} pts).\n\n"
    
    rapport += "#### 📊 COMPARAISON DES HABITUDES (MOYENNES)\n\n"
    rapport += comparatif_num.to_markdown(index=False)
    
    rapport += "\n\n#### 🏡 COMPARAISON DE L'ENVIRONNEMENT ET CADRE SOCIAL\n\n"
    rapport += comparatif_cat.to_markdown(index=False)
    
    return rapport




st.set_page_config(page_title="Agent Student", page_icon="🏙️", layout="wide")
st.title("🏙️ Agent Student")

llm = ChatOpenAI(model="llama3.2", base_url="http://localhost:11434/v1", api_key="ollama", temperature=0)

basic_tools = [describe_dataset, bar_chart, top_values, numeric_stats,make_dashboard,pie_distribution,scatter_correlation]
ia_impact_tools = [analyse_impact_ia_par_groupe,detecter_dependance_ia,impact_temps_utilisation]
perf_tools = [top_progression,analyse_par_type_ecole,facteurs_echec_reussite]

local_tools = basic_tools + ia_impact_tools + perf_tools
local_tools_dict = {tool.name: tool for tool in local_tools}

# Sauvegarder l'instance de mémoire dans la session de Streamlit pour éviter qu'elle soit recréée
if "agent_memory" not in st.session_state:
    st.session_state["agent_memory"] = MemorySaver()

# On utilise cette instance persistante
memory = st.session_state["agent_memory"]


def build_home_dashboard() -> go.Figure:
    """Génère un dashboard d'accueil 2x2 combinant les trois thématiques du projet."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Répartition des Étudiants par Filière (Major)", 
            "Top Utilisation Principale de l'IA", 
            "Impact des Heures d'Étude sur la Note d'Examen", 
            "Santé mentale en fonction de l'utilisation des réseaux sociaux"
        ),
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "box"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.10
    )

    # ---- FIG 1 : Pie Chart - Répartition par Major Category (Dataset IA) ----
    try:
        df_ia = pd.read_csv("ai_student_impact_dataset.csv")
        df_major = df_ia['Major_Category'].value_counts().reset_index()
        df_major.columns = ['Major_Category', 'count']
        fig_pie = px.pie(df_major, values='count', names='Major_Category', color_discrete_sequence=px.colors.qualitative.Pastel)
        for trace in fig_pie.data:
            fig.add_trace(trace, row=1, col=1)
    except Exception:
        pass

    # ---- FIG 2 : Bar Chart - Primary Use Case de l'IA (Dataset IA) ----
    try:
        df_use = df_ia['Primary_Use_Case'].value_counts().head(5).reset_index()
        df_use.columns = ['Primary_Use_Case', 'count']
        fig_bar = px.bar(df_use, x='count', y='Primary_Use_Case', orientation='h', color='count', color_continuous_scale='Blues')
        for trace in fig_bar.data:
            fig.add_trace(trace, row=1, col=2)
    except Exception:
        pass

    # ---- FIG 3 : Scatter Plot - Heures d'étude vs Exam Score (Dataset Performance) ----
    try:
        df_perf = pd.read_csv("StudentPerformanceFactors.csv")
        # Échantillon de 500 points max pour éviter de surcharger le graphique
        df_perf_sample = df_perf.sample(n=min(500, len(df_perf)), random_state=42)
        fig_scat = px.scatter(df_perf_sample, x='Hours_Studied', y='Exam_Score', opacity=0.6, color='Motivation_Level')
        for trace in fig_scat.data:
            fig.add_trace(trace, row=2, col=1)
    except Exception:
        pass

   # ---- FIG 4 : Scatter Plot - Score de Santé Mentale vs Temps d'utilisation quotidien ----
    try:
        df_mental = pd.read_csv("Student Social Media And Mental Health Impact.csv")
        df_mental_sample = df_mental.sample(n=min(500, len(df_mental)), random_state=42)
        
        fig_scat = px.scatter(df_mental_sample, x='Avg_Daily_Usage_Hours', y='Mental_Health_Score', opacity=0.6, color='Sleep_Hours_Per_Night')
        for trace in fig_scat.data:
            fig.add_trace(trace, row=2, col=2)
    except Exception:
        pass

    # ---- Ligne 2, Col 1 : Heures d'étude vs Exam Score ----
    fig.update_xaxes(title_text="Heures étudiées", row=2, col=1)
    fig.update_yaxes(title_text="Note à l'examen", row=2, col=1)

    # ---- Ligne 2, Col 2 : Santé Mentale vs Réseaux Sociaux ----
    fig.update_xaxes(title_text="Utilisation quotidienne (Heures)", row=2, col=2)
    fig.update_yaxes(title_text="Score de santé mentale", row=2, col=2)

    # Ajustements globaux du layout
    fig.update_layout(
        height=800,
        showlegend=False,
        template="plotly_white",
        title_text="📊 Tableau de Bord Global des Indicateurs Étudiants",
        title_font_size=20,
        title_x=0.5,
        margin=dict(l=30, r=30, t=80, b=40)
    )
    
    return fig

dashboard_fig = build_home_dashboard()
st.plotly_chart(dashboard_fig, use_container_width=True)






st.markdown("---")
async def run_combined_agent(question : str):
    # 2. Connexion au serveur MCP (Santé Mentale)
    # Remplace par le nom exact de ton fichier serveur validé à l'étape précédente
    mcp_client = MultiServerMCPClient({
        'mental_health': {
            'command': sys.executable,
            'args': [os.path.abspath('mcp_social_health_server.py')],
            'transport': 'stdio',
        }
    })

    

    config = {"configurable": {"thread_id": "thread_student"}}
    
    print("🔄 Connexion au serveur MCP et récupération des outils...")
    # Récupération des outils distants (contenant describe_mental_health, query_data, etc.)
    mcp_tools = await mcp_client.get_tools()
    
    # 3. Fusion de la boîte à outils
    # L'agent verra TOUS ces outils d'un coup et choisira le bon
    all_tools = local_tools + mcp_tools
    print(f"✅ Outils configurés : {len(local_tools)} locaux + {len(mcp_tools)} via MCP.")

    # 5. Définition des consignes du système (System Prompt)
    SYSTEM_PROMPT = """
        Tu es un expert en data science spécialisé dans l'analyse de la vie étudiante. 
        Tu as accès à 3 datasets interconnectés via tes outils :
        1. L'impact de l'IA sur les étudiants : 'ai_student_impact_dataset.csv'(outils locaux ia_impact_tools et basic_tools)
        2. Les performances scolaires: 'StudentPerformanceFactors.csv' (outils locaux perf_tools et basic_tools)
        3. La santé mentale et les réseaux sociaux (via le serveur MCP mental_health)
        Règles cruciales :
        - Utilise TOUJOURS le format Markdown pour tes réponses.
        - Analyse les données objectivement en croisant les résultats si l'utilisateur te le demande.
        - Si l'outil MCP 'query_data' est requis pour une question complexe sur la santé mentale, 
        génère la requête SQL appropriée.
        - Quand on te demande de comparer des données ou des chiffres issus d'un outil précédent, retourne les valeurs exactes et ne fais pas de généralisations
    """

    # 6. Création de l'agent ReAct (Raisonnement + Action)
    print(f"🤖 Initialisation de l'agent avec la question : '{question}'\n")
    agent = create_react_agent(llm, tools=all_tools, prompt=SYSTEM_PROMPT,checkpointer=memory)

    


    steps = []
    
    # 1. Une SEULE exécution complète et stable
    result = await agent.ainvoke({"messages": [("user", question)]},config)
    
    # 2. On extrait les outils utilisés pour alimenter l'expander Streamlit
    messages = result.get("messages", [])
    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                steps.append(f"⚙️ **Appel de l'outil :** `{tool_call['name']}` avec `{tool_call['args']}`")
        elif msg.type == "tool":
            steps.append(f"👁️ **Résultat obtenu** de l'outil `{msg.name}`.")
            
    # 3. Récupération du message final écrit par le LLM
    answer = messages[-1].content
    return answer, steps
st.subheader("🤖 Posez vos questions à l'Assistant IA")
st.caption("Exemple : 'Quel est l'impact du temps d'utilisation de l'IA sur l'évolution du GPA des étudiants ?'")

# Initialisation de la mémoire du chat Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des messages historiques
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture d'une nouvelle entrée utilisateur
if user_query := st.chat_input("Votre message..."):
    # Affichage du message utilisateur
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Génération de la réponse de l'IA
    with st.chat_message("assistant"):
        with st.spinner("L'agent réfléchit et consulte le serveur MCP..."):
            try:
                # Exécution de la boucle asynchrone requise par le client MCP
                loop = asyncio.get_event_loop()
                answer, steps = loop.run_until_complete(run_combined_agent(user_query))
                
                # Affichage des étapes de raisonnement dans un menu accordéon déroulant
                if steps:
                    with st.expander("🔍 Voir le cheminement et les outils utilisés"):
                        for step in steps:
                            st.write(step)

                

                
                # Affichage de la réponse finale
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                error_msg = f"❌ Une erreur est survenue lors de l'appel de l'agent : {str(e)}"
                st.error(error_msg)
    

