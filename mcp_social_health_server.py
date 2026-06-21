
from mcp.server.fastmcp import FastMCP
import duckdb, os

mcp = FastMCP('SocialHealthServer')
CSV_PATH = os.path.join(os.path.dirname(__file__), 'Student Social Media And Mental Health Impact.csv')

def get_conn():
    # TODO : créer connexion DuckDB + Student Social Media And Mental Health Impact.csv
    conn = duckdb.connect()
    conn.execute(f"CREATE TABLE IF NOT EXISTS mental_health AS SELECT * FROM read_csv_auto('{CSV_PATH}')")
    return conn

@mcp.tool()
def describe_mental_health() -> str:
    """Retourne une vue générale du dataset sur la santé mentale."""
    # TODO : utiliser get_conn() et retourner un résumé sous forme de string
    conn=get_conn()
    try:
        summary = conn.execute("SUMMARIZE mental_health").df().to_string()
        total_rows = conn.execute("SELECT COUNT(*) FROM mental_health").fetchone()[0]
        return f"Nombre total d'étudiants : {total_rows}\n\nRésumé des colonnes :\n{summary}"
    except Exception as e:
        return f"Erreur lors de la description : {str(e)}"
    finally:
        conn.close()

@mcp.tool()
def query_data(sql: str) -> str:
    """Execute une requête SQL sur le dataset. Table disponible : mental_health.
    Colonnes: Gender, Country, Academic_Level, Most_Used_Platform, Purpose_Of_Use, Stress_Level,
    Age, Avg_Daily_Usage_Hours, Daily_Unlocks, Study_Hours, Physical_Activity_Hours, Sleep_Hours_Per_Night, Mental_Health_Score."""
    conn = get_conn()
    try:
        # Exécution sécurisée/directe de la requête SQL fournie
        result = conn.execute(sql).df().to_string()
        return result
    except Exception as e:
        return f"Erreur lors de l'exécution de la requête : {str(e)}"
    finally:
        conn.close()


@mcp.tool()
def stats_par_segment(colonne_segment: str = 'Country') -> str:
    """
    Filtre et renvoie les moyennes des indicateurs selon un segment précis.
    Valeurs recommandées pour colonne_segment : 'Country', 'Stress_Level', 'Gender', 'Most_Used_Platform'
    """
    if colonne_segment not in ['Country', 'Stress_Level', 'Gender', 'Most_Used_Platform', 'Academic_Level']:
        return "Erreur : Colonne de segmentation non valide."
        
    conn = get_conn()
    try:
        query = f"""
            SELECT {colonne_segment}, 
                   COUNT(*) as Effectif,
                   ROUND(AVG(Age), 1) as Age_Moyen,
                   ROUND(AVG(Avg_Daily_Usage_Hours), 2) as RS_Heures_Moy,
                   ROUND(AVG(Sleep_Hours_Per_Night), 2) as Sommeil_Moy,
                   ROUND(AVG(Mental_Health_Score), 2) as Sante_Mentale_Moy
            FROM mental_health
            GROUP BY {colonne_segment}
            ORDER BY Sante_Mentale_Moy DESC
        """
        return conn.execute(query).df().to_markdown(index=False)
    except Exception as e:
        return f"Erreur : {str(e)}"
    finally:
        conn.close()

@mcp.tool()
def indicateurs_complets_sante_et_social() -> str:
    """
    Renvoie les caractéristiques globales (Moyenne, Min, Max, Médiane) 
    dédiées à la santé mentale et à la consommation des réseaux sociaux.
    """
    conn = get_conn()
    try:
        # Utilisation des fonctions natives DuckDB pour générer un profil complet
        query = """
            SELECT 
                'Mental_Health_Score' as Indicateur, AVG(Mental_Health_Score) as Moyenne, MIN(Mental_Health_Score) as Min, MAX(Mental_Health_Score) as Max, MEDIAN(Mental_Health_Score) as Mediane FROM mental_health
            UNION ALL
            SELECT 
                'Avg_Daily_Usage_Hours' as Indicateur, AVG(Avg_Daily_Usage_Hours) as Moyenne, MIN(Avg_Daily_Usage_Hours) as Min, MAX(Avg_Daily_Usage_Hours) as Max, MEDIAN(Avg_Daily_Usage_Hours) as Mediane FROM mental_health
            UNION ALL
            SELECT 
                'Daily_Unlocks' as Indicateur, AVG(Daily_Unlocks) as Moyenne, MIN(Daily_Unlocks) as Min, MAX(Daily_Unlocks) as Max, MEDIAN(Daily_Unlocks) as Mediane FROM mental_health
            UNION ALL
            SELECT 
                'Sleep_Hours_Per_Night' as Indicateur, AVG(Sleep_Hours_Per_Night) as Moyenne, MIN(Sleep_Hours_Per_Night) as Min, MAX(Sleep_Hours_Per_Night) as Max, MEDIAN(Sleep_Hours_Per_Night) as Mediane FROM mental_health
        """
        res = conn.execute(query).df()
        # Arrondir les résultats pour la propreté du rendu
        for col in ['Moyenne', 'Min', 'Max', 'Mediane']:
            res[col] = res[col].round(2)
            
        return "### 🧠 RESUME ANALYTIQUE DES INDICATEURS CLES\n\n" + res.to_markdown(index=False)
    except Exception as e:
        return f"Erreur : {str(e)}"
    finally:
        conn.close()

if __name__ == '__main__':
    mcp.run()
