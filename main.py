import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# Création du dossier pour sauvegarder les images si pal
os.makedirs('reports/figures', exist_ok=True)
print("Dossier 'reports/figures' prêt pour la sauvegarde des graphes.")

# --- 1. CHARGEMENT ET NETTOYAGE RAPIDE ---
def load_and_clean_for_viz(path):
    """
    Charge les données et remplit les valeurs manquantes juste pour l'affichage.
    Ce n'est PAS le nettoyage pour le Machine Learning.
    """
    print(f"Chargement des données depuis {path}...")
    df = pd.read_csv(path)
    
    # Remplissage de sécurité (médiane) pour éviter les erreurs de graphiques
    # si ton dataset brut contient des vides.
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
        
    print("Données chargées et nettoyées pour la visualisation.")
    return df

# --- 2. APPROCHE DESCRIPTIVE : Histogrammes ---
def plot_descriptive(df):
    print("Génération des graphes descriptifs...")
    # Configuration esthétique globale
    sns.set_theme(style="whitegrid", context="talk")

    # Histogramme de la cible (Target Distribution)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Medical Cost'], kde=True, color='#3498db', bins=40)
    plt.title('Approche Descriptive : Distribution des Coûts Médicaux Annuels')
    plt.xlabel('Frais Médicaux ($)')
    plt.ylabel("Nombre d'individus")
    
    # Sauvegarde et affichage
    save_path = 'reports/figures/1_distribution_couts.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Sauvegardé : {save_path}")
    plt.show()

# --- 3. APPROCHE DIAGNOSTIQUE : Corrélations & Features ---
def plot_diagnostic(df):
    print("\nGénération des graphes diagnostiques (Corrélations & Features)...")
    sns.set_theme(style="whitegrid", context="talk")

    # --- A. Matrice de Corrélation ---
    # On crée une copie pour encoder temporairement les textes en chiffres
    df_corr = df.copy()
    # Encodage manuel simple juste pour voir la corrélation
    mappings = {'yes': 1, 'no': 0, 'male': 0, 'female': 1}
    df_corr['Smoker_Code'] = df_corr['Smoker'].map(mappings)
    df_corr['Sex_Code'] = df_corr['Sex'].map(mappings)
    
    # Calcul de la corrélation sur les colonnes numériques uniquement
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt=".2f", linewidths=1)
    plt.title('Diagnostique : Matrice de Corrélation')
    
    save_path_corr = 'reports/figures/2_correlation_heatmap.png'
    plt.savefig(save_path_corr, bbox_inches='tight')
    print(f"Sauvegardé : {save_path_corr}")
    plt.show()

    # --- B. Combinaison de Features (Le "Cœur" du diagnostic métier) ---
    # C'est CE graphe qui explique le "Pourquoi" : l'impact combiné BMI + Tabac
    plt.figure(figsize=(14, 8))
    scatter = sns.scatterplot(
        data=df, 
        x='BMI', 
        y='Medical Cost', 
        hue='Smoker',   # Change la couleur selon si fumeur ou non
        style='Smoker', # Change la forme du point
        palette='deep',
        s=60,           # Taille des points
        alpha=0.7       # Transparence
    )
    plt.title('Diagnostique Métier : Impact combiné de l\'Obésité (BMI) et du Tabac sur les Coûts')
    plt.axvline(x=30, color='red', linestyle='--', label='Seuil Obésité (BMI 30)')
    plt.legend(title='Fumeur ?')
    
    save_path_diag = 'reports/figures/3_diagnostic_bmi_smoker.png'
    plt.savefig(save_path_diag, bbox_inches='tight')
    print(f"Sauvegardé : {save_path_diag}")
    plt.show()

# --- 4. EXÉCUTION PRINCIPALE ---
if __name__ == "__main__":
    # Chemin vers ton fichier CSV (vérifie qu'il est bien dans un dossier 'data')
    DATA_PATH = 'data/medical_costs.csv'
    
    # Lancement du pipeline de visualisation
    try:
        df = load_and_clean_for_viz(DATA_PATH)
        plot_descriptive(df)
        plot_diagnostic(df)
        print("\n--- Visualisation terminée avec succès ! ---")
        print("Retrouve tes images dans le dossier 'reports/figures/'.")
    except FileNotFoundError:
        print(f"\nERREUR : Le fichier {DATA_PATH} est introuvable.")
        print("Vérifie que tu as bien créé le dossier 'data' et mis le CSV dedans.")