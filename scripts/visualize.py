import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Chargement et Nettoyage rapide (obligatoire pour les graphes)
def load_and_clean(path):
    df = pd.read_csv(path)
    # Remplissage des données vides pour ne pas fausser les graphes
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    df['Medical Cost'] = df['Medical Cost'].fillna(df['Medical Cost'].median())
    return df

def generate_visuals(df):
    # Configuration esthétique
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis")

    # --- A. APPROCHE DESCRIPTIVE : Histogrammes & Distributions ---
    # Distribution de la cible (Medical Cost)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Medical Cost'], kde=True, color='teal', bins=30)
    plt.title('Distribution Descriptive : Coûts Médicaux Annuels')
    plt.xlabel('Coût ($)')
    plt.show()

    # --- B. CORRÉLATIONS : Heatmap ---
    plt.figure(figsize=(10, 8))
    # Encodage rapide pour voir les corrélations avec les variables texte
    df_corr = df.copy()
    df_corr['Smoker_Num'] = df_corr['Smoker'].map({'yes': 1, 'no': 0})
    df_corr['Sex_Num'] = df_corr['Sex'].map({'male': 1, 'female': 0})
    
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Matrice de Corrélation (Diagnostic des dépendances)')
    plt.show()

    # --- C. COMBINAISONS DE FEATURES : Diagnostic Métier ---
    # Le but est de répondre au "Pourquoi on observe ces phénomènes ?"
    
    # Graphe 1 : BMI vs Coût avec distinction Fumeur (Le plus important)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='BMI', y='Medical Cost', hue='Smoker', style='Smoker', alpha=0.7)
    plt.title('Diagnostic : Impact combiné de l\'Obésité (BMI) et du Tabac')
    plt.show()

    # Graphe 2 : Âge vs Coût (Analyse des paliers)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Age', y='Medical Cost', hue='Smoker', estimator='mean')
    plt.title('Diagnostic : Évolution des coûts par âge et statut fumeur')
    plt.show()

    # Graphe 3 : Nombre d'enfants et frais
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Children', y='Medical Cost', palette='Set2')
    plt.title('Distribution des coûts selon le nombre d\'enfants')
    plt.show()

if __name__ == "__main__":
    # Assure-toi que le chemin vers ton dataset est correct
    data_path = 'data/medical_cost.csv' 
    df_medical = load_and_clean(data_path)
    generate_visuals(df_medical)