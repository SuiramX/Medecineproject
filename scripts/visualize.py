import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_clean_for_viz(path):
    """Charge et prépare les données pour la visualisation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} est introuvable.")
    
    df = pd.read_csv(path)
    # Nettoyage minimal pour les graphes
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())
    df['Medical Cost'] = df['Medical Cost'].fillna(df['Medical Cost'].median())
    return df

def plot_descriptive(df):
    """Génère l'histogramme de distribution des coûts."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Medical Cost'], kde=True, color='#3498db', bins=30)
    plt.title('Approche Descriptive : Distribution des Coûts Médicaux')
    plt.savefig('reports/figures/1_distribution_couts.png')
    plt.close() # Ferme la figure pour libérer la mémoire

def plot_correlation(df):
    """Génère la heatmap incluant les régions et le tabac."""
    plt.figure(figsize=(12, 8))
    df_corr = df.copy()
    df_corr['Smoker_Num'] = df_corr['Smoker'].map({'yes': 1, 'no': 0})
    df_corr['Sex_Num'] = df_corr['Sex'].map({'male': 1, 'female': 0})
    df_corr = pd.get_dummies(df_corr, columns=['Region'], prefix='Reg')
    
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Diagnostic : Matrice de Corrélation Complète')
    plt.savefig('reports/figures/2_heatmap_complete.png')
    plt.close()

def plot_bmi_diagnostic(df):
    """Génère le scatter plot global et le zoom sur les non-fumeurs."""
    # 1. Global
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='BMI', y='Medical Cost', hue='Smoker', alpha=0.7)
    plt.title('Diagnostic : Impact BMI & Tabac')
    plt.savefig('reports/figures/3_diagnostic_bmi_smoker.png')
    plt.close()

    # 2. Zoom Non-fumeurs
    df_no_smoker = df[df['Smoker'] == 'no']
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_no_smoker, x='BMI', y='Medical Cost', 
                scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'})
    plt.title('Zoom Diagnostic : BMI vs Coût (Non-fumeurs)')
    plt.savefig('reports/figures/4_bmi_zoom_non_fumeurs.png')
    plt.close()

def plot_regional_impact(df):
    """Génère le boxplot par région."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Region', y='Medical Cost', palette='Set3')
    plt.title('Diagnostic : Coûts par Région')
    plt.savefig('reports/figures/5_region_distribution.png')
    plt.close()

def run_all_visualizations(data_path):
    """Fonction de haut niveau pour tout lancer d'un coup."""
    os.makedirs('reports/figures', exist_ok=True)
    df = load_and_clean_for_viz(data_path)
    
    print("Génération des graphiques en cours...")
    plot_descriptive(df)
    plot_correlation(df)
    plot_bmi_diagnostic(df)
    plot_regional_impact(df)
    print("Tous les graphiques ont été sauvegardés dans reports/figures/")