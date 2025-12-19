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
    plt.close()

def plot_correlation(df):
    """Génère la heatmap complète incluant régions, sexe et tabac."""
    plt.figure(figsize=(12, 10))
    df_corr = df.copy()
    
    df_corr['Smoker_Num'] = df_corr['Smoker'].map({'yes': 1, 'no': 0})
    df_corr['Sex_Num'] = df_corr['Sex'].map({'male': 1, 'female': 0})
    df_corr = pd.get_dummies(df_corr, columns=['Region'], prefix='Reg', dtype=int)
    
    corr_matrix = df_corr.select_dtypes(include=[np.number]).corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Diagnostic : Matrice de Corrélation Complète', pad=20)
    
    plt.savefig('reports/figures/2_heatmap_complete.png', bbox_inches='tight')
    plt.close()

def plot_bmi_diagnostic(df):
    """Génère le scatter plot global et les zooms par catégorie de fumeur."""
    # 1. Global (Relation BMI/Coût avec couleur Smoker)
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='BMI', y='Medical Cost', hue='Smoker', alpha=0.7)
    plt.title('Diagnostic : Impact combiné BMI & Tabac')
    plt.savefig('reports/figures/3_diagnostic_bmi_smoker.png')
    plt.close()

    # 2. Zoom Non-fumeurs (Tendance douce)
    df_no_smoker = df[df['Smoker'] == 'no']
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_no_smoker, x='BMI', y='Medical Cost', 
                scatter_kws={'alpha':0.4, 'color':'teal'}, line_kws={'color':'red'})
    plt.title('Zoom Diagnostic : BMI vs Coût (Non-fumeurs)')
    plt.savefig('reports/figures/4_bmi_zoom_non_fumeurs.png')
    plt.close()

    # 3. Zoom Fumeurs (Tendance forte)
    df_smoker = df[df['Smoker'] == 'yes']
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df_smoker, x='BMI', y='Medical Cost', 
                scatter_kws={'alpha':0.4, 'color':'orange'}, line_kws={'color':'red'})
    plt.title('Zoom Diagnostic : BMI vs Coût (Fumeurs)')
    plt.savefig('reports/figures/5_bmi_zoom_fumeurs.png')
    plt.close()

def plot_regional_boxplot(df):
    """Boxplot par région simple sans distinction fumeur."""
    plt.figure(figsize=(12, 7))
    my_order = df.groupby("Region")["Medical Cost"].median().sort_values(ascending=False).index

    sns.boxplot(
        data=df, 
        x='Region', 
        y='Medical Cost', 
        hue='Region',
        order=my_order, 
        palette='Set3',
        legend=False
    )

    plt.title('Diagnostic : Distribution des Coûts par Région', pad=20)
    plt.ylabel('Coût Médical ($)')
    plt.xlabel('Région (Triées par médiane)')
    
    # On passe à 6 pour suivre la suite des zooms
    plt.savefig('reports/figures/6_region_boxplot_simple.png', bbox_inches='tight')
    plt.close()

def run_all_visualizations(data_path):
    """Lancement de tout le pipeline visuel."""
    os.makedirs('reports/figures', exist_ok=True)
    df = load_and_clean_for_viz(data_path)
    
    print("Génération des graphiques en cours...")
    plot_descriptive(df)
    plot_correlation(df)
    plot_bmi_diagnostic(df)
    plot_regional_boxplot(df)
    print("Tous les graphiques ont été sauvegardés dans reports/figures/")