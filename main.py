from scripts.visualize import run_all_visualizations

def main():
    # Chemin vers ton fichier de données
    DATA_PATH = 'data/medical_costs.csv'
    
    try:
        print("--- DÉMARRAGE DU PIPELINE DE VISUALISATION ---")
        run_all_visualizations(DATA_PATH)
        print("--- PIPELINE TERMINÉ AVEC SUCCÈS ---")
        
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()