python -m venv venv
call .\venv\Scripts\activate
python -m pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo ERREUR : Le fichier requirements.txt est introuvable.
)
echo Appuyez sur une touche pour fermer...
pause