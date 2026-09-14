# Version Python (projet MAM3)

Implémentation d'origine du projet : module de compression `jpeg_compression.py`
et application Streamlit `app.py`. Présentation complète du projet dans le
[README principal](../README.md).

Réalisé par Romain Ben, Evrard Lecureur et Zouhair Saitout (MAM3, Polytech Nice Sophia).

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

Démo en ligne : [jpeg-csr-compression.streamlit.app](https://jpeg-csr-compression.streamlit.app/)

## Lancer les tests

```bash
pip install -r requirements-dev.txt
python -m pytest -v
```

- `tests/test_compression.py` : matrice DCT, rognage, compression, décompression, conversion CSR ;
- `tests/test_app.py` : démarrage de l'application Streamlit.

Les versions des dépendances sont figées dans `requirements.txt` et
`requirements-dev.txt` : Dependabot propose chaque semaine les mises à jour.

## Documents

- [Rapport](docs/Rapport.pdf)
- [Présentation](docs/Presentation.pdf)
