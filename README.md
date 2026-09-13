# Compression d'image par DCT et matrices creuses (CSR)

Compression d'image inspirée de la norme JPEG : l'image est découpée en blocs
8x8, chaque bloc passe dans le domaine fréquentiel par la transformée en
cosinus discrète (DCT), les coefficients sont quantifiés puis les hautes
fréquences sont supprimées. La matrice obtenue est très creuse : elle est
stockée au format CSR (Compressed Sparse Row), qui ne conserve que les valeurs
non nulles.

Le dépôt contient deux implémentations du même algorithme.

| Dossier | Langage | Contexte | Contenu |
|---|---|---|---|
| [`python/`](python) | Python (numpy, scipy, Streamlit) | Projet MAM3, janvier 2026 : Romain Ben, Evrard Lecureur, Zouhair Saitout | module de compression, application Streamlit, tests, rapport et présentation |
| [`cpp/`](cpp) | C++20 | Projet C++ MAM4, septembre 2026 : Romain Ben, Karim Zrig | programme `jpeg_csr` en ligne de commande, tests, rapport de synthèse et présentation |

## Algorithme commun

1. Rognage de l'image aux multiples de 8 et centrage des intensités dans [-128, 127].
2. DCT de chaque bloc : `D = P M Pᵀ`, avec `P` la matrice orthogonale de la DCT-II.
3. Quantification : division terme à terme par la matrice `Q` (éventuellement `alpha * Q`) et troncature.
4. Seuil et suppression des hautes fréquences.
5. Stockage des coefficients de chaque canal R, G, B en CSR.
6. Décompression : multiplication par `Q`, DCT inverse `M = Pᵀ D P`, recentrage.

## Démarrage rapide

Version Python, démo en ligne : [jpeg-csr-compression.streamlit.app](https://jpeg-csr-compression.streamlit.app/)

```bash
cd python
pip install -r requirements.txt
streamlit run app.py
```

Version C++ :

```bash
cd cpp
make test
make
./jpeg_csr compress images/astronaut.png --alpha 5
```

## Documents

- Python : [rapport](python/docs/Rapport.pdf), [présentation](python/docs/Presentation.pdf)
- C++ : [rapport de synthèse](cpp/docs/rapport.pdf), [présentation](cpp/docs/presentation.pdf)
