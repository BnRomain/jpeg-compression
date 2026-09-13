# Compression d'image DCT + CSR : version C++

Portage en C++20 de la [version Python](../python) du projet : compression
d'image inspirée de JPEG (DCT sur des blocs 8x8, quantification, troncature des
hautes fréquences) et stockage des coefficients au format creux CSR. Le
programme `jpeg_csr` fait en ligne de commande ce que font `jpeg_compression.py`
et l'application Streamlit, et reprend les analyses du rapport MAM3.

Projet de programmation C++, MAM4 Polytech Nice Sophia : Romain Ben et Karim Zrig.

- Rapport de synthèse (2 pages) : [`docs/rapport.pdf`](docs/rapport.pdf)
- Présentation : [`docs/presentation.pdf`](docs/presentation.pdf)

## Compiler

Prérequis : `g++` compatible C++20 (GCC 10 ou plus récent) et `make`. Sous
Windows, MSYS2 fournit `g++` et `mingw32-make`, à utiliser à la place de `make`.

```bash
make                    # construit ./jpeg_csr
make test               # exécute les tests unitaires
make demo               # compresse images/astronaut.png avec plusieurs réglages
make test SANITIZE=1    # tests sous AddressSanitizer et UBSan (Linux, macOS)
```

Options de compilation du cours : `-std=c++20 -Wall -Wextra -pedantic`.

## Utiliser

```bash
./jpeg_csr compress images/astronaut.png --alpha 5 --out resultats/alpha5
./jpeg_csr decompress resultats/alpha5/astronaut.csr resultats/relue.png
```

| Option | Rôle | Défaut |
|---|---|---|
| `--table standard\|uniform\|low\|high` | matrice de quantification Q | `standard` |
| `--alpha A` | facteur de qualité : Q devient A x Q | `1` |
| `--threshold S` | annule les coefficients quantifiés de valeur absolue < S | `2` |
| `--mask square\|triangle` | troncature carrée (k, l < F) ou triangulaire (k + l < F) | `square` |
| `--cutoff F` | fréquence de coupure | `6` |
| `--noise P` | bruit poivre et sel de probabilité P ajouté avant compression | `0` |
| `--out DOSSIER` | dossier des résultats | `resultats` |

Les valeurs par défaut sont celles de l'application Python. `compress` écrit
`<nom>.csr` (les trois matrices CSR, équivalent du `.npz`) et
`<nom>_reconstruite.png`, puis affiche les indicateurs de Streamlit :

```text
Coefficients   : 45566 non nuls sur 786432 (taux de conservation 5.79 %)
Qualité        : erreur L2 relative 5.43 %, PSNR 30.49 dB
Mémoire dense  : 6144.00 Ko (float64, comme img.nbytes)
Mémoire CSR    : 273.00 Ko (valeurs int16, indices int32)
Gain mémoire   : dense / CSR = 22.51x
Fichier .csr   : 273.52 Ko, image source 773.00 Ko (source / .csr = 2.83x)
Temps          : compression 28.23 ms, décompression 21.55 ms
```

## Organisation du code

```text
cpp/
├── Makefile
├── include/           un en-tête commenté par module
├── src/               définitions et main.cpp
├── tests/             tests unitaires (assert)
├── scripts/           comparaison avec Python, génération des figures
├── images/            images de test
├── third_party/       stb_image et stb_image_write (domaine public)
└── docs/              rapport et présentation (LaTeX et PDF)
```

| Module | Rôle | Notions du cours |
|---|---|---|
| `Matrix8` | bloc 8x8 et produit matriciel | `std::array`, surcharge de `operator()` et `operator*`, règle de zéro |
| `Dct` | matrice P calculée une fois, D = P M Pᵀ et M = Pᵀ D P | classe, liste d'initialisation, méthodes `const` |
| `QuantizationTable` | Q standard, uniforme, basses ou hautes fréquences, facteur alpha | invariant (diviseurs >= 1), `explicit`, exceptions |
| `FrequencyMask` | interface, implémentée par `SquareMask` (Python) et `TriangleMask` (sujet) | classe abstraite, `virtual`, `override`, destructeur virtuel |
| `Image` | image RGB, rognage aux multiples de 8 | invariant, `std::vector`, accès `const` et non `const`, `at()` vérifié |
| `image_io` | lecture PNG/JPEG et écriture PNG via stb | RAII (`StbPixels`), copie interdite (`= delete`) |
| `SparseMatrix` | matrice CSR écrite à la main | invariant vérifié, déplacement (`std::move`) |
| `CompressedImage` | 3 matrices CSR + Q, fichier binaire `.csr` | composition, flux `std::ofstream` (RAII) |
| `codec` | `compress` et `decompress` | `const T&`, référence sur l'interface |
| `metrics`, `noise` | erreur L2 relative, PSNR, bruit poivre et sel | `T&` pour modifier, `<random>` |
| `options`, `main` | ligne de commande et affichage | `std::string`, `enum class`, `try` / `catch` |

## Correspondance avec la version Python

| Python (`jpeg_compression.py`, `app.py`) | C++ |
|---|---|
| `init(img)` | `Image::cropped_to_blocks()` et centrage dans `compress` |
| `DCT2_P()`, `D_matrix(img_8, P)` | classe `Dct` |
| `compression(img, seuil)` | `compress(image, table, threshold, mask)` |
| `decompression(img_compressed)` | `decompress(compressed)` |
| `csr_matrix(canal.astype(np.int16))` | classe `SparseMatrix` |
| `np.savez_compressed(...)` | `save_compressed` et `load_compressed` |
| métriques de l'application | affichage de `jpeg_csr compress` |

## Validation

- `make test` reprend les tests pytest de la version Python (orthogonalité de P,
  rognage, suppression des hautes fréquences, bornes, conversion CSR) et vérifie
  en plus les invariants de chaque classe, l'aller-retour par fichier `.csr` et
  l'analyse de la ligne de commande. La CI GitHub les relance sous sanitizers.
- `python scripts/compare_with_python.py images/astronaut.png` compresse la même
  image avec les deux versions : 786 407 coefficients sur 786 432 sont identiques
  et le nombre de non nuls est le même. Les 25 autres diffèrent de 1 : en Python,
  l'image passe par /255 puis x255 et D/Q vaut par exemple 23.999999999999996 au
  lieu de 24, que la troncature ramène à 23. Le C++ est environ 11 fois plus
  rapide (50 ms contre 543 ms pour compression et décompression).
- `python scripts/make_figures.py` régénère les figures de `docs/figures`.

## Crédits

- [stb](https://github.com/nothings/stb) de Sean Barrett, domaine public.
- Images de test issues de [scikit-image](https://scikit-image.org/) :
  `astronaut.png` (NASA, domaine public) et `coffee.png` (Rachel Michetti, CC0).
