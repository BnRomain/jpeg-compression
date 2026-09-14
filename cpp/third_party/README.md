# Bibliothèques tierces

| Fichier | Version | Source | Licence |
|---|---|---|---|
| `stb_image.h` | 2.30 | [nothings/stb](https://github.com/nothings/stb), branche `master` au 13/09/2026 | domaine public ou MIT |
| `stb_image_write.h` | 1.16 | idem | domaine public ou MIT |

Ces en-têtes sont copiés dans le dépôt : il n'y a pas de gestionnaire de paquets
C++ dans ce projet, donc Dependabot ne peut pas les suivre. Les nouvelles versions
et les correctifs de sécurité de stb se vérifient à la main sur le dépôt d'origine.

## Configuration

stb n'est compilé que dans `src/image_io.cpp`, avec :

- `STBI_ONLY_PNG`, `STBI_ONLY_JPEG`, `STBI_ONLY_BMP` : seuls les décodeurs utiles
  sont compilés (PSD, GIF, PIC, HDR, TGA et PNM sont exclus du programme) ;
- `STBI_MAX_DIMENSIONS 16384`, égal à `Image::max_dimension` (vérifié par un
  `static_assert`) : stb calcule des tailles de tampons avec des produits d'`int`,
  et `(16384 * 3 + 1) * 16384 < 2^31` garantit qu'ils ne débordent pas.

## Modifications locales

CodeQL signalait des produits d'entiers convertis en `size_t` après coup
(`cpp/integer-multiplication-cast-to-long`) : un débordement de l'`int` fausserait
la taille allouée. Les copies du dépôt convertissent l'un des facteurs en `size_t`
avant de multiplier.

| Fichier | Fonction | Alerte CodeQL | Correctif |
|---|---|---|---|
| `stb_image_write.h` | `stbiw__sbgrowf` | #13 | PR #5 |
| `stb_image_write.h` | `stbiw__encode_png_line` | #12 | PR #6 |
| `stb_image_write.h` | `stbi_write_png_to_mem` (`malloc` de `filt` et `line_buffer`, `memmove`) | #9, #10, #11 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__convert_format16` | #8 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__create_png_image_raw` | #7 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__psd_load` (non compilé) | #6 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__pic_load` (non compilé) | #5 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__gif_load_next` (non compilé) | #3, #4 | branche `docs-security-ci` |
| `stb_image.h` | `stbi__load_gif_main` (non compilé) | #1, #2 | branche `docs-security-ci` |

## Mettre à jour stb

1. Remplacer les deux fichiers par ceux de la nouvelle version.
2. Réappliquer les modifications ci-dessus si l'amont ne les a pas intégrées
   (`git log -p -- cpp/third_party` montre les lignes concernées).
3. Mettre à jour le tableau des versions, puis lancer `make test SANITIZE=1`.
