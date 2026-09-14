# Third-party libraries

| File | Version | Source | License |
|---|---|---|---|
| `stb_image.h` | 2.30 | [nothings/stb](https://github.com/nothings/stb), `master` branch as of 2026-09-13 | public domain or MIT |
| `stb_image_write.h` | 1.16 | same | public domain or MIT |

These headers are copied into the repository: the C++ project has no package
manager, so Dependabot cannot track them. New stb releases and security fixes
must be checked manually in the upstream repository.

## Configuration

stb is only compiled in `src/image_io.cpp`, with:

- `STBI_ONLY_PNG`, `STBI_ONLY_JPEG`, `STBI_ONLY_BMP`: only the decoders the
  program needs are compiled (PSD, GIF, PIC, HDR, TGA and PNM are left out);
- `STBI_MAX_DIMENSIONS 16384`, equal to `Image::max_dimension` (checked by a
  `static_assert`): stb computes buffer sizes with `int` products, and
  `(16384 * 3 + 1) * 16384 < 2^31` guarantees that they do not overflow.

## Local modifications

CodeQL reported integer products converted to `size_t` only after the
multiplication (`cpp/integer-multiplication-cast-to-long`): an `int` overflow
would make the allocated size wrong. The copies in this repository convert one
of the factors to `size_t` before multiplying.

| File | Function | CodeQL alert | Fix |
|---|---|---|---|
| `stb_image_write.h` | `stbiw__sbgrowf` | #13 | PR #5 |
| `stb_image_write.h` | `stbiw__encode_png_line` | #12 | PR #6 |
| `stb_image_write.h` | `stbi_write_png_to_mem` (`malloc` of `filt` and `line_buffer`, `memmove`) | #9, #10, #11 | PR #7 |
| `stb_image.h` | `stbi__convert_format16` | #8 | PR #7 |
| `stb_image.h` | `stbi__create_png_image_raw` | #7 | PR #7 |
| `stb_image.h` | `stbi__psd_load` (not compiled) | #6 | PR #7 |
| `stb_image.h` | `stbi__pic_load` (not compiled) | #5 | PR #7 |
| `stb_image.h` | `stbi__gif_load_next` (not compiled) | #3, #4 | PR #7 |
| `stb_image.h` | `stbi__load_gif_main` (not compiled) | #1, #2 | PR #7 |

## Updating stb

1. Replace both files with those of the new version.
2. Reapply the modifications above if upstream has not integrated them
   (`git log -p -- cpp/third_party` shows the lines involved).
3. Update the version table, then run `make test SANITIZE=1`.
