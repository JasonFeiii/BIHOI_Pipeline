# V2 publication scope

This is a new private source repository based on v1 at `3c7c9ea`, with the current
local 6D tracking and WebUI modifications. It starts with a clean source snapshot;
v1 history and the measured calibration files uploaded there are not republished.
The original repository is unchanged.

The publication audit checks the exact Git index for runtime data, model weights,
private configuration, secret patterns, unexpected binary files, symlinks,
submodules and unresolved conflicts. Source files over 5 MiB fail unless explicitly
allowlisted as a licensed display asset. Four bundled font files are SHA-256
allowlisted in `configs/source-assets.json`; the existing SAM3 tokenizer is retained.

Model caches, videos, meshes, measured camera calibration, native build products,
external MoSE3/WiLoR installations, job logs and local environments are ignored.
Tests may use ignored links to operator-supplied assets; these links are not part
of a Git clone. `.env.example` contains only configuration examples.

See `docs/upload-manifest.tsv` for published paths and hashes, `THIRD_PARTY.md`
for source/model terms, and `docs/VALIDATION_V2.md` for validation scope. Pattern
scanning is not a guarantee against every possible secret representation.
