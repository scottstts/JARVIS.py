# Vendored sqlite-vec

This directory vendors the upstream `sqlite-vec` amalgamation source used by the Jarvis dev image build.

Pinned upstream snapshot:

- release tag: `v0.1.7-alpha.10`
- release asset: `sqlite-vec-0.1.7-alpha.10-amalgamation.zip`
- downloaded archive sha256: `51457d5137ee7a649b0ff0602effeb22bf5da1275995df1bb80e4a22b7a4c46a`
- embedded upstream source commit: `ce7b53e8490e40cd44af73aee463f99b6b50598c`

Vendored source layout:

- `v0.1.7-alpha.10/amalgamation/sqlite-vec.c`
- `v0.1.7-alpha.10/amalgamation/sqlite-vec.h`

Licensing:

- upstream license texts are copied under `licenses/`
- upstream ships both `LICENSE-MIT` and `LICENSE-APACHE`

Build usage:

- `Dockerfile.dev` compiles `/opt/sqlite-vec/vec0.so` from this vendored source during image build
- the build fails unless SQLite can load that compiled extension and `select vec_version()` returns the pinned version
- runtime code prefers `/opt/sqlite-vec/vec0.so` explicitly before falling back to the Python package loader
