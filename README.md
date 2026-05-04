# solution-x

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude por busca vetorial 5-NN sobre 3M vetores 14D.

## Stack

- **Rust** + **glommio** (io_uring thread-per-core)
- **IVF k-means** (K=1024, nprobe=4 + bbox repair) com quantização int16
- AVX2 no kernel de scan (block 16 vetores), centroid scan via crate `wide`
- HTTP/1.1 sobre Unix Domain Socket
- LB: nginx stream L4

## Build

```bash
# Gera index.bin (~1min com rayon)
cargo run --release --bin build_index ../resources/references.json.gz resources/index.bin

# Roda stack local
docker compose build
docker compose up
```

## Validação de recall

```bash
cargo run --release --bin verify -- resources/index.bin ../resources/references.json.gz ../test/test-data.json 1000
```

Deve reportar `IVF == brute (frauds): 1000/1000 (100.00%)`.

## Submissão

A engine da Rinha precisa apenas de `docker-compose.yml`, `nginx.conf` e `info.json` na raiz da branch `submission`. O índice e o binário ficam na imagem publicada em `ghcr.io/athospugliese/solution-x:latest`.

```bash
bash scripts/make-submission.sh
# copia para dist/submission/, daí push para a branch submission no repo
```

## Layout

```
src/
├── main.rs            entry point
├── server.rs          server glommio (linux-only)
├── ivf.rs             IVF search + bbox repair + AVX2 scan
├── vectorizer.rs      14D feature extraction + parser ISO-8601
├── index.rs           formato RIVF1 + mmap reader
└── bin/
    ├── build_index.rs k-means++ build
    └── verify.rs      diff IVF vs brute-force
```

Nota: o server depende de io_uring; só compila/roda em Linux. Para dev em macOS, rodar via Docker.
