FROM rust:1.85-slim AS build
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev gcc make git \
    && rm -rf /var/lib/apt/lists/*
COPY .cargo .cargo
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release --bin solution-x

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /app/target/release/solution-x /usr/local/bin/
COPY resources/mcc_risk.json /data/mcc_risk.json
COPY resources/index.bin /data/index.bin
ENV INDEX_PATH=/data/index.bin
ENV MCC_PATH=/data/mcc_risk.json
ENTRYPOINT ["/usr/local/bin/solution-x"]
