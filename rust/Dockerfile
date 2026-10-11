# syntax=docker/dockerfile:1.7
#
# Small image with only the Rust binary, ONNX Runtime and the exported models.
#   docker build -t brainssl-infer -f rust/Dockerfile .
#   docker run --rm -v "$PWD/scans:/scans" brainssl-infer segment \
#       --model /models/segmentation --input /scans/case.nii.gz --output /scans/case_seg.nii.gz

FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY rust/Cargo.toml rust/Cargo.lock ./
# Build the dependency graph once against a stub so source edits stay cheap.
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src
COPY rust/src ./src
RUN touch src/main.rs && cargo build --release

FROM debian:bookworm-slim AS runtime
ARG ORT_VERSION=1.22.0
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    | tar -xz -C /opt \
 && mv "/opt/onnxruntime-linux-x64-${ORT_VERSION}" /opt/onnxruntime \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
COPY --from=build /src/target/release/brainssl-infer /usr/local/bin/brainssl-infer
COPY exports /models
RUN useradd --create-home --uid 1000 brainssl
USER brainssl
ENTRYPOINT ["brainssl-infer"]
CMD ["--help"]
