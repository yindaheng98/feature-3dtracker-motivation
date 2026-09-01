FROM ubuntu:24.04

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y curl ca-certificates git \
    patch jq ripgrep fd-find file \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /tmp/workspace /workspace