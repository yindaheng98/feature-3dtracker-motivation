FROM ubuntu:20.04

SHELL ["/bin/bash", "-lc"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl ca-certificates git patch jq ripgrep fd-find file \
    && rm -rf /var/lib/apt/lists/*
