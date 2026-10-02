FROM nvcr.io/nvidia/isaac-sim:6.0.1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && apt-get clean \
    && find /var/lib/apt/lists -mindepth 1 -delete
RUN /isaac-sim/python.sh -m pip install --no-cache-dir \
      "simready-foundation-tier-core[benchmark]==2026.7.1" \
      "simready-validate==2026.7.1" \
      "simready-isaac-asset-transformer[usd] @ git+https://github.com/NVIDIA/simready-foundation.git@07639a286a7afa7e2b4e4d02b4e541fdff16523a#subdirectory=nv_core/cip_specs/isaac_asset_transformer"

ENV ACCEPT_EULA=Y
