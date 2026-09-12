FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash curl coreutils findutils procps \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 evaluated \
    && useradd --uid 1000 --gid 1000 --home-dir /workspace/container-1 --no-create-home evaluated \
    && mkdir -p /opt/harness /run/harness /workspace \
    && chmod 700 /opt/harness /run/harness /root

COPY server.py runner.py /opt/harness/
RUN chmod 600 /opt/harness/*.py
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-u", "/opt/harness/server.py"]
