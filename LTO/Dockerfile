FROM python:3.7-slim-buster


WORKDIR /work

RUN python -m pip install --no-cache-dir --upgrade "pip==23.2.1" && \
    python -m pip install --no-cache-dir \
      "cma==3.0.3" \
      "numpy==1.18.5" \
      "matplotlib==3.3.4" \
      "seaborn==0.11.1" \
      "tensorflow==1.15.0" \
      "protobuf==3.17.1"

RUN printf '%s\n' \
      'deb [check-valid-until=no] http://archive.debian.org/debian buster main' \
      'deb [check-valid-until=no] http://archive.debian.org/debian-security buster/updates main' \
      > /etc/apt/sources.list && \
    echo 'Acquire::Check-Valid-Until "false";' \
      > /etc/apt/apt.conf.d/99archive && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "setuptools==68.0.0" \
    "wheel==0.41.3" \
    "Cython==0.29.36"

RUN python -m pip install --no-cache-dir \
    --no-build-isolation \
    --no-deps \
    --ignore-requires-python \
    "coco-experiment==2.6.100"

RUN python -m pip install --no-cache-dir \
    --no-deps \
    --ignore-requires-python \
    "cocopp==2.6.4" \
      "protobuf==3.17.1"

