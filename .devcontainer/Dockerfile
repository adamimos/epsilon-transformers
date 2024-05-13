ARG PYTHON_VERSION=3.11.9
ARG AWS_CLI_VERSION=2.15.49

FROM amazon/aws-cli:${AWS_CLI_VERSION} AS aws-cli

FROM python:${PYTHON_VERSION}-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bash-completion \
        groff \
        jq \
        less \
        nano \
        rsync \
 && rm -rf /var/lib/apt/lists/*

ARG CUDA_DISTRO=ubuntu2204
ARG CUDA_VERSION=12.3

RUN CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_DISTRO}/x86_64" \
 && CUDA_GPG_KEY=/usr/share/keyrings/nvidia-cuda.gpg \
 && wget -O- "${CUDA_REPO}/3bf863cc.pub" | gpg --dearmor > "${CUDA_GPG_KEY}" \
 && echo "deb [signed-by=${CUDA_GPG_KEY} arch=amd64] ${CUDA_REPO}/ /" > /etc/apt/sources.list.d/nvidia-cuda.list \
 && apt-get update -y \
 && apt-get install -yq --no-install-recommends \
        cuda-libraries-${CUDA_VERSION} \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-${CUDA_VERSION}/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=aws-cli /usr/local/aws-cli/v2/current /usr/local

WORKDIR /sources
COPY pyproject.toml .
RUN mkdir epsilon_transformers \
 && touch epsilon_transformers/__init__.py \
 && pip install .

ARG APP_USER=simplex
ARG APP_DIR=/home/${APP_USER}/app
ARG USER_ID=1000
ARG GROUP_ID=1000

RUN groupadd -g ${GROUP_ID} ${APP_USER} \
 && useradd -m -u ${USER_ID} -g ${APP_USER} -s /bin/bash ${APP_USER} \
 && mkdir -p ${APP_DIR} /home/${APP_USER}/.aws \
 && chown -R ${USER_ID}:${GROUP_ID} ${APP_DIR} /home/${APP_USER}

WORKDIR ${APP_DIR}
COPY . .
RUN pip install -e .
USER ${APP_USER}
