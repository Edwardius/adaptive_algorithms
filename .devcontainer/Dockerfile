# ===============================================
# Install Common Development Tools in Either Base
# ===============================================
FROM ubuntu:noble AS dev-tools

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-pip \
    python3-venv \
    openssh-client \
    git \
    curl \
    tmux \
    wget \
    htop \
    nano \
    tree \
    libssl-dev \
    x11-apps \
    xauth \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY .devcontainer/entrypoint.sh /tmp/entrypoint.sh

# ===============================================
# Add User Configuration
# ===============================================
FROM dev-tools AS user-conf
ARG USERNAME
ARG USER_GID
ARG USER_UID

# Cater image to user
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN existing_user=$(getent passwd ${USER_UID} | cut -d: -f1) \
    && if [ -n "$existing_user" ]; then userdel -r "$existing_user" 2>/dev/null || true; fi \
    && if ! getent group ${USER_GID} >/dev/null; then groupadd --gid ${USER_GID} ${USERNAME}; fi \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m $USERNAME --shell /bin/bash \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && cp /etc/skel/.bashrc /home/$USERNAME/.bashrc \
    && cp /etc/skel/.profile /home/$USERNAME/.profile \
    && chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc /home/$USERNAME/.profile \
    && cat /tmp/entrypoint.sh >> /home/$USERNAME/.bashrc \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Create Python virtual environment and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /home/$USERNAME/.venv \
    && /home/$USERNAME/.venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
