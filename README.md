# ECE 457A

Workspace for ECE 457A (Cooperative and Adaptive Algorithms) assignments.

## Dev Container Setup

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension

### Getting Started

1. Clone this repository and open it in VS Code.

2. Open the command palette (`Ctrl+Shift+P`), run **Tasks: Run Task**, and select **Setup Devcontainer**. This generates a `devcontainer.json` with your `USERNAME`, `USER_UID`, and `USER_GID` so file permissions work correctly with bind mounts.

3. Open the command palette again and run **Dev Containers: Rebuild and Reopen in Container**.

VS Code will build the Docker image and reopen the workspace inside the container.

Alternatively you can setup a Python Venv:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt
```

### X11 Forwarding

The devcontainer is configured to forward X11 so GUI applications (e.g. matplotlib plots) display on your host. This works out of the box on Linux — the container mounts `/tmp/.X11-unix` and passes through your `DISPLAY` variable.
