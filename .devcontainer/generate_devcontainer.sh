#!/bin/bash

# Copyright (c) 2025-present WATonomous. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# generate_devcontainer.sh
# Usage:
#   ./generate_devcontainer.sh <ros_distro> <container_type> [nvidia_container_tag]

set -e

USERNAME=${USER:-vscode}

echo "Generating devcontainer configuration..."

# Generate container name and build args based on type
CONTAINER_NAME="ECE 457A Development Container"
BUILD_ARGS='"USERNAME": "'$USERNAME'",
  "USER_UID": "'$(id -u)'",
  "USER_GID": "'$(id -g)'"'
RUN_ARGS='"--network=host"'

# Generate the devcontainer.json with shared structure
cat > .devcontainer/devcontainer.json << EOF
{
  "name": "$CONTAINER_NAME",
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    "args": {
      $BUILD_ARGS
    }
  },
  "runArgs": [
    $RUN_ARGS
  ],
  "containerEnv": {
    "DISPLAY": "\${localEnv:DISPLAY}"
  },
  "mounts": [
    "source=\${localWorkspaceFolder},target=/deep_ros_ws,type=bind,consistency=cached",
    "source=/tmp/.X11-unix,target=/tmp/.X11-unix,type=bind"
  ],
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-vscode.cpptools",
        "ms-python.python",
        "ms-vscode.cmake-tools",
        "redhat.vscode-yaml",
        "ms-iot.vscode-ros"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/bin/python3",
        "terminal.integrated.shell.linux": "/bin/bash"
      }
    }
  },
  "remoteUser": "$USERNAME"
}
EOF

echo "Devcontainer configuration generated successfully!"
echo "Files created:"
echo "  - .devcontainer/devcontainer.json"
echo ""
echo "Environment variables set for this session"
echo ""
echo "You can now:"
echo "  1. Open the command palette (Ctrl+Shift+P)"
echo "  2. Run 'Dev Containers: Rebuild and Reopen in Container'"
