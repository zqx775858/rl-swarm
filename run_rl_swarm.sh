#!/bin/bash

set -euo pipefail

# General arguments
ROOT=$PWD

export PUB_MULTI_ADDRS
export PEER_MULTI_ADDRS
export HOST_MULTI_ADDRS
export IDENTITY_PATH
export CONNECT_TO_TESTNET
export ORG_ID
export HF_HUB_DOWNLOAD_TIMEOUT=120  # 2 minutes

# Check if public multi-address is given else set to default
DEFAULT_PUB_MULTI_ADDRS=""
PUB_MULTI_ADDRS=${PUB_MULTI_ADDRS:-$DEFAULT_PUB_MULTI_ADDRS}

# Check if peer multi-address is given else set to default
DEFAULT_PEER_MULTI_ADDRS="/ip4/38.101.215.13/tcp/30002/p2p/QmQ2gEXoPJg6iMBSUFWGzAabS2VhnzuS782Y637hGjfsRJ" # gensyn coordinator node
PEER_MULTI_ADDRS=${PEER_MULTI_ADDRS:-$DEFAULT_PEER_MULTI_ADDRS}

# Check if host multi-address is given else set to default
DEFAULT_HOST_MULTI_ADDRS="/ip4/0.0.0.0/tcp/38331"
HOST_MULTI_ADDRS=${HOST_MULTI_ADDRS:-$DEFAULT_HOST_MULTI_ADDRS}

# Path to an RSA private key. If this path does not exist, a new key pair will be created.
# Remove this file if you want a new PeerID.
DEFAULT_IDENTITY_PATH="$ROOT"/swarm.pem
IDENTITY_PATH=${IDENTITY_PATH:-$DEFAULT_IDENTITY_PATH}

SMALL_SWARM_CONTRACT="0x69C6e1D608ec64885E7b185d39b04B491a71768C"
BIG_SWARM_CONTRACT="0x6947c6E196a48B77eFa9331EC1E3e45f3Ee5Fd58"

# Will ignore any visible GPUs if set.
CPU_ONLY=${CPU_ONLY:-""}

# Set if successfully parsed from modal-login/temp-data/userData.json.
ORG_ID=${ORG_ID:-""}

GREEN_TEXT="\033[32m"
BLUE_TEXT="\033[34m"
RESET_TEXT="\033[0m"

echo_green() {
    echo -e "$GREEN_TEXT$1$RESET_TEXT"
}

echo_blue() {
    echo -e "$BLUE_TEXT$1$RESET_TEXT"
}

ROOT_DIR="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)"

# Function to clean up the server process upon exit
cleanup() {
    echo_green ">> Shutting down trainer..."

    # Remove modal credentials if they exist
    rm -r $ROOT_DIR/modal-login/temp-data/*.json 2> /dev/null || true

    # Kill all processes belonging to this script's process group
    kill -- -$$ || true

    exit 0
}

trap cleanup EXIT

echo -e "\033[38;5;224m"
cat << "EOF"
    ██████  ██            ███████ ██     ██  █████  ██████  ███    ███
    ██   ██ ██            ██      ██     ██ ██   ██ ██   ██ ████  ████
    ██████  ██      █████ ███████ ██  █  ██ ███████ ██████  ██ ████ ██
    ██   ██ ██                 ██ ██ ███ ██ ██   ██ ██   ██ ██  ██  ██
    ██   ██ ███████       ███████  ███ ███  ██   ██ ██   ██ ██      ██

    From Gensyn

EOF

# 检测可用的GPU数量
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo_green ">> 检测到 $GPU_COUNT 个GPU"

# 检查RL-swarm目录
RL_SWARM_DIR="$PWD"
if [ ! -f "$RL_SWARM_DIR/run_rl_swarm.sh" ]; then
    echo_green ">> 请在RL-swarm主目录中运行此脚本"
    exit 1
fi

# 创建工作目录
WORK_DIR="$HOME/rl-swarm-multi-gpu"
mkdir -p "$WORK_DIR"
echo_green ">> 创建工作目录: $WORK_DIR"

# 询问用户是否连接到Testnet
while true; do
    echo -en $GREEN_TEXT
    read -p ">> 是否连接到Testnet? [Y/n] " yn
    echo -en $RESET_TEXT
    yn=${yn:-Y}  # 默认为"Y"
    case $yn in
        [Yy]*)  CONNECT_TO_TESTNET=true && break ;;
        [Nn]*)  CONNECT_TO_TESTNET=false && break ;;
        *)  echo ">>> 请回答yes或no." ;;
    esac
done

# 询问用户选择哪个swarm
while true; do
    echo -en $GREEN_TEXT
    read -p ">> 您想加入哪个swarm (Math (A) 或 Math Hard (B))? [A/b] " ab
    echo -en $RESET_TEXT
    ab=${ab:-A}  # 默认为"A"
    case $ab in
        [Aa]*)  USE_BIG_SWARM=false && break ;;
        [Bb]*)  USE_BIG_SWARM=true && break ;;
        *)  echo ">>> 请回答A或B." ;;
    esac
done

# 询问用户选择参数大小
while true; do
    echo -en $GREEN_TEXT
    read -p ">> 您想使用多少参数(单位:十亿)? [0.5, 1.5, 7, 32, 72] " pc
    echo -en $RESET_TEXT
    pc=${pc:-0.5}  # 默认为"0.5"
    case $pc in
        0.5 | 1.5 | 7 | 32 | 72) PARAM_B=$pc && break ;;
        *)  echo ">>> 请在[0.5, 1.5, 7, 32, 72]中选择." ;;
    esac
done

# 询问Hugging Face token
echo -en $GREEN_TEXT
read -p ">> 您想将训练的模型推送到Hugging Face Hub吗? [y/N] " yn
echo -en $RESET_TEXT
yn=${yn:-N} # 默认为"N"
case $yn in
    [Yy]*) read -p "请输入您的Hugging Face访问令牌: " HUGGINGFACE_ACCESS_TOKEN ;;
    [Nn]*) HUGGINGFACE_ACCESS_TOKEN="None" ;;
    *) echo ">>> 没有给出答案，因此不会将模型推送到Hugging Face Hub" && HUGGINGFACE_ACCESS_TOKEN="None" ;;
esac

# 为每个GPU创建和启动实例
for ((i=0; i<$GPU_COUNT; i++)); do
    GPU_DIR="$WORK_DIR/gpu$i"
    mkdir -p "$GPU_DIR"
    
    # 复制必要的文件
    cp -r "$RL_SWARM_DIR"/* "$GPU_DIR/"
    
    # 修改modal-login的端口
    PORT=$((3000 + $i))
    if [ -f "$GPU_DIR/modal-login/.env" ]; then
        sed -i "s/PORT=.*/PORT=$PORT/" "$GPU_DIR/modal-login/.env"
    else
        echo "PORT=$PORT" > "$GPU_DIR/modal-login/.env"
    fi
    
    # 创建启动脚本
    cat > "$GPU_DIR/start_gpu$i.sh" << EOF
#!/bin/bash
export CUDA_VISIBLE_DEVICES=$i
cd "$GPU_DIR"
./run_rl_swarm.sh
EOF
    
    chmod +x "$GPU_DIR/start_gpu$i.sh"
    
    echo_green ">> GPU $i 的配置已创建在 $GPU_DIR"
    echo_blue ">> 要启动GPU $i，请运行: $GPU_DIR/start_gpu$i.sh"
    echo_blue ">> 登录界面将在端口 $PORT 上可用"
done

# 创建一个一键启动所有GPU的脚本
cat > "$WORK_DIR/start_all_gpus.sh" << EOF
#!/bin/bash
GREEN_TEXT="\033[32m"
RESET_TEXT="\033[0m"

echo -e "\${GREEN_TEXT}>> 启动所有GPU实例...\${RESET_TEXT}"

for ((i=0; i<$GPU_COUNT; i++)); do
    GPU_DIR="$WORK_DIR/gpu\$i"
    echo -e "\${GREEN_TEXT}>> 启动GPU \$i...\${RESET_TEXT}"
    gnome-terminal --title="RL-Swarm GPU \$i" -- bash -c "cd \$GPU_DIR && ./start_gpu\$i.sh; exec bash"
    sleep 5  # 给每个实例一些启动时间
done

echo -e "\${GREEN_TEXT}>> 所有GPU实例已启动!\${RESET_TEXT}"
EOF

chmod +x "$WORK_DIR/start_all_gpus.sh"

echo_green ">> 配置完成!"
echo_blue ">> 要一键启动所有GPU实例，请运行: $WORK_DIR/start_all_gpus.sh"
echo_blue ">> 注意: 您需要为每个实例使用相同的电子邮件登录，以便它们链接到同一个账户"
echo_blue ">> 登录端口将从3000开始，每个GPU递增1 (GPU 0: 3000, GPU 1: 3001, ...)"

if [ "$CONNECT_TO_TESTNET" = true ]; then
    # Run modal_login server.
    echo "Please login to create an Ethereum Server Wallet"
    cd modal-login
    # Check if the yarn command exists; if not, install Yarn.

    # Node.js + NVM setup
    if ! command -v node > /dev/null 2>&1; then
        echo "Node.js not found. Installing NVM and latest Node.js..."
        export NVM_DIR="$HOME/.nvm"
        if [ ! -d "$NVM_DIR" ]; then
            curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
        fi
        [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
        [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
        nvm install node
    else
        echo "Node.js is already installed: $(node -v)"
    fi

    if ! command -v yarn > /dev/null 2>&1; then
        # Detect Ubuntu (including WSL Ubuntu) and install Yarn accordingly
        if grep -qi "ubuntu" /etc/os-release 2> /dev/null || uname -r | grep -qi "microsoft"; then
            echo "Detected Ubuntu or WSL Ubuntu. Installing Yarn via apt..."
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
            sudo apt update && sudo apt install -y yarn
        else
            echo "Yarn not found. Installing Yarn globally with npm (no profile edits)…"
            # This lands in $NVM_DIR/versions/node/<ver>/bin which is already on PATH
            npm install -g --silent yarn
        fi
    fi
    yarn install
    yarn dev > /dev/null 2>&1 & # Run in background and suppress output

    SERVER_PID=$!  # Store the process ID
    echo "Started server process: $SERVER_PID"
    sleep 5

    # Try to open the URL in the default browser
    if open http://localhost:3000 2> /dev/null; then
        echo_green ">> Successfully opened http://localhost:3000 in your default browser."
    else
        echo ">> Failed to open http://localhost:3000. Please open it manually."
    fi

    cd ..

    echo_green ">> Waiting for modal userData.json to be created..."
    while [ ! -f "modal-login/temp-data/userData.json" ]; do
        sleep 5  # Wait for 5 seconds before checking again
    done
    echo "Found userData.json. Proceeding..."

    ORG_ID=$(awk 'BEGIN { FS = "\"" } !/^[ \t]*[{}]/ { print $(NF - 1); exit }' modal-login/temp-data/userData.json)
    echo "Your ORG_ID is set to: $ORG_ID"

    # Wait until the API key is activated by the client
    echo "Waiting for API key to become activated..."
    while true; do
        STATUS=$(curl -s "http://localhost:3000/api/get-api-key-status?orgId=$ORG_ID")
        if [[ "$STATUS" == "activated" ]]; then
            echo "API key is activated! Proceeding..."
            break
        else
            echo "Waiting for API key to be activated..."
            sleep 5
        fi
    done

    ENV_FILE="$ROOT"/modal-login/.env
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS version
        sed -i '' "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    else
        # Linux version
        sed -i "3s/.*/SMART_CONTRACT_ADDRESS=$SWARM_CONTRACT/" "$ENV_FILE"
    fi
fi

echo_green ">> Getting requirements..."

pip install --upgrade pip
if [ -n "$CPU_ONLY" ] || ! command -v nvidia-smi &> /dev/null; then
    # CPU-only mode or no NVIDIA GPU found
    pip install -r "$ROOT"/requirements-cpu.txt
    CONFIG_PATH="$ROOT/hivemind_exp/configs/mac/grpo-qwen-2.5-0.5b-deepseek-r1.yaml" # TODO: Fix naming.
    GAME="gsm8k"
else
    # NVIDIA GPU found
    pip install -r "$ROOT"/requirements-gpu.txt
    pip install flash-attn --no-build-isolation

    case "$PARAM_B" in
        32 | 72) CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-bnb-4bit-deepseek-r1.yaml" && break ;;
        0.5 | 1.5 | 7) CONFIG_PATH="$ROOT/hivemind_exp/configs/gpu/grpo-qwen-2.5-${PARAM_B}b-deepseek-r1.yaml" && break ;;
        *)  echo ">>> Please answer in [0.5, 1.5, 7, 32, 72]." ;;
    esac
    if [ "$USE_BIG_SWARM" = true ]; then
        GAME="dapo"
    else
        GAME="gsm8k"
    fi
fi

echo_green ">> Done!"

echo_green ">> Good luck in the swarm!"
echo_blue ">> Post about rl-swarm on X/twitter! --> https://tinyurl.com/swarmtweet"
echo_blue ">> And remember to star the repo on GitHub! --> https://github.com/gensyn-ai/rl-swarm"

if [ -n "$ORG_ID" ]; then
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --modal_org_id "$ORG_ID" \
        --contract_address "$SWARM_CONTRACT" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
else
    python -m hivemind_exp.gsm8k.train_single_gpu \
        --hf_token "$HUGGINGFACE_ACCESS_TOKEN" \
        --identity_path "$IDENTITY_PATH" \
        --public_maddr "$PUB_MULTI_ADDRS" \
        --initial_peers "$PEER_MULTI_ADDRS" \
        --host_maddr "$HOST_MULTI_ADDRS" \
        --config "$CONFIG_PATH" \
        --game "$GAME"
fi

wait  # Keep script running until Ctrl+C
