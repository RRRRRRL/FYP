Reproduce the expierment in lab machine

PowerShell
wsl --install 

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3-pip python3-venv -y

# Create virtual environment
python3 -m venv triton_env
source triton_env/bin/activate

# Install PyTorch with CUDA support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Triton (Linux version)
pip3 install triton

# Install other dependencies
pip3 install numpy pandas matplotlib tabulate

py3 flashattention.py