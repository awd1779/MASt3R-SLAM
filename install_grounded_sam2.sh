#!/bin/bash
# Installation script for Grounded-SAM-2

echo "Installing Grounded-SAM-2 for MAST3R-SLAM..."

# Check if already exists
if [ -d "$HOME/Grounded-SAM-2" ]; then
    echo "Grounded-SAM-2 already exists at ~/Grounded-SAM-2"
    echo "Reinstalling grounding_dino..."
    cd $HOME/Grounded-SAM-2/grounding_dino
    pip install -e .
else
    # Clone the repository
    echo "Cloning Grounded-SAM-2..."
    cd $HOME
    git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
    
    # Install grounding_dino
    echo "Installing grounding_dino..."
    cd Grounded-SAM-2/grounding_dino
    pip install -e .
fi

# Install SAM2 if not already installed
echo -e "\nChecking SAM2 installation..."
python -c "import sam2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing SAM2..."
    pip install segment-anything-2
else
    echo "SAM2 already installed"
fi

# Install additional dependencies
echo -e "\nInstalling additional dependencies..."
pip install supervision transformers

# Create models directory
echo -e "\nCreating models directory..."
mkdir -p $HOME/models/GroundingDINO/weights
mkdir -p $HOME/models/segment-anything-2/checkpoints

echo -e "\n✅ Installation complete!"
echo -e "\nNext steps:"
echo "1. Download model weights:"
echo "   - SAM2: https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
echo "   - Grounding DINO: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
echo "2. Place them in:"
echo "   - ~/models/segment-anything-2/checkpoints/"
echo "   - ~/models/GroundingDINO/weights/"
echo "3. Run: python check_grounded_sam2.py"