curl -fsSL https://pyenv.run | bash
pyenv install 3.12
pyenv global 3.12.13 # Easiest for our development in 3.12, set as global

# Set up direnv for easy venv loading/unloading
sudo pacman -S direnv
