# Required for powerprofilesctl launching (Python GOBJECT)
python -m pip install --upgrade pip setuptools wheel
python -m pip install pygobject

sudo pacman -S power-profiles-daemon # The actual daemon for power profiles
sudo systemctl enable --now power-profiles-daemon.service

