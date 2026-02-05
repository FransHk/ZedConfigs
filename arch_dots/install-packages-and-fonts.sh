#!/bin/bash

# Update the package database
sudo pacman -Syu
sudo pacman -S --needed waybar wofi swaylock btop kitty swappy zed thunar firefox zed starship base-devel hyprpaper pulseaudio github-cli # basic pkgs
sudo pacman -S --needed coreutils # cutils
sudo pacman -S --needed ttf-dejavu ttf-liberation ttf-ubuntu-font-family ttf-roboto ttf-nerd-fonts-symbols  # fonts

echo "All necessary packages and fonts have been installed."
