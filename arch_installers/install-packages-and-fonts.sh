#!/bin/bash

# Update the package database
sudo pacman -Syu
sudo pacman -S --needed waybar wofi swaylock btop kitty swappy zed thunar firefox zed starship base-devel hyprpaper pulseaudio github-cli zip unzip swaylock xorg-xrandr brightnessctl less hyprlock # basic pkgs 
sudo pacman -S --needed coreutils # cutils


echo "All necessary packages and fonts have been installed."
