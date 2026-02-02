#!/bin/bash

# Update the package database
sudo pacman -Syu

# Install Waybar and its dependencies
sudo pacman -S waybar wofi swaylock btop kitty asusctl brightnessctl pamixer pavucontrol wlogout python wlr-workspaces blueman nm-applet polkit-gnome dbus xdg-portal-hyprland hyprpaper grim swappy zed thunar firefox

# Install additional dependencies for scripts and other tools
sudo pacman -S coreutils

# Install common fonts
sudo pacman -S ttf-dejavu ttf-liberation ttf-ubuntu-font-family ttf-roboto ttf-nerd-fonts-symbols ttf-nerd-fonts-complete

echo "All necessary packages and fonts have been installed."
