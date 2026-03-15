wget https://github.com/ryanoasis/nerd-fonts/releases/download/v3.4.0/JetBrainsMono.zip
mkdir tmp_fonts
mv JetBrainsMono.zip tmp_fonts
cd tmp_fonts
unzip JetBrainsMono.zip
sudo mkdir /usr/share/fonts/TTF/
sudo cp *.ttf /usr/share/fonts/TTF/


# Also, install emojis (e.g. starship icon for Python, Git, etc)
yay -S noto-fonts-emoji

