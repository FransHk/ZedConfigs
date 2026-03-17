sudo pacman -S bluez bluez-utils
sudo systemctl enable --now bluetooth.service # Turn on bluetooth

# Next, we use: 
# bluetoothctl > launches bluetooth control
# power on 
# agent on
# default-agent
# scan on > will start printing MAC addresses
# Then, use the combination of pair -> trust -> connect on the correct mac address


