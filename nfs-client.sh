#!/bin/bash

# Installa i pacchetti necessari per il client NFS
sudo apt-get update
sudo apt-get install -y nfs-common

# Crea una directory dove montare il disco condiviso
sudo mkdir -p /mnt/nfs

# Monta il disco condiviso sulla directory creata
sudo mount 172.16.5.64:/mnt/nfs /mnt/nfs

# Aggiungi l'entry nel file fstab per montare il disco condiviso all'avvio
echo '172.16.5.64:/mnt/nfs /mnt/nfs nfs defaults 0 0' | sudo tee -a /etc/fstab