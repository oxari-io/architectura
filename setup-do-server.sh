#!/bin/bash

set -e

read -p 'Github Username: ' uservar
read -p 'Github Personal Access Token: ' passvar
echo
echo Thank you $uservar

echo Installing dependencies from apt for anaconda
sudo apt update
sudo apt install build-essential -y
sudo apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y

echo Installing anaconda... 
export ANACONDA_INSTALLER_VERSION="2024.02-1"
curl -O https://repo.anaconda.com/archive/Anaconda3-$ANACONDA_INSTALLER_VERSION-Linux-x86_64.sh && bash Anaconda3-$ANACONDA_INSTALLER_VERSION-Linux-x86_64.sh
source ~/.bashrc

echo Create conda environment... 
conda create -n py310 python=3.10 -y
git clone https://$uservar:$passvar@github.com/oxari-io/architectura.git
cd architectura

echo Install poetry... 
conda install poetry -y


echo Setup architectura environment... 
poetry env use /root/anaconda3/envs/py310/bin/python
poetry export --without-hashes --format=requirements.txt > requirements.txt
poetry run pip install -r requirements.txt


echo DON'T FORGET TO SET THE .env FILE!!!!