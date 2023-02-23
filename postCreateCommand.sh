echo y| kill 7913 apt apt-get
echo y | rm /var/lib/apt/lists/lock
echo y | rm /var/cache/apt/archives/lock
echo y | rm /var/lib/dpkg/lock
echo y | dpkg --configure -a
echo y | apt update

echo y | /usr/bin/python -m pip install --upgrade pip  
echo y | /usr/bin/python -m pip install --root-user-action=ignore -r requirements.txt 
echo y | export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64 && cp -f ./resources/cws.py /usr/local/lib/python3.8/dist-packages/scaleogram 
echo y | add-apt-repository ppa:mc3man/trusty-media
echo y | apt-get dist-upgrade
echo y | apt-get update  
echo y | apt-get upgrade
echo y | apt install ffmpeg libgl1 libgl1-mesa-glx libgtk2.0-dev pkg-config python3-opencv 
echo y | apt-get install xvfb python-opengl 
echo y | apt-get install --reinstall libexpat1
echo y | apt-get remove --purge alsa-base pulseaudio
echo y | apt-get install alsa-base pulseaudio
echo y | apt-get -f install &&  apt-get -y autoremove &&  apt-get autoclean &&  apt-get clean &&  sync && echo 3 |  tee /proc/sys/vm/drop_caches
echo y | chown -R $USER:$USER $HOME/
echo y | apt install x11-apps
echo y | apt-get update -y
echo y | apt-get install -y libx11-dev
echo y | apt-get install --reinstall -y libexpat1
echo y | apt-get install -y python3-tk python3-seaborn
echo y | tensorboard --logdir=data/model/TensorBoard --port 8088