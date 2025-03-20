# nick_visit

## map the shared drive with data (one time)
```
curl https://jetstream2.exosphere.app/exosphere/assets/scripts/mount_ceph.py | sudo python3 - mount \
  --access-rule-name="AntsThings-rw" \
  --access-rule-key="AQCla8JnBX8uHRAABBGeIU/rLGcEE6feaY3u1w==" \
  --share-path="149.165.158.38:6789,149.165.158.22:6789,149.165.158.54:6789,149.165.158.70:6789,149.165.158.86:6789:/volumes/_nogroup/a908c299-c866-4920-b921-d68e8d36e068/c92aedad-932e-412d-a99f-551e402a98a7" \
  --share-name="AntsThings"
```
Will showup as **/media/share/AntsThings/** in your file browser. Remember when you drag and drop files to the browser window (or use SFTP to transfer) they are saved to **/media/volume/MyData/**, which is your personal storage. if you want to share them with other, copy them to the appropriate place in AntsThings share.

## setting the antspynet python environment
```
cd /media/volume/MyData
python3 -m venv tf
source tf/bin/activate
python3 -m pip install 'tensorflow[and-cuda]'
# test tf. Should see GPU:0
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
git clone https://github.com/ANTsX/ANTsPyNet
cd ANTsPyNet
nano requirements.txt
# scroll down to find the tensorflow line and comment out, then save.
python3 -m pip install .
```
you should be set. 

This recipe works on the lab server as well. The only difference you may need to do is to run the `python3 -m pip install 'tensorflow[and-cuda]'` command last (i.e., let ANTsPyNet install an older version of the tf, and then overwrite that with the newest version. It should 2.19)


## launching R
from the terminal window:
```module load rstudio```
and then either `R` or `rstudio`
