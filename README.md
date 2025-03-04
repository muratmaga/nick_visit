# nick_visit

## map the shared drive with data (one time)
```
curl https://jetstream2.exosphere.app/exosphere/assets/scripts/mount_ceph.py | sudo python3 - mount \
  --access-rule-name="AntsThings-rw" \
  --access-rule-key="AQCla8JnBX8uHRAABBGeIU/rLGcEE6feaY3u1w==" \
  --share-path="149.165.158.38:6789,149.165.158.22:6789,149.165.158.54:6789,149.165.158.70:6789,149.165.158.86:6789:/volumes/_nogroup/a908c299-c866-4920-b921-d68e8d36e068/c92aedad-932e-412d-a99f-551e402a98a7" \
  --share-name="AntsThings"
```
Will showup as **/media/share/AntsThings/** in your file browser. Remember when you drag and drop files to the browser window (or use SFTP to transfer) they are saved to /media/volume/MyData, which your personal storage. if you want to share them with other, copy them to the appropriate place in AntsThings share.
