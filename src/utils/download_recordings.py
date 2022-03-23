from webdav3.client import Client

options = {
 'webdav_hostname': "https://owncloud.hpi.de/remote.php/dav/files/tobias.fiedler/",
 'webdav_login':    "tobias.fiedler",
 'webdav_password': "JNTXU-VAXEH-OLVCF-MYOLW"
}
client = Client(options)
client.pull(remote_directory="/ML Prototype Recordings", local_directory="/dhc/groups/bp2021ba1/data/owncloud")