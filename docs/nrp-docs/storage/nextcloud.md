# Nextcloud

**Source:** https://nrp.ai/documentation/userdocs/storage/nextcloud

# Nextcloud

We provide access to the [Nextcloud](https://nextcloud.com/) [instance](https://nextcloud.nrp-nautilus.io) running in our cluster and using our CephFS storage. It’s similar to other file sharing systems ([Dropbox](https://www.dropbox.com), [Google Drive](https://www.google.com/drive/), etc.) and can be used to get data in the cluster, temporarily stage the results, share data and so on.

**After registration, account will be initially in disabled state.** You must request account activation to the [Matrix](/contact) Support chat, including your email associated with Nextcloud.

If you’re planning to use Nextcloud for large datasets, please contact us first with the usage plan. Note that large files may take a long time to show up after the upload because the uploaded files are chunked and then reassembled. This is a similar constraint to our Ceph S3 storage.

## Using Nextcloud from shell

To access your Nextcloud storage from shell (or Jupyter), you can use the [rclone](https://rclone.org/) tool. It’s already installed in our [Jupyterlab](https://jupyterhub-west.nrp-nautilus.io) service. While the web interface is also possible to use, rclone with WebDAV may be more reliable for many or large files.

For updated information about accessing NextCloud with WebDAV, always refer to [Nextcloud’s official documentation.](https://docs.nextcloud.com/server/latest/user_manual/en/files/access_webdav.html)

To get access, create new rclone config:

Terminal window

```
jovyan@jupyter:~$ rclone config

2019/04/23 17:05:08 NOTICE: Config file "/home/jovyan/.config/rclone/rclone.conf" not found - using defaults

No remotes found - make a new one

n) New remote

s) Set configuration password

q) Quit config

n/s/q> n

name> nextcloud

Type of storage to configure.

Enter a string value. Press Enter for the default ("").

Choose a number from below, or type in your own value

1 / A stackable unification remote, which can appear to merge the contents of several remotes

&#92; "union"

2 / Alias for a existing remote

&#92; "alias"

3 / Amazon Drive

&#92; "amazon cloud drive"

4 / Amazon S3 Compliant Storage Provider (AWS, Alibaba, Ceph, Digital Ocean, Dreamhost, IBM COS, Minio, etc)

&#92; "s3"

5 / Backblaze B2

&#92; "b2"

6 / Box

&#92; "box"

7 / Cache a remote

&#92; "cache"

8 / Dropbox

&#92; "dropbox"

9 / Encrypt/Decrypt a remote

&#92; "crypt"

10 / FTP Connection

&#92; "ftp"

11 / Google Cloud Storage (this is not Google Drive)

&#92; "google cloud storage"

12 / Google Drive

&#92; "drive"

13 / Hubic

&#92; "hubic"

14 / JottaCloud

&#92; "jottacloud"

15 / Koofr

&#92; "koofr"

16 / Local Disk

&#92; "local"

17 / Mega

&#92; "mega"

18 / Microsoft Azure Blob Storage

&#92; "azureblob"

19 / Microsoft OneDrive

&#92; "onedrive"

20 / OpenDrive

&#92; "opendrive"

21 / Openstack Swift (Rackspace Cloud Files, Memset Memstore, OVH)

&#92; "swift"

22 / Pcloud

&#92; "pcloud"

23 / QingCloud Object Storage

&#92; "qingstor"

24 / SSH/SFTP Connection

&#92; "sftp"

25 / Webdav

&#92; "webdav"

26 / Yandex Disk

&#92; "yandex"

27 / http Connection

&#92; "http"

Storage> 25

See help for webdav backend at: https://rclone.org/webdav/

URL of http host to connect to

Enter a string value. Press Enter for the default ("").

Choose a number from below, or type in your own value

1 / Connect to example.com

&#92; "https://example.com"

url> https://nextcloud.nrp-nautilus.io/remoremote.php/dav/files/USERNAME/

Name of the Webdav site/service/software you are using

Enter a string value. Press Enter for the default ("").

Choose a number from below, or type in your own value

1 / Nextcloud

&#92; "nextcloud"

2 / Owncloud

&#92; "owncloud"

3 / Sharepoint

&#92; "sharepoint"

4 / Other site/service or software

&#92; "other"

vendor> 1

User name

Enter a string value. Press Enter for the default ("").

user> YOUR NEXTCLOUD USERNAME

Password.

y) Yes type in my own password

g) Generate random password

n) No leave this optional password blank

y/g/n> y

Enter the password:

password: YOUR PASSWORD, or CREATE A TOKEN IN SETTINGS IF USING 2-FACTOR

Confirm the password:

password:

Bearer token instead of user/pass (eg a Macaroon)

Enter a string value. Press Enter for the default ("").

bearer_token>

Remote config

--------------------

[nextcloud]

type = webdav

url = https://nextcloud.nrp-nautilus.io/remoremote.php/dav/files/USERNAME/

vendor = nextcloud

user = {YOURUSER}

pass =  ENCRYPTED

--------------------

y) Yes this is OK

e) Edit this remote

d) Delete this remote

y/e/d> y

Current remotes:

Name                 Type

====                 ====

nextcloud            webdav

e) Edit existing remote

n) New remote

d) Delete remote

r) Rename remote

c) Copy remote

s) Set configuration password

q) Quit config

e/n/d/r/c/s/q> q
```

Then copy in your data:

Terminal window

```
rclone copy -P nextcloud:/Downloads .
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/nextcloud.md)

[Previous  
Linstor](/documentation/userdocs/storage/linstor)  [Next  
Syncthing](/documentation/userdocs/storage/syncthing)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.