# Ceph S3

**Source:** https://nrp.ai/documentation/userdocs/storage/ceph-s3

# Ceph S3

The Nautilus Ceph storage cluster can be accessed via S3 protocol. It uses our own storage, which is free for our users and is not part of Amazon or any commercial cloud.

### Ceph filesystems data use

Credit: [Ceph data usage](https://observablehq.com/d/b9c19d9f7c57a186)

[S3 Ceph grafana dashboard](https://grafana.nrp-nautilus.io/d/WaJ_lohMk/ceph-s3)

## Access

You can get your **credentials** (key and secret) in the [user portal](/s3token/), `User->S3 Tokens` page for public Ceph pools. To get credentials for private ones, contact the admin managing the particular S3 storage.

## S3 regions settings

Use the appropriate endpoint URL for your S3 client or library.

| Pool | Inside endpoint | Outside endpoint |
| --- | --- | --- |
| *West pool (default)* | <http://rook-ceph-rgw-nautiluss3.rook> | <https://s3-west.nrp-nautilus.io> |
| *Central pool* | <http://rook-ceph-rgw-centrals3.rook-central> | <https://s3-central.nrp-nautilus.io> |
| *East pool* | <http://rook-ceph-rgw-easts3.rook-east> | <https://s3-east.nrp-nautilus.io> |
| *HaoSu pool* | <http://rook-ceph-rgw-haosu.rook-haosu> | <https://s3-haosu.nrp-nautilus.io> |
| *Tide pool* | <http://rook-ceph-rgw-tide.rook-tide> | <https://s3-tide.nrp-nautilus.io> |

Note that the inside endpoint is **http** (without SSL) and the outside endpoint is **https** (with SSL). You can use the outside endpoint within the kubernetes cluster but it will end up going through a load balancer. By using the inside endpoint it is possible for multiple parallel requests from one or many machines to hit multiple separate storage servers (called Object Storage Devices (OSD) in ceph) and therefore achieve very large training set bandwith.

## Using Rclone

The easiest way to access S3 is [Rclone](https://rclone.org/s3/).

Use these options:

*Storage*: Amazon S3 Compliant Storage Providers

*S3 provider*: Ceph Object Storage

*AWS Access Key ID, AWS Secret Access Key*: [ask in Matrix chat](#access)

*Endpoint*: [use the regions section](#s3-regions-settings)

## Using s3cmd

[s3cmd](https://s3tools.org/s3cmd) is an open-source tool for accessing S3.

To configure, create the `~/.s3cfg` file with contents if you’re accessing from outside of the cluster:

```
[default]

access_key = <your_key>

host_base = https://s3-west.nrp-nautilus.io

host_bucket = https://s3-west.nrp-nautilus.io

secret_key = <your_secret>

use_https = True
```

or this if accessing from inside:

```
[default]

access_key = <your_key>

host_base = http://rook-ceph-rgw-nautiluss3.rook

host_bucket = http://rook-ceph-rgw-nautiluss3.rook

secret_key = <your_secret>

use_https = False
```

Run `s3cmd ls` to see the available buckets.

### Uploading files

Upload files with the `s3cmd put FILE`

```
$ s3cmd put <FILE> s3://<BUCKET>/<DIR>
```

Or, to upload a file to be public, use the `-P` for public file:

```
$ s3cmd put -P <FILE> s3://<BUCKET>/<DIR>

Public URL of the object is: http://s3-west.nrp-nautilus.io/...
```

## Using the AWS S3 tool

### Credentials

First add your credentials to **~/.aws/credentials**.

If you are familiar with the AWS CLI you can create an additional profile preserving your AWS credentials by adding it to **~/.aws/credentials**:

```
[default]

aws_access_key_id=xxxx

aws_secret_access_key=yyyy

[profile prp]

aws_access_key_id=iiiiii

aws_secret_access_key=jjjjj
```

If you don’t use AWS then you can just add credentials to [default] and skip the [profile] selection.

We recommend to use **[awscli-plugin-endpoint](https://github.com/wbingli/awscli-plugin-endpoint)** to write endpoint url in **.aws/config**, instead of typing endpoint in the CLI repeatedly. Install the plugin with:

```
pip install awscli-plugin-endpoint
```

There are a few steps on the **[awscli-plugin-endpoint](https://github.com/wbingli/awscli-plugin-endpoint)** README.md to install this plugin. If you do not wish to add this plugin, add `--endpoint-url https://s3-west.nrp-nautilus.io` to all commands below.

Your .aws/config file should look like:

```
[profile prp]

s3api =

endpoint_url = https://s3-west.nrp-nautilus.io

s3 =

endpoint_url = https://s3-west.nrp-nautilus.io

[plugins]

endpoint = awscli_plugin_endpoint
```

### Using the AWS CLI

The AWS CLI (command line interface) has two modes of operation for S3, `aws s3` are used for basic file manipulations (copy, list, delete, move, etc), and `aws s3api` for creating/deleting buckets, manipulating permissions, etc.

You can specify the endpoint on the command line (example: `aws --endpoint https://s3-west.nrp-nautilus.io s3 ls s3://bucket-name/path`) or via the s3 endpoint plugin (which is sometimes hard to install).

1. Create a bucket:

   ```
   aws s3api create-bucket --bucket BUCKETNAME --profile prp
   ```
2. List objects in the bucket:

   ```
   aws s3api list-buckets --profile prp

   aws s3 ls --profile prp
   ```
3. Upload a file:

   ```
   aws s3 cp ~/hello.txt s3://BUCKETNAME/path --profile prp
   ```
4. Upload a file and make it publicly accessible:

   ```
   aws s3 cp ~/hello.txt s3://BUCKETNAME/path --profile prp --acl public-read
   ```

   You can how access this file via a browser as <https://s3-west.nrp-nautilus.io/my-bucket/hello.txt>
5. Download a file:

   ```
   aws s3 cp s3://BUCKETNAME/path/hello.txt hello.txt
   ```

### Known Issue: UploadPart Failures with Large Files

When using the **AWS CLI** with our Ceph-backed S3, uploads larger than ~80 MB may fail with errors during the `UploadPart` stage of multipart uploads.  
This is a **compatibility issue** between the AWS CLI and Ceph’s S3 implementation.

#### Workarounds

* Use a different client such as:
  + [s3cmd](https://s3tools.org/s3cmd)
  + [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
  + [rclone](https://rclone.org/s3/)

These tools do not exhibit the same problem.

* Or, adjust your AWS CLI multipart configuration to reduce failure points. Add this profile to `~/.aws/config`:

```
[profile ceph-s3-large-files]

output = json

s3 =

signature_version = s3v4

addressing_style = path

multipart_threshold = 1GB

multipart_chunksize = 256MB

max_bandwidth = 200MB/s

use_accelerate_endpoint = false

use_dualstack_endpoint = false
```

Then use the profile in your commands:

Terminal window

```
aws s3 cp largefile.bin s3://BUCKETNAME/path --profile ceph-s3-large-files
```

This configuration has been tested to work reliably with files larger than 1 GB.

### Give multiple users full access to the bucket

When multiple users need to access a bucket you can set those permissions with the bucket policy. You set the bucket policy using the aws s3api command:

```
aws s3api put-bucket-policy --bucket BUCKETNAME --policy file://policy.json
```

Create `policy.json` with the following text (replace USER# and BUCKETNAME with your own users and bucket name), this policy will give all users full control over the bucket, other more granular bucket policies are certainly supported as well:

```
{

"Version": "2012-10-17",

"Statement": [

{

"Effect": "Allow",

"Principal": {

"AWS": [

"arn:aws:iam:::user/USERA",

"arn:aws:iam:::user/USERB",

"arn:aws:iam:::user/USERC"

]

},

"Action": "s3:*",

"Resource": [

"arn:aws:s3:::BUCKETNAME",

"arn:aws:s3:::BUCKETNAME/*"

]

}

]

}
```

More detailed policy.json examples at: <https://docs.aws.amazon.com/cli/latest/reference/s3api/put-bucket-policy.html>

### Delete Data From Buckets

The fastest way to delete is to use `rclone`. The steps for configuring `rclone` are above.

#### Delete Data

Terminal window

```
rclone delete S3:bucket-name>/<data-path> --transfers 1000 --checkers 2000 --disable ListR --progress
```

* `--transfers 1000`: Number of parallel file transfers.
* `--checkers 2000`: Number of simultaneous checks.
* `--disable ListR`: **Speeds up deletion** by skipping recursive listing.
* `--progress`: Shows progress.

#### Delete Buckets (After Deleting All Data)

Terminal window

```
rclone rmdir S3:bucket-name> --progress
```

### Host a File for Public Download

S3 Region Settings First, determine which S3 pool and region you are using. Your endpoint should follow this format based on your region:

```
https://s3-REGION.nrp-nautilus.io
```

Replace `REGION` with the appropriate region code provided by your data center.

#### Step 1: Upload the File to Your Bucket

Use `s3cmd` to upload your file to the bucket:

Terminal window

```
s3cmd put file-to-upload.txt s3://BUCKETNAME/ --host=s3-REGION.nrp-nautilus.io --host-bucket=s3-REGION.nrp-nautilus.io
```

Replace:

* `file-to-upload.txt` with the file you want to upload.
* `BUCKETNAME` with your bucket name.
* `REGION` with your actual region.

#### Step 2: Set the Bucket Policy for Public Access

To allow public access to the objects inside the bucket, modify the bucket’s ACL (Access Control List) using `s3cmd`:

Terminal window

```
s3cmd setacl s3://BUCKETNAME --acl-public --host=s3-REGION.nrp-nautilus.io --host-bucket=s3-REGION.nrp-nautilus.io
```

This command will make the entire bucket publicly accessible. If you prefer to make specific objects public rather than the entire bucket, proceed to the next step.

#### Step 3: Set Public Access Control on a Specific Object

If you want only certain files to be publicly accessible (instead of the whole bucket), set the ACL for the specific file:

Terminal window

```
s3cmd setacl s3://BUCKETNAME/file-to-upload.txt --acl-public --host=s3-REGION.nrp-nautilus.io --host-bucket=s3-REGION.nrp-nautilus.io
```

#### Step 4: Verify Public Access

To verify the file is publicly accessible, you can access it through the URL. The format for accessing the file will follow this structure:

```
https://s3-REGION.nrp-nautilus.io/BUCKETNAME/file-to-upload.txt
```

Replace `REGION`, `BUCKETNAME`, and `file-to-upload.txt` with your actual region, bucket name, and file name.

### Example

Here’s an example where we upload `report.pdf` to the bucket and make it publicly accessible:

Terminal window

```
s3cmd put report.pdf s3://public-bucket/ --host=s3-west.nrp-nautilus.io --host-bucket=s3-west.nrp-nautilus.io

s3cmd setacl s3://public-bucket/report.pdf --acl-public --host=s3-west.nrp-nautilus.io --host-bucket=s3-west.nrp-nautilus.io
```

Public URL:

```
https://s3-west.nrp-nautilus.io/public-bucket/report.pdf
```

Now, you can share this URL to allow anyone to download the file.

## Using Cyberduck

[Cyberduck](https://cyberduck.io) is a free S3 client for Mac and Windows. It can be used to upload and download files to/from S3 buckets. To use Cyberduck with Ceph S3 endpoints you need to leverage “deprecated” path style requests. The simplest way to do this is to install the appropriate profile into Cyberduck referenced in the [Cyberduck profiles documentation](https://docs.cyberduck.io/protocols/profiles/), [S3 (Deprecated path style requests).cyberduckprofile](https://github.com/iterate-ch/profiles/blob/master/S3%20(Deprecated%20path%20style%20requests).cyberduckprofile).

Once you add the profile, you can connect to the S3 endpoint by entering the endpoint hostname in the “Server” field. If you enter it as a URL instead of a hostname, it will likely trigger the selection of a different and undesired connection profile. For example, to connect to the S3 endpoint the for the NRP project’s western region, you would enter `s3-west.nrp-nautilus.io` in the “Server” field. You can then enter your access key and secret key in the “Access Key ID” and “Secret Access Key” fields, respectively.

# S3 Cookbook

S3 from tensorflow

```
with smart_open.open('s3://bucket/myfile.mat', 'rb') as f:

# yield your samples from the f file in your tensorflow dataset as usual
```

Note that smart\_open supports both local and S3 files, so when you’re testing this on a local file, it’ll work as well as when you run it on the cluster and pass it in a file located on S3.

See this [TFRecord](https://docs.google.com/presentation/d/16kHNtQslt-yuJ3w8GIx-eEH6t_AvFeQOchqGRFpAD7U) presentation for details.

Using S3 in GitLab CIsummary>

In GitLab project go to `Settings`->`CI/CD`, open the `Variables` tab, and add the variables holding your S3 credentials: `ACCESS_KEY_ID` and `SECRET_ACCESS_KEY`. Choose `protect variable` and `mask variable`.

Your `.gitlab-ci.yml` file can look like:

```
build:

image: ubuntu

before_script:

- apt-get update && apt-get install -y curl unzip

- curl https://rclone.org/install.sh | bash

stage: build

script:

- rclone config create nautilus-s3 s3 endpoint https://s3-west.nrp-nautilus.io provider Ceph access_key_id $ACCESS_KEY_ID secret_access_key $SECRET_ACCESS_KEY

- rclone ls "nautilus-s3:"
```

Creating a new bucket in S3

* Create a new bucket (change profile to match what is in `~/.aws/credentials`, and endpoint to the appropriate endpoint (Ceph/S3/West is used in this example):

```
aws --endpoint https://s3-west.nrp-nautilus.io s3api create-bucket --bucket my-bucket-name --profile prp
```

### Migrating from different S3-compatible storage

Consult the <https://github.com/clyso/chorus> project.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/ceph-s3.md)

[Previous  
Ceph FS / RBD](/documentation/userdocs/storage/ceph)  [Next  
CVMFS](/documentation/userdocs/storage/cvmfs)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.