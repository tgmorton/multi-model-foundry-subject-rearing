# Managing JWT Credentials

**Source:** https://nrp.ai/documentation/userdocs/storage/jwt-credential

# Managing JWT Credentials

[JSON Web Tokens (JWT)](https://jwt.io) are used to authenticate access to storage in the [Open Science Data Federation (OSDF)](https://osdf.osg-htc.org). JWTs are used to authenticate users and authorize access to resources, including storage in the cluster. Nautilus is able to manage your JWTs for you, refreshing and resigning tokens as needed using the OSG’s [token operator](https://github.com/osg-htc/token-manager).

## Requirements

In order to use the token operator, you will need:

1. A private key to sign the JWT
2. The location of the public key, usually on a website (this is the issuer)
3. The keyid of the public key
4. The algorithm used to sign the JWT
5. Scopes that should be included in the JWT, such as “storage.write:/”, “storage.read:/”, etc.

All of the above information can be provided by a representative of the OSDF.

## Creating a JWT

To create a JWT, you will need to create a “Custom Resource” in your namespace. This resource will contain the information needed to create the JWT. The token operator will then create the JWT and store it in a secret in your namespace. Below is an example of a Custom Resource that creates a JWT:

my-jwt.yaml

```
apiVersion: tokens.osg-htc.org/v1

kind: JWT

metadata:

name: my-jwt

spec:

issuer: https://issuer.example.com/

keyId: my-jwt-key

data:

scope: "storage.read:/ storage.create:/"

sub: john.doe

aud:

- "https://wlcg.cern.ch/jwt/v1/any"

wlcg.ver: "1.0"

expiryTime:

days: 1

resignBefore:

hours: 5

algorithm: ES256

key:

value: |

-----BEGIN EC PRIVATE KEY-----

MHcCAQEEIIphQalRpd3lclrnNmbR8df1/iljebEgI/CLxsmfd4GYoAoGCCqGSM49

AwEHoUQDQgAEdN/1YF8Q1BGJdmL9zWDMi5D+2Nfc6iAAXXFvA88HPElN+eOxHy0m

D1ygqiC82+ZMBTqt9l5dn6JFpd2AawPi7A==

-----END EC PRIVATE KEY-----
```

In the above example:

* The resulting JWT will be written into the secret `my-jwt` in the same namespace, with the key `token`.
* The JWT will be signed with the private key provided in the `key` field.
* The JWT will be valid for 1 day and will be resigned 5 hours before it expires.
* The JWT will have the claims: `scope`, and `sub`, and `ver`.
* The JWT will be signed using the `ES256` algorithm.

## Mounting the credential

Once the JWT is created, you can mount it into your pod as a volume. The token operator will automatically refresh and resign the JWT as needed. Below is an example of a pod that mounts the JWT as a volume.

my-deployment.yaml

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: token-test

spec:

replicas: 1

strategy:

type: Recreate

selector:

matchLabels:

app: token-test

template:

metadata:

labels:

app: token-test

spec:

containers:

- name: test-container

image: busybox

command: ["sleep", "3600"]

env:

- name: BEARER_TOKEN_FILE

value: /var/run/secrets/jwt/token

volumeMounts:

- mountPath: /var/run/secrets/jwt

name: my-jwt

resources:

requests:

memory: "128Mi"

cpu: "500m"

limits:

memory: "256Mi"

cpu: "1"

volumes:

- name: my-jwt

secret:

secretName: my-jwt
```

In the above example:

* The token is mounted into the pod and available at `/var/run/secrets/jwt/token`.
* The environment variable `BEARER_TOKEN_FILE` is set to the path of the token, which pelican will use to authenticate with the storage system.
* The token will be refreshed and resigned as needed by the token operator.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/jwt-credential.md)

[Previous  
Purging](/documentation/userdocs/storage/purging)  [Next  
Building in GitLab](/documentation/userdocs/development/gitlab)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.