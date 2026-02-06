# Exposing HTTP

**Source:** https://nrp.ai/documentation/userdocs/running/ingress

# Exposing HTTP

While pods are not accessible from outside the cluster, you can expose the http services provided by pods by using the [Ingress](https://kubernetes.io/docs/concepts/services-networking/ingress/) controllers. In general we don’t allow exposing non-http applications via TCP ports, but if you really need to do that, contact us on [Matrix](/contact).

For a complete example see the ([Tutorial](/documentation/userdocs/tutorial/basic2/))

Refer to the [haproxy ingress documentation](https://haproxy-ingress.github.io/docs/configuration/keys/) to set up additional config.

To expose a port in your pod, you first need to create a service for it. For the pod

```
apiVersion: v1

kind: Pod

metadata:

name: my-pod

labels:

k8s-app: test-http

spec:

containers:

- name: mypod

image: nginxdemos/hello:plain-text
```

The service might look like:

```
apiVersion: v1

kind: Service

metadata:

labels:

k8s-app: test-svc

name: test-svc

spec:

ports:

- port: 8080

protocol: TCP

targetPort: 80

selector:

k8s-app: test-http

type: ClusterIP
```

Where `spec.selector.<label>` should match the label of the target pod (`k8s-app: test-http`), and `targetPort` should match the port you want to expose. You can test the pod/service by creating a tunnel (`kubectl port-forward service/test-svc 8080:8080`) and querying service (`curl http://localhost:8080/`).

After that you can create the Ingress object:

```
apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

name: test-ingress

spec:

ingressClassName: haproxy

rules:

- host: test-service.nrp-nautilus.io

http:

paths:

- path: /

pathType: Prefix

backend:

service:

name: test-svc

port:

number: 8080

tls:

- hosts:

- test-service.nrp-nautilus.io
```

You can choose the host subdomain to be whatever you want (`<whatever>.nrp-nautilus.io`). This is enough to have your pod served under `test-service.nrp-nautilus.io`. You can test this example via curl (`curl https://test-service.nrp-nautilus.io`)

## Using my own domain name

If you need to use your own domain, you would have to also provide a valid certificate in a secret in your namespace:

```
apiVersion: v1

kind: Secret

metadata:

name: my-own-hostname-tls

type: kubernetes.io/tls

data:

ca.crt: <optional base64-encoded ca>

tls.crt: <base64-encoded crt>

tls.key: <base64-encoded key>
```

and add a section to the Ingress:

```
apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

name: test-ingress

spec:

ingressClassName: haproxy

rules:

- host: my-own-hostname.com

http:

paths:

- backend:

service:

name: test-svc

port:

number: 8080

path: /

pathType: Prefix

tls:

- hosts:

- my-own-hostname.com

secretName: my-own-hostname-tls
```

Create the CNAME DNS record for your domain pointing to `nrp-nautilus.io` (for geo-balanced multi-region DNS record) or `east.nrp-nautilus.io` (just the eastern region).

## Auto renewing the certificate

The [Cert Manager](https://cert-manager.io) installed in our cluster supports auto retrieving and updating [ACME Let’s Encrypt](https://letsencrypt.org/docs/client-options/) certificates. To set up the HTTP challenge, create an issuer:

```
apiVersion: cert-manager.io/v1

kind: Issuer

metadata:

name: letsencrypt

spec:

acme:

email: <your_email>

preferredChain: ""

privateKeySecretRef:

name: issuer-account-key

server: https://acme-v02.api.letsencrypt.org/directory

solvers:

- http01:

ingress:

class: haproxy

ingressTemplate:

metadata:

annotations:

ingress.kubernetes.io/ssl-redirect: "false"

serviceType: ClusterIP
```

And then request the new certificate:

```
apiVersion: cert-manager.io/v1

kind: Certificate

metadata:

annotations:

name: my-own-hostname-cert

spec:

commonName: my-own-hostname.com

dnsNames:

- my-own-hostname.com

issuerRef:

kind: Issuer

name: letsencrypt

secretName: my-own-hostname-tls
```

Check the status of your certificate:

`kubectl get certificate -o wide`

If `STATUS` is not `Certificate is up to date and has not expired`, use `kubectl describe certificate ...` and `kubectl describe order ...` to debug the problem.

Once the certificate was issued, refer to the previous section to use one in your Ingress.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/ingress.md)

[Previous  
Client scripts](/documentation/userdocs/running/scripts)  [Next  
GatewayAPI](/documentation/userdocs/running/gateway)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.