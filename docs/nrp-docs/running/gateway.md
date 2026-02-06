# Gateway API

**Source:** https://nrp.ai/documentation/userdocs/running/gateway

# Gateway API

Danger

This is a temporary solution until we completely migrate the top domain from Ingresses to Gateway API. Once the migration is done, any service in the top level domain will be possible to create as GRPC.

[Gateway API](https://gateway-api.sigs.k8s.io/) is a Kubernetes API for defining and managing network gateways and their associated routing rules. It provides a declarative way to define how traffic should be routed to services within a Kubernetes cluster, making it easier to manage complex network topologies and improve the scalability and reliability of applications. It obsoletes [Ingress](/documentation/userdocs/running/ingress/).

# Exposing HTTP services

Note

The service will be exposed on ports 50080 and 50443.

Create [HTTPRoute](https://gateway-api.sigs.k8s.io/api-types/httproute/) to expose HTTP services.

```
apiVersion: gateway.networking.k8s.io/v1

kind: HTTPRoute

metadata:

name: my-service-route

spec:

hostnames:

- my-service-name.nrp-nautilus.io

parentRefs:

- group: gateway.networking.k8s.io

kind: Gateway

name: ingress

namespace: haproxy

sectionName: https

rules:

- backendRefs:

- group: ""

kind: Service

name: service-name

port: 8443

weight: 1

matches:

- path:

type: PathPrefix

value: /
```

Replace:

* `.metadata.name` - any name unique within the namespace
* `.spec.hostnames[0]` - the hostname for the HTTP service. Choose any unique one within `.nrp-nautilus.io`.
* `.spec.rules[0].backendRefs[0].name` - the name of the service in the namespace where it is deployed.
* `.spec.rules[0].backendRefs[0].port` - the port of the service in the namespace where it is deployed.

Use the [official docs](https://gateway-api.sigs.k8s.io/api-types/httproute/) / [reference spec](https://gateway-api.sigs.k8s.io/reference/spec/#httproute) for other fields.

# Exposing GRPC services

Note

The service will be exposed on port 50051.

Create [GRPCRoute](https://gateway-api.sigs.k8s.io/api-types/grpcroute/) to expose GRPC services.

```
apiVersion: gateway.networking.k8s.io/v1

kind: GRPCRoute

metadata:

name: my-service-route

spec:

hostnames:

- my-service-name.nrp-nautilus.io

parentRefs:

- group: gateway.networking.k8s.io

kind: Gateway

name: ingress

namespace: haproxy

sectionName: grpc

rules:

- backendRefs:

- kind: Service

name: service-name

port: 50051

weight: 1
```

Replace:

* `.metadata.name` - any name unique within the namespace
* `.spec.hostnames[0]` - the hostname for the GRPC service. Choose any unique one within `.nrp-nautilus.io`.
* `.spec.rules[0].backendRefs[0].name` - the name of the service in the namespace where it is deployed.
* `.spec.rules[0].backendRefs[0].port` - the port of the service in the namespace where it is deployed.

Use the [official docs](https://gateway-api.sigs.k8s.io/api-types/grpcroute/) / [reference spec](https://gateway-api.sigs.k8s.io/reference/spec/#grpcroute) for other fields.

# Testing

## Testing GRPC

Caution

Please delete the deployed objects after testing.

(Taken from [envoy gateway docs](https://gateway.envoyproxy.io/docs/tasks/traffic/grpc-routing/))

Deploy the test app:

```
apiVersion: apps/v1

kind: Deployment

metadata:

labels:

app: yages

example: grpc-routing

name: yages

spec:

selector:

matchLabels:

app: yages

replicas: 1

template:

metadata:

labels:

app: yages

spec:

containers:

- name: grpcsrv

image: ghcr.io/projectcontour/yages:v0.1.0

resources:

limits:

cpu: 100m

memory: 100Mi

requests:

cpu: 50m

memory: 50Mi

ports:

- containerPort: 9000

protocol: TCP

---

apiVersion: v1

kind: Service

metadata:

labels:

app: yages

example: grpc-routing

name: yages

spec:

type: ClusterIP

ports:

- name: http

port: 9000

protocol: TCP

targetPort: 9000

selector:

app: yages

---

apiVersion: gateway.networking.k8s.io/v1

kind: GRPCRoute

metadata:

name: yages

labels:

example: grpc-routing

spec:

hostnames:

- test-service.nrp-nautilus.io

parentRefs:

- group: gateway.networking.k8s.io

kind: Gateway

name: ingress

namespace: haproxy

sectionName: grpc

rules:

- backendRefs:

- kind: Service

name: yages

port: 9000

weight: 1
```

Check the status of GRPCRoute:

Terminal window

```
kubectl get grpcroutes --selector=example=grpc-routing -o yaml
```

The status for the GRPCRoute should surface “Accepted=True” and a `parentRef` that references the example Gateway. The `yages` route matches any traffic for “test-service.nrp-nautilus.io” and forwards it to the “yages” Service.

Test GRPC routing to the yages backend using the grpcurl command.

Terminal window

```
grpcurl test-service.nrp-nautilus.io:50051 yages.Echo/Ping
```

You should see the below response

```
{

"text": "pong"

}
```

Envoy Gateway also supports gRPC-Web requests for this configuration. The below curl command can be used to send a grpc-Web request with over HTTP/2. You should receive the same response seen in the previous command.

The data in the body `AAAAAAA=` is a base64 encoded representation of an empty message (data length 0) that the Ping RPC accepts.

Terminal window

```
curl -s https://test-service.nrp-nautilus.io:50051/yages.Echo/Ping -H 'Content-Type: application/grpc-web-text' -H 'Accept: application/grpc-web-text' -XPOST -d'AAAAAAA=' | base64 -d
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/gateway.md)

[Previous  
Exposing HTTP](/documentation/userdocs/running/ingress)  [Next  
Special use](/documentation/userdocs/running/special)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.