# NRP-Managed vector database

**Source:** https://nrp.ai/documentation/userdocs/ai/vector-database

# NRP-Managed vector database

The NRP provides managed vector database [Milvus](https://milvus.io/docs/overview.md) as a service to be used in LLMs.

**You don’t need to install the operator, create the milvus instance, or create the database. All that is already done for you.**

Tip

Milvus GRPC endpoint:

**milvus.nrp-nautilus.io:50051**

To get access to Milvus:

* Start from getting your username and password. Navigate to the [Milvus password page](/milvus) and click the “Get milvus password” button. The link to secure page with your password will come to your email.
* [Make sure](/namespaces) you’re a member of a group with Milvus feature enabled. If you’re a namespace admin, you can create one. Otherwise ask your namespace admin to add you to one.

Note

Groups can only contain alphanumeric symbols and dashes. Milvus databases can only contain alphanumeric symbols and underscores. All dashes in your group name will be converted to underscores in milvus database name.

* You can access the Milvus databases according to the NRP group membership. Start [defining collections](https://milvus.io/docs/manage-collections.md) or [searching data](https://milvus.io/docs/single-vector-search.md).

### Milvus UI

Check [Attu quickstart instructions](https://milvus.io/docs/quickstart_with_attu.md) for accessing milvus from a standalone GUI app.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/ai/vector-database.mdx)

[Previous  
Cloud AI 100 Cards](/documentation/userdocs/ai/qaic)  [Next  
FABRIC Integration](/documentation/userdocs/networks/fabric)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.