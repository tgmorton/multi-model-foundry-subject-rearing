# NRP-Managed LLMs

**Source:** https://nrp.ai/documentation/userdocs/ai/llm-managed

# NRP-Managed LLMs

The NRP provides several hosted open-weights LLM for either API access, or use with our hosted chat interfaces.

[Chat with NRP LLMs](https://librechat.nrp-nautilus.io) Use the LibreChat interface to chat with NRP hosted LLMs

[Get an API token for NRP LLMs](#api-access-to-llms-via-envoy-gateway) Get an API token to programically interact with the LLMs or use the LLMs in other apps

## Chat Interfaces

### Librechat

If you are looking to chat with an LLM model similar to the interface provided by [ChatGPT](https://chatgpt.com/), we provide [LibreChat](https://librechat.nrp-nautilus.io), based on the [LibreChat](https://www.librechat.ai/) project. This is a simple chat interface for all of the NRP hosted models. You can use it to chat with the models, or to test out the models.

[Visit the LibreChat interface](https://librechat.nrp-nautilus.io)

On MacOS and Safari you can make it always available in Dock for quick access: having librechat open in Safari, click **File->Add to Dock**.

### Chatbox

You can install the standalone chatbox app or use the web interface version.

[Visit the Chatbox app web site](https://chatboxai.app)

Generate the config for it in the [LLM token generation page](/llmtoken). Copy the generated config to clipboard - it will already have your personal token. **Please always leave `Max Output Tokens` empty and only fill in `Context Window`.**

In chatbox app go to *Settings*->*Model Provider*, scroll down to the end of providers list and click *Import from clipboard*.

**Please always leave empty or do not specify `Max Output Tokens`/`max_tokens`/`max_output_tokens`.** If required, specify it to somewhere between 1/8 to 1/2 of the total context length. **Do NOT specify the full context length.**

## API Access to LLMs via Envoy gateway

To access our LLMs through the [Envoy AI Gateway](https://aigateway.envoyproxy.io), you need to be a [member of a group with LLM flag](/documentation/userdocs/start/hierarchy). Your membership info can be found on the [namespaces page](/namespaces).

Start from [creating a token](/llmtoken). You can use this token to query the OpenAI-compatible LLM endpoint:

Envoy AI API endpoint

<https://ellm.nrp-nautilus.io/v1>

with CURL or any OpenAI API compatible tool.

```
curl -H "Authorization: Bearer <your_token>" https://ellm.nrp-nautilus.io/v1/models
```

## Examples

### Python Code

To access the NRP LLMs, you can use the [OpenAI Python client](https://github.com/openai/openai-python). Below is an example of how to use the OpenAI Python client to access the NRP LLMs.

nrp-llm.py

```
import os

from openai import OpenAI

client = OpenAI(

# This is the default and can be omitted

api_key = os.environ.get("OPENAI_API_KEY"),

base_url = "https://ellm.nrp-nautilus.io/v1"

)

completion = client.chat.completions.create(

model="gemma3",

messages=[

{"role": "system", "content": "Talk like a pirate."},

{

"role": "user",

"content": "How do I check if a Python object is an instance of a class?",

},

],

)

print(completion.choices[0].message.content)
```

### Bash+Curl

```
curl -H "Authorization: Bearer <TOKEN>" https://ellm.nrp-nautilus.io/v1/models
```

```
curl -H "Authorization: Bearer <TOKEN>" -X POST "https://ellm.nrp-nautilus.io/v1/chat/completions" \

-H "Content-Type: application/json" \

-d '{

"model": "meta-llama/Llama-3.2-90B-Vision-Instruct",

"messages": [

{"role": "user", "content": "Hey!"}

]

}'
```

### Kimi CLI

<https://github.com/MoonshotAI/kimi-cli>

Open Configuration

Contents of `~/.kimi/config.json`:

```
{

"default_model": "kimi",

"models": {

"kimi": {

"provider": "nrp",

"model": "kimi",

"max_context_size": 262144,

"capabilities": ["thinking"]

}

},

"providers": {

"nrp": {

"type": "openai_legacy",

"base_url": "https://ellm.nrp-nautilus.io/v1",

"api_key": "<YOUR_API_KEY>"

}

},

"loop_control": {

"max_steps_per_run": 100,

"max_retries_per_step": 3

},

"services": {}

}
```

## Available Models

main - Model is **generally supported**. You can report issues with the service. However, if the model is outdated with no apparent usage purpose, it may be removed if there are no major group or user usage, or switched to a deprecated state. This is to provide our users with the best models within our limited allocation of GPUs.

batch - The LLM is recommended for batch querying and will provide enough performance for most types of queries under heavy load.

tool - The LLM has tools calling enabled.

multimodal - The LLM is multimodal.

evaluating - The LLM is added **for testing** and we’re evaluating it’s capabilities. The model may be unavailable sometimes, and configurations may be changed without notification.

deprecated - LLM is **deprecated** and is likely to go away soon. Please do not start using this model; this is only for existing user groups who have specific purposes of this model.

You can follow all updates and participate in the discussions within our [Matrix](https://nrp.ai/contact/#matrix) [Nautilus Artificial Intelligence/Machine Learning](https://element.nrp-nautilus.io/#/room/#ml:matrix.nrp-nautilus.io) ([NRP Matrix.to](https://matrix-to.nrp-nautilus.io/#/#ml:matrix.nrp-nautilus.io), [Matrix.to](https://matrix.to/#/#ml:matrix.nrp-nautilus.io)) channel. Suggestions and decisions for new models are also made here.

qwen3

main tool multimodal

**[Qwen/Qwen3-VL-235B-A22B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8)**

Multimodal (vision, video), 262,144 tokens, 235B parameters, Official FP8 quantization, [tool calling](https://docs.vllm.ai/en/stable/features/tool_calling.html#qwen-models), Claude/Gemini-level frontier multimodal performance

gpt-oss

evaluating batch tool

**[openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)**

131,072 tokens, [tool calling](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html), Native MXFP4 model weights, frontier agentic performance

kimi

evaluating tool

**[moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)**

262,144 tokens, [tool calling](https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2-Think.html), Native INT4 model weights, frontier general and agentic coding performance

glm-4.7

evaluating tool

**[QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix)**

202,752 tokens, [tool calling](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM.html), GPTQ mixed quantization, frontier agentic coding performance

minimax-m2

evaluating tool

**[MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1)**

262,144 tokens, [tool calling](https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html), Native FP8 model weights, frontier agentic coding performance

glm-v

main batch tool multimodal

**[zai-org/GLM-4.6V-FP8](https://huggingface.co/zai-org/GLM-4.6V-FP8)**

Multimodal (vision, video), 131,072 tokens, [tool calling](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM-V.html), Official FP8 quantization, Gemini Flash level multimodal performance

gemma3

main batch tool multimodal

**[google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)**

Multimodal (vision), 131,072 tokens, [tool calling](https://github.com/vllm-project/vllm/pull/17149)

embed-mistral

main

**[intfloat/e5-mistral-7b-instruct](https://huggingface.co/intfloat/e5-mistral-7b-instruct)**

Embeddings

gorilla

evaluating tool

**[gorilla-llm/gorilla-openfunctions-v2](https://huggingface.co/gorilla-llm/gorilla-openfunctions-v2)**

Function calling

olmo

evaluating

**[allenai/OLMo-2-0325-32B-Instruct](https://allenai.org/blog/olmo2-32B)**

Open source

llama3-sdsc

deprecated

**[meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)**

8 languages, 131,072 tokens, tool use

## How Models are Added and Removed

**Added**: New NRP-managed models are added by feedback and also assessments of various benchmarks and community response to the models by the administrator. We try to take into account quantitative benchmarks (such as <https://artificialanalysis.ai>), but the ultimate decision is based on other qualitative evidence (such as <https://www.reddit.com/r/LocalLLaMA/>) and discussions between administrators and users.

**Removed**: Moreover, we would also like to remove certain models that are deemed sufficiently obsolete, where for instance, smaller models may perform all-round better, or another model completely cleared the usage case in a better way compared to the obsolete model.

**Deprecated**: An exception is when research groups need these models for model reproducibility in various types of research. In that case, we **deprecate** these models first, and keep them up until such research concludes. If a model has been deprecated or pulled down, please reach out through the below **Nautilus Artificial Intelligence/Machine Learning** channel for us to track.

However, we still want to remove deprecated models as soon as possible, due to having limited GPU allocations specialized for deployed LLMs, and these GPUs should be diverted to more recent and better-performing models for the benefit of the whole NRP community. Administrators and researchers use these models for AI-assisted code development, and such models need to be rotated frequently as new and better models are released, vastly increasing individual productivity as newer models are incorporated.

Larger models that require a lot of GPUs are likely to be removed earlier in a more strict way if relative performance falls behind, but smaller, more efficient models or quantized models that do not require as many GPUs may be slightly more lenient in this criterion.

Such discussion is done in the [Nautilus Artificial Intelligence/Machine Learning](https://element.nrp-nautilus.io/#/room/#ml:matrix.nrp-nautilus.io) ([NRP Matrix.to](https://matrix-to.nrp-nautilus.io/#/#ml:matrix.nrp-nautilus.io), [Matrix.to](https://matrix.to/#/#ml:matrix.nrp-nautilus.io)) channel.

### Changelogs

Click to expand

#### **December 2025:**

Added/Changed:

* `glm-v` was changed to [zai-org/GLM-4.6V-FP8](https://huggingface.co/zai-org/GLM-4.6V-FP8), an upgraded model with larger context size.
* `glm-4.6` was renamed to `glm-4.7` and uses the [QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-4.7-GPTQ-Int4-Int8Mix) model.
* The `minimax-m2` model changed from [MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2) to [MiniMaxAI/MiniMax-M2.1](https://huggingface.co/MiniMaxAI/MiniMax-M2.1).

#### **November 2025:**

Added/Changed:

* `qwen3` ([Qwen/Qwen3-235B-A22B-Thinking-2507-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507-FP8)) has been changed to [Qwen/Qwen3-VL-235B-A22B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Thinking-FP8). Very similar characteristics such as number of parameters, context size, and benchmarks, but adds state-of-the-art vision and video multimodal capabilities.
* `kimi` ([moonshotai/Kimi-K2-Thinking](https://huggingface.co/moonshotai/Kimi-K2-Thinking)) is a widely popular programming LLM model and exhibits a similar level of model performance to Claude Sonnet 4.5 or GPT-5 models.
* `glm-4.6` ([QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-4.6-GPTQ-Int4-Int8Mix)) is a widely popular programming LLM model and exhibits a similar level of model performance to Claude Sonnet 4 or Gemini 2.5 Pro models.
* `minimax-m2` ([MiniMaxAI/MiniMax-M2](https://huggingface.co/MiniMaxAI/MiniMax-M2)) is a widely popular programming LLM model and exhibits a similar level of model performance to Claude Sonnet 4 or Gemini 2.5 Pro models, while being able to fit the official FP8 parameters in four A100 GPUs with ample context length.
* `gpt-oss` ([openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)) is a very capable agentic model, adequate for general-purpose usage, while requiring only one A100 GPU or two RTX A6000 GPUs for full context due to sliding window attention and official MXFP4 quantization, which is a fraction of other frontier models. This is our candidate for an “LTS” model used for reproducible research, that supersedes the deprecated or removed Llama3 models.
* `gemma3` was changed to 2x RTX A6000 GPUs instead of 2x A100 GPUs to conserve the latter. The model’s special sliding window attention method allows full context to fit in this case.
* `glm-v` was changed to the official [zai-org/GLM-4.5V-FP8](https://huggingface.co/zai-org/GLM-4.5V-FP8) model and uses 4x L40 GPUs instead of 2x A100 GPUs to conserve the latter and gain FP8 capabilities.

Removed:

* `llama3` ([meta-llama/Llama-3.2-90B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct)) has been officially pulled down, due to consuming 4 A100 GPUs that can be used for much more frontier models, such as MiniMax-M2 or GLM-4.6, while being much worse in performance than models that fit in one GPU.
* `deepseek-r1` ([QuantTrio/DeepSeek-R1-0528-GPTQ-Int4-Int8Mix-Medium](https://huggingface.co/QuantTrio/DeepSeek-R1-0528-GPTQ-Int4-Int8Mix-Medium)) has been officially pulled down, due to consuming 8 GPUs but being very slow (5-6 tokens/s) in any larger context size. There are many similar models that work well, although not necessarily better in every way, and are faster. This is an example of the `larger models that require a lot of GPUs are likely to be removed earlier` phrase above.
* `watt` ([watt-ai/watt-tool-8B](https://huggingface.co/watt-ai/watt-tool-8B)) has been removed due to inactivity.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/ai/llm-managed.mdx)

[Previous  
Deploying Coder](/documentation/userdocs/coder/deploy)  [Next  
LLM in JupyterHub](/documentation/userdocs/ai/llm-jupyterhub)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.