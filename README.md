# MLX Swift LM

MLX Swift LM is a Swift package to build tools and applications with large
language models (LLMs) and vision language models (VLMs) in [MLX Swift](https://github.com/ml-explore/mlx-swift).

Some key features include:

This fork adds native [JANG](https://jangq.ai) mixed-precision quantization, **TurboQuant KV cache compression** (4.7-5.0x memory savings), **Gemma 4**, **Mistral Small 4**, speculative decoding, VLM detection, and MoE performance optimizations on top of the full upstream library. Existing apps don't need to change anything -- all upstream APIs are preserved.

For some example applications and tools that use MLX Swift LM check out
the [MLX Swift Examples](https://github.com/ml-explore/mlx-swift-examples).

# Using MLX Swift LM

The MLXLLM, MLXVLM, MLXLMCommon, and MLXEmbedders libraries are available
as Swift Packages.

Add the following dependency to your Package.swift:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", branch: "main"),
```

or use the latest release:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift-lm/", .upToNextMinor(from: "2.29.1")),
```

Then add one or more libraries to the target as a dependency:
### VLM Detection

Check at runtime whether a model supports vision input:

```swift
if await container.isVLM {
    // safe to pass images
}
```

Works from `MLXLMCommon` alone -- no need to import `MLXVLM`.

### TurboQuant KV Cache Compression

Compress the KV cache **4.7-5.0x** during inference with no quality loss on short outputs and minimal divergence on long outputs. Based on Google DeepMind's research ([arXiv:2504.19874](https://arxiv.org/abs/2504.19874)).

**One line to enable, works with every model -- no model changes needed:**

```swift
// 3-bit (recommended default -- best compression)
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 3, valueBits: 3))

// 4-bit (higher quality, less compression)
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 4, valueBits: 4))
```

**Works with ChatSession for multi-turn conversations:**

```swift
let session = ChatSession(
    container,
    generateParameters: GenerateParameters(
        kvMode: .turboQuant(keyBits: 3, valueBits: 3)))

let reply1 = try await session.respond(to: "What is the capital of Japan?")
// "Tokyo"
let reply2 = try await session.respond(to: "What country is that city in?")
// "Japan" -- context preserved across turns
```

**Works with speculative decoding:**

```swift
let params = GenerateParameters(
    kvMode: .turboQuant(keyBits: 3, valueBits: 3))
let result = try await mainModel.generate(
    input: input, parameters: params, draft: draftModel)
```

#### How It Works

TurboQuant compresses the KV cache after prefill using three techniques:

1. **Randomized Hadamard rotation** -- spreads information uniformly across all dimensions so a single codebook works optimally for every component
2. **Lloyd-Max optimal codebook** -- minimizes quantization error for the statistical distribution of rotated vector components
3. **QJL residual correction** (keys only) -- 1-bit random projection that corrects the exponential error amplification in softmax attention scores

The compressed cache is decoded once into a float16 buffer. During generation, new tokens are scatter-written into a pre-allocated window. Models see normal float16 arrays from `update()` -- they never know compression happened.

#### Memory Savings

| Model | Context | Float Cache | TurboQuant-3 | Savings |
|-------|---------|-------------|-------------|---------|
| Gemma 4 26B MoE | 2K | 84 MB | 17 MB | **4.9x** |
| Qwen 3.5-35B | 32K | 655 MB | 135 MB | **4.9x** |
| Mistral Small 4 (119B) | 2K | 1,208 MB | 244 MB | **4.9x** |

#### Tested Configurations

| Model | Architecture | Format | Modes | Result |
|-------|-------------|--------|-------|--------|
| Gemma 4 26B | MoE (128 experts) | MLX 4-bit | LLM, VLM, multi-turn | Identical on short, near-identical on long |
| Gemma 4 31B | Dense | MLX 4-bit | LLM, multi-turn | Identical on short, near-identical on long |
| Gemma 4 31B | Dense | JANG 4M | LLM | Identical |
| NemotronH 30B-A3B | Hybrid SSM/attention | JANG 4M | LLM, multi-turn | Identical |
| NemotronH 30B-A3B | Hybrid SSM/attention | JANG 2L | LLM | Near-identical |

TurboQuant automatically skips non-KV cache layers (MambaCache for SSM, RotatingKVCache for sliding window). If `maxKVSize` is set (all RotatingKVCache), TurboQuant gracefully does nothing.

---

## Supported Models

### LLMs (50+ architectures)

Llama, Mistral, Phi, Phi-3, Phi-MoE, Gemma, Gemma 2, Gemma 3, Gemma 3n, **Gemma 4**, Qwen2, Qwen3, Qwen3-MoE, Qwen3.5, Qwen3.5-MoE, DeepSeek-V3, Cohere, OpenELM, InternLM2, Starcoder2, MiniCPM, Granite, Granite-MoE-Hybrid, MiMo, MiMo-V2-Flash, MiniMax, GLM-4, GLM-4-MoE, Falcon-H1, Bitnet, SmolLM3, ERNIE 4.5, LFM2, LFM2-MoE, Baichuan-M1, Exaone4, GPT-OSS, Lille-130m, OLMoE, OLMo2, OLMo3, Bailing-MoE, NanoChat, Nemotron-H, AF-MoE, Jamba, **Mistral Small 4** (MLA + MoE), Mistral3, Apertus, and more.

### VLMs (17+ architectures)

PaliGemma, Qwen2-VL, Qwen2.5-VL, Qwen3-VL, Qwen3.5, Qwen3.5-MoE, Gemma 3, **Gemma 4**, SmolVLM2, FastVLM, Pixtral, **Mistral Small 4** (MLA + Pixtral), Mistral3, LFM2-VL, GLM-OCR, Idefics3, and more.

### Embedders

Sentence Transformers, BERT, and other popular embedding models.

---

## Quick Start

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/osaurus-ai/mlx-swift-lm", branch: "main"),
```

Then add tokenizer and downloader integrations:

```swift
.package(url: "https://github.com/DePasqualeOrg/swift-tokenizers-mlx", from: "0.1.0"),
.package(url: "https://github.com/DePasqualeOrg/swift-hf-api-mlx", from: "0.1.0"),
```

And add the libraries to your target:

```swift
.target(
    name: "YourTargetName",
    dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm")
    ]),
```

Alternatively, add `https://github.com/ml-explore/mlx-swift-lm/` to the
`Project Dependencies` and set the `Dependency Rule` to `Branch` and `main` in
Xcode.

# Quick Start

See also [MLXLMCommon](Libraries/MLXLMCommon). You can get started with a wide
variety of open weights LLMs and VLMs using this simplified API:

```swift
let model = try await loadModel(id: "mlx-community/Qwen3-4B-4bit")
let session = ChatSession(model)
print(try await session.respond(to: "What are two things to see in San Francisco?"))
print(try await session.respond(to: "How about a great place to eat?"))
```

Or use the underlying API to control every aspect of the evaluation.

# Documentation

Developers can use these examples in their own programs -- just import the swift package!

- [Porting and implementing models](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon/porting)
- [MLXLLMCommon](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxlmcommon) -- common API for LLM and VLM
- [MLXLLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxllm) -- large language model example implementations
- [MLXVLM](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxvlm) -- vision language model example implementations
- [MLXEmbedders](https://swiftpackageindex.com/ml-explore/mlx-swift-lm/main/documentation/mlxembedders) -- popular Encoders / Embedding models example implementations
