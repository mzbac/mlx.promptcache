# mlx.promptcache

A Swift library for prompt caching in Large Language Models (LLMs) built specifically for [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples)' MLXLLM and MLXLMCommon modules. 

## Requirements

- macOS 14+ or iOS 16+
- Swift 5.9+

## Installation

Add this package to your Swift Package Manager dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/mzbac/mlx.promptcache.git", from: "0.0.1")
]
```

## Usage

### Basic Example

```swift
import mlx_promptcache
import MLXLLM

// Load your model
let modelConfiguration = LLMRegistry.shared.configuration(id: "model-id")
let modelContext = try await LLMModelFactory.shared.load(configuration: modelConfiguration)

// Create cached LLM instance
let cachedLLM = CachedLLM(modelContext: modelContext)

// Generate with automatic caching
let generateParams = GenerateParameters(maxTokens: 100, temperature: 0.7)
let stream = try await cachedLLM.generate(
    input: UserInput(chat: chatHistory),
    parameters: generateParams
)

// Process streaming response
for try await chunk in stream {
    print(chunk, terminator: "")
}
```

### Multi-turn Conversation

```swift
var chatHistory: [Chat.Message] = []

// First turn - cache miss
chatHistory.append(.user("Tell me about Swift"))
let stream1 = try await cachedLLM.generate(
    input: UserInput(chat: chatHistory),
    parameters: generateParams
)
let response1 = try await stream1.collect()
chatHistory.append(.assistant(response1))

// Second turn - cache hit (reuses context)
chatHistory.append(.user("What are its main features?"))
let stream2 = try await cachedLLM.generate(
    input: UserInput(chat: chatHistory),
    parameters: generateParams
)
```

### Monitoring Cache Performance

```swift
// Get cache statistics
let stats = await cachedLLM.cacheStats
print("Cache hits: \(stats.hits)")
print("Cache misses: \(stats.misses)")
print("Tokens reused: \(stats.tokensReused)")
print("Hit rate: \(String(format: "%.2f%%", stats.hitRate * 100))")

// Reset cache if needed
await cachedLLM.resetCache()
```

## Acknowledgments

This library is inspired by and follows the prompt caching patterns from:
- [mlx-lm](https://github.com/ml-explore/mlx-lm) - The original Python implementation of prompt caching in MLX
- [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples) - Swift implementations of MLX examples

The prefix prompt caching approach is directly adapted from mlx-lm's efficient caching strategy, bringing the same performance benefits to Swift applications.
