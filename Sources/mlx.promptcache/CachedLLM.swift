import Foundation
import MLX
@preconcurrency import MLXLMCommon
import Tokenizers

extension ModelContext: @retroactive @unchecked Sendable {}

public actor CachedLLM {
    
    private let modelContext: ModelContext
    private var promptCache: PromptCache?
    
    public private(set) var cacheStats = CacheStats()


    public init(modelContext: ModelContext) {
        self.modelContext = modelContext
    }

    public func clearCache() {
        promptCache = nil
        cacheStats.resets += 1
    }
    
    public func generate(
        input: UserInput,
        parameters: GenerateParameters
    ) async throws -> AsyncThrowingStream<String, Error> {
        
        let lmInput = try await modelContext.processor.prepare(input: input)
        let newTokens = lmInput.text.tokens.asArray(Int.self)
        let modelKey = getCacheKey(temperature: parameters.temperature, topP: parameters.topP)
        
        let (tokensToProcess, existingCache) = getCachedState(
            modelKey: modelKey,
            newTokens: newTokens
        )
        
        let lmInputForIterator = tokensToProcess.count == newTokens.count
            ? lmInput
            : LMInput(tokens: MLXArray(tokensToProcess))
        
        let kvCacheToUse = existingCache?.kvCache ?? makeNewCache()
        
        let hasExistingCache = existingCache != nil
        let existingTokenCount = existingCache?.tokens.count ?? 0

        let iterator = try TokenIterator(
            input: lmInputForIterator,
            model: modelContext.model,
            cache: kvCacheToUse,
            parameters: parameters
        )
        
        return AsyncThrowingStream { continuation in
            let task = Task {
                var allGeneratedTokens = [Int]()
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: self.modelContext.tokenizer)

                for token in iterator {
                    if Task.isCancelled { break }
                    
                    allGeneratedTokens.append(token)
                    
                    detokenizer.append(token: token)
                    if let chunk = detokenizer.next() {
                        continuation.yield(chunk)
                    }
                }
                 
                // Update the cache with the successfully generated tokens
                // The KVCache objects are updated in-place during generation since they're reference types
                if hasExistingCache {
                    // We had an existing cache - need to update it
                    if var currentCache = self.promptCache, currentCache.tokens.count == existingTokenCount {
                        // Cache is still valid and at the expected state
                        currentCache.tokens.append(contentsOf: tokensToProcess)
                        currentCache.tokens.append(contentsOf: allGeneratedTokens)
                        currentCache.lastAccessedAt = Date()
                        self.promptCache = currentCache
                    } else {
                        // Cache was invalidated during generation, create new one
                        let allTokens = newTokens + allGeneratedTokens
                        self.promptCache = PromptCache(
                            modelKey: modelKey,
                            tokens: allTokens,
                            kvCache: kvCacheToUse,
                            createdAt: Date(),
                            lastAccessedAt: Date()
                        )
                    }
                } else {
                    // Create new cache with all tokens
                    let allTokens = newTokens + allGeneratedTokens
                    self.promptCache = PromptCache(
                        modelKey: modelKey,
                        tokens: allTokens,
                        kvCache: kvCacheToUse,
                        createdAt: Date(),
                        lastAccessedAt: Date()
                    )
                }

                continuation.finish()
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }
    
    private func makeNewCache() -> [KVCache] {
        let blueprintCache = modelContext.model.newCache(parameters: nil)
        return blueprintCache.map { _ in CustomKVCache() }
    }

    private func getCacheKey(temperature: Float, topP: Float) -> String {
        "\(modelContext.configuration.name)-\(temperature)-\(topP)"
    }
    
    private func getCachedState(
        modelKey: String,
        newTokens: [Int]
    ) -> (tokensToProcess: [Int], cache: PromptCache?) {
        
        guard var cache = promptCache, cache.modelKey == modelKey, cache.isValid else {
            cacheStats.misses += 1
            clearCache()
            return (newTokens, nil)
        }
        
        let commonLength = commonPrefixLength(cache.tokens, newTokens)
        
        guard commonLength > 0 else {
            cacheStats.misses += 1
            clearCache()
            return (newTokens, nil)
        }
        
        cache.updateAccess()
        
        cacheStats.tokensReused = commonLength
                
        let tokensToTrim = cache.tokens.count - commonLength
        if tokensToTrim > 0 {
            cacheStats.trims += 1

            for case let trimmableCache as CustomKVCache in cache.kvCache {
                trimmableCache.trim(count: tokensToTrim)
            }
            cache.tokens.removeLast(tokensToTrim)
        }
        
        self.promptCache = cache
        return (Array(newTokens[commonLength...]), cache)
    }


    private func commonPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
        let len = min(a.count, b.count)
        for i in 0..<len { if a[i] != b[i] { return i } }
        return len
    }
    
    public struct CacheStats: Sendable {
        public var hits = 0
        public var misses = 0
        public var trims = 0
        public var resets = 0
        public var tokensReused = 0
        public var hitRate: Double { let total = hits + misses; return total > 0 ? Double(hits) / Double(total) : 0 }
    }
}
