import XCTest
@testable import mlx_promptcache
import MLXLMCommon
import MLXLLM

final class MLXPromptCacheTests: XCTestCase {

    var modelContext: ModelContext!
    var cachedLLM: CachedLLM!

    override func setUpWithError() throws {
        try super.setUpWithError()
        
        let modelId = "mlx-community/Qwen3-0.6B-4bit-DWQ-053125"
        
        let modelConfiguration = LLMRegistry.shared.configuration(id: modelId)
        let localModelPath = modelConfiguration.modelDirectory()
        
        guard FileManager.default.fileExists(atPath: localModelPath.path) else {
            throw XCTSkip("""
            Skipping test: Model '\(modelId)' not found locally at \(localModelPath.path).
            Please download it first using the mlx_lm Python package:
            'mlx_lm.generate --model \(modelId) --prompt "hello" --max-tokens 1'
            """)
        }
        
    }

    override func tearDown() {
        modelContext = nil
        cachedLLM = nil
        super.tearDown()
    }
    
    private func collectString(from stream: AsyncThrowingStream<String, Error>) async throws -> String {
        var response = ""
        for try await chunk in stream {
            response += chunk
        }
        return response
    }

    func testMultiTurnCacheWithQwen3() async throws {
        let modelId = "mlx-community/Qwen3-4B-4bit-DWQ-053125"
        let modelConfiguration = LLMRegistry.shared.configuration(id: modelId)
        let localModelPath = modelConfiguration.modelDirectory()
        
        guard FileManager.default.fileExists(atPath: localModelPath.path) else {
            throw XCTSkip("""
            Skipping test: Model '\(modelId)' not found locally at \(localModelPath.path).
            Please download it first using the mlx_lm Python package:
            'mlx_lm.generate --model \(modelId) --prompt "hello" --max-tokens 1'
            """)
        }
        
        print("Loading model: \(modelId)...")
        self.modelContext = try await LLMModelFactory.shared.load(configuration: modelConfiguration)
        print("Model loaded successfully.")

        self.cachedLLM = try XCTUnwrap(
            CachedLLM(modelContext: self.modelContext),
            "Failed to initialize CachedLLM. The model type might not be supported."
        )
        
        let generateParams = GenerateParameters(maxTokens: 50, temperature: 0.0)
        
        // MARK: - Turn 1: Prime the Cache (Expect a Cache Miss)
        print("\n--- Turn 1: Priming the cache ---")
        
        let prompt1 = "My name is Arthur. What is the capital of England?"
        var chatHistory: [Chat.Message] = [.user(prompt1)]
        
        let stream1 = try await cachedLLM.generate(
            input: UserInput(chat: chatHistory),
            parameters: generateParams
        )
        
        let response1 = try await collectString(from: stream1)
        print("User: \(prompt1)")
        print("Assistant: \(response1)")
        
        // Update chat history for the next turn
        chatHistory.append(.assistant(response1))

        // Verification for Turn 1
        var stats = await cachedLLM.cacheStats
        let tokensReusedAfterTurn1 = stats.tokensReused
        print("Tokens reused after Turn 1: \(tokensReusedAfterTurn1)")
        
        XCTAssertEqual(tokensReusedAfterTurn1, 0, "Turn 1 should reuse 0 tokens (no cache yet).")
        XCTAssertTrue(response1.lowercased().contains("london"), "Response for Turn 1 should contain 'london'.")
        
        // MARK: - Turn 2: Use the Cache (Expect a Cache Hit)
        print("\n--- Turn 2: Testing cache hit with context ---")
        
        let prompt2 = "And what is my name?"
        chatHistory.append(.user(prompt2)) // Add new user message to the history
        
        let stream2 = try await cachedLLM.generate(
            input: UserInput(chat: chatHistory),
            parameters: generateParams
        )
        
        let response2 = try await collectString(from: stream2)
        print("User: \(prompt2)")
        print("Assistant: \(response2)")
        
        // Update chat history
        chatHistory.append(.assistant(response2))
        
        // Verification for Turn 2
        stats = await cachedLLM.cacheStats
        let tokensReusedAfterTurn2 = stats.tokensReused
        let tokensReusedInTurn2 = tokensReusedAfterTurn2 - tokensReusedAfterTurn1
        print("Tokens reused after Turn 2: \(tokensReusedAfterTurn2)")
        XCTAssertGreaterThan(tokensReusedInTurn2, 0, "Turn 2 should reuse tokens from the cache.")
        XCTAssertEqual(stats.trims, 0, "No trimming should occur for a simple follow-up.")
        XCTAssertTrue(response2.lowercased().contains("arthur"), "Response for Turn 2 must contain 'Arthur', proving context was retained from the cache.")

        // MARK: - Turn 3: Invalidate the Cache (Expect a Cache Miss)
        print("\n--- Turn 3: Testing cache miss after history change ---")
        
        let prompt3 = "What is the capital of Germany?"
        let newChatHistory: [Chat.Message] = [.user(prompt3)] // Start a new, unrelated conversation
        
        let stream3 = try await cachedLLM.generate(
            input: UserInput(chat: newChatHistory),
            parameters: generateParams
        )
        
        let response3 = try await collectString(from: stream3)
        print("User: \(prompt3)")
        print("Assistant: \(response3)")
        
        // Verification for Turn 3
        stats = await cachedLLM.cacheStats
        let tokensReusedAfterTurn3 = stats.tokensReused
        let tokensReusedInTurn3 = tokensReusedAfterTurn3 - tokensReusedAfterTurn2
        print("Tokens reused after Turn 3: \(tokensReusedAfterTurn3)")
        // Even it's a new conversation, we might still reuse some tokens like BOS, EOS, etc.
        XCTAssertLessThan(tokensReusedInTurn3, tokensReusedInTurn2, "Turn 3 should reuse fewer tokens than Turn 2 since it's a new conversation.")
        XCTAssertTrue(response3.lowercased().contains("berlin"), "Response for Turn 3 should contain 'Berlin'.")
        XCTAssertFalse(response3.lowercased().contains("arthur"), "Response for Turn 3 should not contain old context, proving the cache was reset.")
    }
}
