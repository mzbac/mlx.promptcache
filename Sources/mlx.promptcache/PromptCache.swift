import Foundation
import MLX
import MLXLMCommon

public struct PromptCache {
    let modelKey: String
    
    var tokens: [Int]
    
    var kvCache: [KVCache]
    
    let createdAt: Date
    
    var lastAccessedAt: Date
    
    var isValid: Bool {
        Date().timeIntervalSince(lastAccessedAt) < 1800 // 30 minutes
    }
    
    mutating func updateAccess() {
        lastAccessedAt = Date()
    }
}
