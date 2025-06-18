import Foundation
import MLX
import MLXLMCommon


final class CustomKVCache: KVCache, Evaluatable {
    
    
    private(set) var keys: MLXArray?
    private(set) var values: MLXArray?
    
    private(set) var offset = 0
    
    private let step = 256
    
    
    var maxSize: Int? { nil }
    
    func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    // Copied the simple KVCache implementation from MLXLLM, ideally we should update mlx-swift-examples to support trim cache
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previousOffset = self.offset
        
        let shouldExpand =
            if let currentKeys = self.keys, (previousOffset + keys.dim(2)) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        
        if shouldExpand {
            expandCache(keyTemplate: keys, valueTemplate: values)
        }
        
        self.offset += keys.dim(2)
        
        // Update the internal arrays with the new data
        self.keys?[0..., 0..., previousOffset ..< self.offset, 0...] = keys
        self.values?[0..., 0..., previousOffset ..< self.offset, 0...] = values
        
        // Return the full cache up to the new offset
        return (
            self.keys![0..., 0..., ..<self.offset, 0...],
            self.values![0..., 0..., ..<self.offset, 0...]
        )
    }

    func trim(count: Int) {
        guard count > 0, offset >= count else { return }
        
        let newOffset = offset - count
        
        // Slice the arrays to the new, smaller size
        if let keys = self.keys, let values = self.values {
            self.keys = keys[0..., 0..., ..<newOffset, 0...]
            self.values = values[0..., 0..., ..<newOffset, 0...]
        }
        
        self.offset = newOffset
    }
    

    private func expandCache(keyTemplate: MLXArray, valueTemplate: MLXArray) {
        let (B, kvHeads, S, kHeadDim) = keyTemplate.shape4
        let vHeadDim = valueTemplate.dim(3)
        
        let nSteps = (step + S - 1) / step
        let kShape = [B, kvHeads, nSteps * step, kHeadDim]
        let vShape = [B, kvHeads, nSteps * step, vHeadDim]
        
        let newK = MLXArray.zeros(kShape, dtype: keyTemplate.dtype)
        let newV = MLXArray.zeros(vShape, dtype: valueTemplate.dtype)
        
        if var currentKeys = self.keys, var currentValues = self.values {
            if offset % step != 0 {
                currentKeys = currentKeys[0..., 0..., ..<offset, 0...]
                currentValues = currentValues[0..., 0..., ..<offset, 0...]
            }
            self.keys = concatenated([currentKeys, newK], axis: 2)
            self.values = concatenated([currentValues, newV], axis: 2)
        } else {
            self.keys = newK
            self.values = newV
        }
    }
}
