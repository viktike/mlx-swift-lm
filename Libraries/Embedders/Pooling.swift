// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXLinalg
import MLXLMCommon
import MLXNN

public struct PoolingConfiguration: Codable {
    public let dimension: Int
    public let poolingModeClsToken: Bool
    public let poolingModeMeanTokens: Bool
    public let poolingModeMaxTokens: Bool
    public let poolingModeLastToken: Bool

    enum CodingKeys: String, CodingKey {
        case dimension = "word_embedding_dimension"
        case poolingModeClsToken = "pooling_mode_cls_token"
        case poolingModeMeanTokens = "pooling_mode_mean_tokens"
        case poolingModeMaxTokens = "pooling_mode_max_tokens"
        case poolingModeLastToken = "pooling_mode_lasttoken"
    }
}

func loadPooling(modelDirectory: URL, model: EmbeddingModel) -> Pooling {
    let configurationURL = modelDirectory.appending(components: "1_Pooling", "config.json")
    if let poolingConfig = try? JSONDecoder.json5().decode(
        PoolingConfiguration.self, from: Data(contentsOf: configurationURL))
    {
        return Pooling(config: poolingConfig)
    }

    if let strategy = model.poolingStrategy {
        return Pooling(strategy: strategy)
    }

    return Pooling(strategy: .none)
}

public class Pooling: Module {
    public enum Strategy {
        case mean
        case cls
        case first
        case last
        case max
        case none
    }
    public private(set) var strategy: Strategy
    public private(set) var dimension: Int?

    public init(
        strategy: Strategy, dimension: Int? = nil
    ) {
        self.strategy = strategy
        self.dimension = dimension
    }

    public init(
        config: PoolingConfiguration
    ) {
        dimension = config.dimension
        if config.poolingModeClsToken {
            strategy = .cls
        } else if config.poolingModeMeanTokens {
            strategy = .mean
        } else if config.poolingModeMaxTokens {
            strategy = .max
        } else if config.poolingModeLastToken {
            strategy = .last
        } else {
            strategy = .first
        }
    }

    public func callAsFunction(
        _ inputs: EmbeddingModelOutput, mask: MLXArray? = nil, normalize: Bool = false,
        applyLayerNorm: Bool = false
    ) -> MLXArray {
        let _mask = mask ?? MLXArray.ones(Array(inputs.hiddenStates?.shape[0 ..< 2] ?? [0]))

        var pooled: MLXArray
        switch self.strategy {
        case .mean:
            pooled =
                sum(
                    inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]),
                    axis: 1)
                / sum(_mask, axis: -1, keepDims: true)
        case .max:
            pooled = MLX.max(
                inputs.hiddenStates! * _mask.expandedDimensions(axes: [-1]), axis: 1)
        case .first:
            pooled = inputs.hiddenStates![0..., 0, 0...]
        case .last:
            let hiddenStates = inputs.hiddenStates!
            let tokenCounts = sum(_mask.asType(.int32), axis: -1)
            let tokenIndices = MLX.maximum(
                tokenCounts - MLXArray(Int32(1)),
                MLXArray(Int32(0))
            )
            let indices = tokenIndices.expandedDimensions(axes: [1, 2])
            let gathered = MLX.takeAlong(hiddenStates, indices, axis: 1)
            pooled = MLX.squeezed(gathered, axis: 1)
        case .cls:
            pooled =
                inputs.pooledOutput
                ?? inputs.hiddenStates![0..., 0, 0...]
        case .none:
            pooled = inputs.pooledOutput ?? inputs.hiddenStates!
        }
        if applyLayerNorm {
            pooled = MLXFast.layerNorm(pooled, eps: 1e-5)
        }
        if let dimension {
            pooled = pooled[0..., 0 ..< dimension]
        }
        if normalize {
            pooled = pooled.l2Normalized()
        }
        return pooled
    }
}
