import * as tf from '@tensorflow/tfjs-core';

import { LoadConfig, ModelProfile, TextEmbedder } from '../types';

/**
 * Minimal structural types for the parts of `@huggingface/transformers` we
 * use. The dependency is optional and lazily imported, so we avoid taking a
 * hard type dependency on it.
 */
interface FeatureExtractionOutput {
  tolist(): number[][];
  dims: number[];
}

type FeatureExtractionPipeline =
    (texts: string[],
     options: {pooling: 'none'|'mean'|'cls'; normalize: boolean}) =>
        Promise<FeatureExtractionOutput>;

interface PipelineOptions {
  dtype?: string;
}

interface TransformersEnv {
  allowRemoteModels: boolean;
  allowLocalModels: boolean;
  remoteHost?: string;
  localModelPath?: string;
}

interface TransformersModule {
  pipeline(
      task: string, model: string,
      options?: PipelineOptions): Promise<FeatureExtractionPipeline>;
  env: TransformersEnv;
}

/**
 * Embedder backed by the `@huggingface/transformers` feature-extraction
 * pipeline (ONNX runtime). Used for multilingual text-embedding models such
 * as `Xenova/paraphrase-multilingual-MiniLM-L12-v2`.
 *
 * `@huggingface/transformers` is an optional dependency and is imported
 * lazily so the classic USE / tfjs-graph path never pulls in the ONNX
 * runtime.
 */
export class TransformersJsEmbedder implements TextEmbedder {
  readonly id: string;
  readonly embeddingDim: number;
  readonly outputType = 'tfjs-tensor2d' as const;

  private extractor!: FeatureExtractionPipeline;

  constructor(
      private readonly profile: ModelProfile,
      private readonly config: LoadConfig = {}) {
    this.id = profile.id;
    this.embeddingDim = profile.embeddingDim;
  }

  async load(): Promise<void> {
    const transformers = await this.importTransformers();

    if (this.config.localModelPath != null) {
      transformers.env.allowLocalModels = true;
      transformers.env.localModelPath = this.config.localModelPath;
    }
    if (this.config.remoteHost != null) {
      transformers.env.remoteHost = this.config.remoteHost;
    }

    this.extractor = await transformers.pipeline(
        'feature-extraction', this.profile.hfModelId!,
        {dtype: this.config.dtype || 'q8'});
  }

  async embed(inputs: string[]|string): Promise<tf.Tensor2D> {
    const texts = typeof inputs === 'string' ? [inputs] : inputs;

    const result = await this.extractor(texts, {
      pooling: this.profile.pooling === 'none' ? 'none' : 'mean',
      normalize: this.profile.normalize,
    });

    const vectors = result.tolist();
    return tf.tensor2d(vectors, [vectors.length, this.embeddingDim], 'float32');
  }

  private async importTransformers(): Promise<TransformersModule> {
    // Use a non-literal specifier so the bundler/compiler treats this as a
    // truly optional, lazily resolved dependency.
    const moduleId = ['@huggingface', 'transformers'].join('/');
    try {
      const mod: TransformersModule = await import(moduleId);
      return mod;
    } catch (cause) {
      throw new Error(
          'The model "' + this.profile.id + '" requires the optional ' +
          'dependency "@huggingface/transformers". Install it with: ' +
          'npm install @huggingface/transformers',
          {cause});
    }
  }
}
