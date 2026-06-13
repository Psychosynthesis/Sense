import * as tf from '@tensorflow/tfjs-core';

/**
 * Identifier of a supported text-embedding model.
 *
 * The universe of supported models is intentionally limited to
 * text-embedding models. This library does not try to support "any
 * transformer", only models that map text to a fixed-size embedding.
 */
export type ModelId =
    |'use-lite'
    |'use-qna'
    |'paraphrase-multilingual-minilm';

/**
 * Runtime that executes a model.
 *
 * - `tfjs-graph`: a TensorFlow.js GraphModel (the classic USE path).
 * - `transformers-js`: the `@huggingface/transformers` feature-extraction
 *   pipeline (ONNX based).
 */
export type Runtime = 'tfjs-graph'|'transformers-js';

/**
 * Where to resolve model weights from.
 *
 * - `github`: the weights bundled in this repository (served from GitHub
 *   raw or any mirror). This is the default and removes the dependency on
 *   tfhub.dev.
 * - `tfhub`: the original tfhub.dev / Kaggle hosted weights.
 * - `huggingface`: the Hugging Face hub (for `transformers-js` models).
 * - `custom`: a fully custom location provided through explicit URLs.
 */
export type WeightsSource = 'github'|'tfhub'|'huggingface'|'custom';

/**
 * Static description of a text-embedding model. Replaces the previous loose
 * collection of `modelUrl` / `vocabUrl` / `backend` parameters.
 */
export interface ModelProfile {
  id: ModelId;
  runtime: Runtime;
  embeddingDim: number;

  /** Default GraphModel url (only for `tfjs-graph` runtimes). */
  modelUrl?: string;
  /** Default vocabulary url (only for `tfjs-graph` runtimes). */
  vocabUrl?: string;
  /** Hugging Face model id (only for `transformers-js` runtimes). */
  hfModelId?: string;

  tokenizer: 'use-sentencepiece'|'runtime-managed';
  inputFormat: 'use-sparse'|'runtime-managed';
  pooling: 'none'|'mean';
  normalize: boolean;
}

/**
 * The single contract every embedder in this library implements.
 *
 * `embed()` always returns a `tf.Tensor2D` of shape `[N, embeddingDim]` so
 * downstream code (classifier heads, similarity, knn) stays runtime
 * agnostic.
 */
export interface TextEmbedder {
  readonly id: string;
  readonly embeddingDim: number;
  readonly outputType: 'tfjs-tensor2d';

  embed(inputs: string[]|string): Promise<tf.Tensor2D>;
  dispose?(): void;
}

/**
 * Tokenizer adapter contract. The encoding type is generic because different
 * models produce very different encodings (sparse indices/values for USE,
 * dense token id matrices for others), so a single `encode(): number[]`
 * signature is not expressive enough.
 */
export interface TokenizerAdapter<TEncoding> {
  encodeBatch(texts: string[]): Promise<TEncoding>;
}

/**
 * Options accepted by the public `load()` entry point.
 */
export interface LoadConfig {
  /** Which model profile to load. Defaults to `use-lite`. */
  model?: ModelId;

  /** Where to resolve weights from. Defaults to `github`. */
  weightsSource?: WeightsSource;

  /** Override the GraphModel url (implies `weightsSource: 'custom'`). */
  modelUrl?: string;
  /** Override the vocabulary url (implies `weightsSource: 'custom'`). */
  vocabUrl?: string;

  /** Override the Hugging Face model id for `transformers-js` models. */
  hfModelId?: string;
  /**
   * Custom base location for `transformers-js` weights, e.g. a GitHub raw
   * url or any self-hosted mirror. Maps to transformers.js `env.remoteHost`.
   */
  remoteHost?: string;
  /** Local directory for self-hosted `transformers-js` weights. */
  localModelPath?: string;
  /** Quantization dtype for `transformers-js` (defaults to `q8`). */
  dtype?: 'fp32'|'fp16'|'q8'|'int8'|'uint8'|'q4';
}

/**
 * Metadata that ties a trained classifier head to the exact embedding
 * backend it was trained on. Persist this next to the classifier weights so
 * that, a month later, it is unambiguous which head matches which
 * embeddings.
 */
export interface ClassifierBundleMetadata {
  embeddingModel: ModelId;
  embeddingDim: number;
  pooling: 'none'|'mean';
  normalize: boolean;
  headVersion: string;
  labels: string[];
}
