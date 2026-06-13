/**
* Sense - Universal text embedding library for TensorFlow.js and Transformers.js models
 *
 * A small, model-agnostic library that turns text into fixed-size embeddings
 * (feature vectors). Its scope is intentionally limited to text-embedding
 * models; it does not try to support "any transformer".
 */

import { createEmbedder } from './factory';
import { GITHUB_RAW_BASE, resolveProfile } from './profiles';
import { loadTokenizer as loadTokenizerInternal } from './tokenizer/tokenizer';
import { LoadConfig, TextEmbedder } from './types';
import { loadQnA } from './use_qna';

// Injected at build time from package.json by @rollup/plugin-replace.
export const version: string = '__SENSE_PKG_VERSION__';

/**
 * Loads a text-embedding model and returns a `TextEmbedder`.
 *
 * By default this loads the classic USE lite model (512-dim English
 * embeddings) from the weights bundled in this repository, preserving the
 * original behaviour. Pass `{model: 'paraphrase-multilingual-minilm'}` for
 * multilingual (incl. Russian) embeddings.
 *
 * @param config Optional load configuration.
 */
export async function load(config: LoadConfig = {}): Promise<TextEmbedder> {
  const profile = resolveProfile(config);
  return createEmbedder(profile, config);
}

/**
 * Load the USE SentencePiece tokenizer independently from any embedder.
 *
 * @param pathToVocabulary (optional) Provide a path to the vocabulary file.
 */
export async function loadTokenizer(pathToVocabulary?: string) {
  return loadTokenizerInternal(
      pathToVocabulary || `${GITHUB_RAW_BASE}/use-lite/vocab.json`);
}

export { loadQnA };
export { UseQnAEmbedder } from './use_qna';
export { createEmbedder };
export { UseLiteEmbedder } from './embedders/use_lite_embedder';
export { TransformersJsEmbedder } from './embedders/transformers_js_embedder';
export { UseSparseInputAdapter } from './embedders/use_sparse_input';
export { TfjsGraphRuntime } from './runtime/tfjs_graph_runtime';
export { UseSentencePieceTokenizer } from './tokenizer/tokenizer';
export { DEFAULT_MODEL_ID, GITHUB_RAW_BASE, MODEL_PROFILES, resolveProfile } from './profiles';

export {
  ClassifierBundleMetadata,
  LoadConfig,
  ModelId,
  ModelProfile,
  Runtime,
  TextEmbedder,
  TokenizerAdapter,
  WeightsSource,
} from './types';
