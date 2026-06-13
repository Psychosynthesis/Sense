import { LoadConfig, ModelId, ModelProfile } from './types';

/**
 * Base url for the model weights bundled in this repository. By default
 * weights are served from GitHub raw, which removes the hard dependency on
 * tfhub.dev. Override `weightsSource`/`modelUrl`/`vocabUrl` in `LoadConfig`
 * to point somewhere else.
 */
export const GITHUB_RAW_BASE =
    'https://raw.githubusercontent.com/Kurai-Nova/UFS/main/models';

/** Original tfhub.dev / Kaggle hosted weights, kept as an opt-in fallback. */
export const TFHUB_USE_LITE =
    'https://tfhub.dev/tensorflow/tfjs-model/universal-sentence-encoder-lite/1/default/1';
export const TFHUB_USE_LITE_VOCAB =
    'https://storage.googleapis.com/tfjs-models/savedmodel/universal_sentence_encoder/vocab.json';
export const TFHUB_USE_QNA =
    'https://tfhub.dev/google/tfjs-model/universal-sentence-encoder-qa-ondevice/1';

/**
 * Static profiles for every supported text-embedding model.
 *
 * Scope is intentionally limited to text-embedding models; this is not a
 * generic "run any transformer" registry.
 */
export const MODEL_PROFILES = {
  'use-lite': {
    id: 'use-lite',
    runtime: 'tfjs-graph',
    embeddingDim: 512,
    tokenizer: 'use-sentencepiece',
    inputFormat: 'use-sparse',
    pooling: 'none',
    normalize: false,
  },

  'use-qna': {
    id: 'use-qna',
    runtime: 'tfjs-graph',
    embeddingDim: 100,
    tokenizer: 'use-sentencepiece',
    inputFormat: 'use-sparse',
    pooling: 'none',
    normalize: false,
  },

  'paraphrase-multilingual-minilm': {
    id: 'paraphrase-multilingual-minilm',
    runtime: 'transformers-js',
    embeddingDim: 384,
    hfModelId: 'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
    tokenizer: 'runtime-managed',
    inputFormat: 'runtime-managed',
    pooling: 'mean',
    normalize: true,
  },
} satisfies Record<ModelId, ModelProfile>;

export const DEFAULT_MODEL_ID: ModelId = 'use-lite';

function githubModelUrl(id: ModelId): string {
  return `${GITHUB_RAW_BASE}/${id}/model.json`;
}

function githubVocabUrl(id: ModelId): string {
  return `${GITHUB_RAW_BASE}/${id}/vocab.json`;
}

function assertCustomTfjsUrls(config: LoadConfig): asserts config is LoadConfig & {
  modelUrl: string;
  vocabUrl: string;
} {
  if (config.modelUrl == null || config.vocabUrl == null) {
    throw new Error(
      'weightsSource="custom" requires both modelUrl and vocabUrl for tfjs-graph models.'
    );
  }
}

/**
 * Resolves the effective `ModelProfile` for a given `LoadConfig`, filling in
 * the concrete weight locations according to the requested `weightsSource`
 * and any explicit url overrides.
 */
export function resolveProfile(config: LoadConfig = {}): ModelProfile {
  const id: ModelId = config.model || DEFAULT_MODEL_ID;
  const base = MODEL_PROFILES[id];
  if (base == null) {
    throw new Error(`Unknown model id: ${id}`);
  }

  const profile: ModelProfile = {...base};

  const hasUrlOverride =
      config.modelUrl != null || config.vocabUrl != null;
  const source = config.weightsSource || (hasUrlOverride ? 'custom' : 'github');

  if (profile.runtime === 'tfjs-graph') {
    profile.modelUrl = resolveTfjsModelUrl(id, source, config);
    profile.vocabUrl = resolveTfjsVocabUrl(id, source, config);
  } else {
    profile.hfModelId = config.hfModelId || profile.hfModelId;
  }

  return profile;
}

function resolveTfjsModelUrl(
    id: ModelId, source: string, config: LoadConfig): string {
  if (config.modelUrl != null) {
    return config.modelUrl;
  }
  switch (source) {
      case 'tfhub':
          return id === 'use-qna' ? TFHUB_USE_QNA : TFHUB_USE_LITE;
      case 'github':
          return githubModelUrl(id);
      case 'custom':
          assertCustomTfjsUrls(config);
          return config.modelUrl;
      default:
          return githubModelUrl(id);
  }
}

function resolveTfjsVocabUrl(
    id: ModelId, source: string, config: LoadConfig): string {
  if (config.vocabUrl != null) {
    return config.vocabUrl;
  }
  switch (source) {
    case 'tfhub':
      return id === 'use-qna' ? `${TFHUB_USE_QNA}/vocab.json?tfjs-format=file` : TFHUB_USE_LITE_VOCAB;
    case 'github':
      return githubVocabUrl(id);
    case 'custom':
      assertCustomTfjsUrls(config);
      return config.vocabUrl;
    default:
      return githubVocabUrl(id);
  }
}

/**
 * Whether a `tfjs-graph` model url points at tfhub.dev and therefore needs
 * the `fromTFHub` loader flag.
 */
export function isTfHubUrl(url: string): boolean {
  return url.includes('tfhub.dev');
}
