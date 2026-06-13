import * as tf from '@tensorflow/tfjs-core';

import { TfjsGraphRuntime } from '../runtime/tfjs_graph_runtime';
import { loadVocabulary, UseSentencePieceTokenizer } from '../tokenizer/tokenizer';
import { ModelProfile, TextEmbedder } from '../types';

import { UseSparseInputAdapter } from './use_sparse_input';

/**
 * USE lite embedder. Wraps the original Universal Sentence Encoder lite
 * GraphModel behind the generic `TextEmbedder` contract without changing its
 * numerical behaviour.
 *
 * Responsibilities are split across collaborators:
 *  - `UseSentencePieceTokenizer` performs SentencePiece segmentation.
 *  - `UseSparseInputAdapter` builds the sparse `{indices, values}` inputs.
 *  - `TfjsGraphRuntime` loads and runs the GraphModel.
 */
export class UseLiteEmbedder implements TextEmbedder {
  readonly id: string;
  readonly embeddingDim: number;
  readonly outputType = 'tfjs-tensor2d' as const;

  private runtime!: TfjsGraphRuntime;
  private tokenizer!: UseSentencePieceTokenizer;
  private inputAdapter!: UseSparseInputAdapter;

  constructor(private readonly profile: ModelProfile) {
    this.id = profile.id;
    this.embeddingDim = profile.embeddingDim;
  }

  async load(): Promise<void> {
    const runtime = new TfjsGraphRuntime();
    const [, vocabulary] = await Promise.all([
      runtime.load(this.profile.modelUrl!),
      loadVocabulary(this.profile.vocabUrl!),
    ]);

    this.runtime = runtime;
    this.tokenizer = new UseSentencePieceTokenizer(vocabulary);
    this.inputAdapter = new UseSparseInputAdapter(this.tokenizer);
  }

  /**
   * Returns a 2D Tensor of shape `[inputs.length, embeddingDim]` that contains
   * the Universal Sentence Encoder embeddings for each input.
   *
   * @param inputs A string or an array of strings to embed.
   */
  async embed(inputs: string[] | string): Promise<tf.Tensor2D> {
    const texts = typeof inputs === 'string' ? [inputs] : inputs;
    const { indices, values } = await this.inputAdapter.encodeBatch(texts);

    try {
      const embeddings = await this.runtime.executeAsync({ indices, values });
      return embeddings as tf.Tensor2D;
    } finally {
      indices.dispose();
      values.dispose();
    }
  }

  dispose(): void {
    if (this.runtime != null) {
      this.runtime.dispose();
    }
  }
}
