/**
 * @license
 * Copyright 2019 Google LLC.
 * Modifications Copyright 2026 Nick G.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified from the original Google LLC source.
 * =============================================
 */

import * as tf from '@tensorflow/tfjs-core';

import { resolveProfile } from './profiles';
import { TfjsGraphRuntime } from './runtime/tfjs_graph_runtime';
import { loadVocabulary, UseSentencePieceTokenizer } from './tokenizer/tokenizer';
import { LoadConfig } from './types';

// Index in the vocab file that needs to be skipped.
const SKIP_VALUES = [0, 1, 2];
// Offset value for skipped vocab index.
const OFFSET = 3;
// Input tensor size limit.
const INPUT_LIMIT = 192;
// Model node name for query.
const QUERY_NODE_NAME = 'input_inp_text';
// Model node name for query.
const RESPONSE_CONTEXT_NODE_NAME = 'input_res_context';
// Model node name for response.
const RESPONSE_NODE_NAME = 'input_res_text';
// Model node name for response result.
const RESPONSE_RESULT_NODE_NAME = 'Final/EncodeResult/mul';
// Model node name for query result.
const QUERY_RESULT_NODE_NAME = 'Final/EncodeQuery/mul';
// Reserved symbol count for tokenizer.
const RESERVED_SYMBOLS_COUNT = 3;
// Value for token padding
const TOKEN_PADDING = 2;
// Start value for each token
const TOKEN_START_VALUE = 1;

export interface ModelOutput {
  queryEmbedding: tf.Tensor;
  responseEmbedding: tf.Tensor;
}

export interface ModelInput {
  queries: string[];
  responses: string[];
  contexts?: string[];
}

export async function loadQnA(config: LoadConfig = {}) {
  const qna = new UseQnAEmbedder();
  await qna.load(config);
  return qna;
}

export class UseQnAEmbedder {
  private runtime!: TfjsGraphRuntime;
  private tokenizer!: UseSentencePieceTokenizer;

  async load(config: LoadConfig = {}) {
    const profile = resolveProfile({...config, model: 'use-qna'});
    const runtime = new TfjsGraphRuntime();
    const [, vocabulary] = await Promise.all([
      runtime.load(profile.modelUrl!),
      loadVocabulary(profile.vocabUrl!),
    ]);

    this.runtime = runtime;
    this.tokenizer =
        new UseSentencePieceTokenizer(vocabulary, RESERVED_SYMBOLS_COUNT);
  }

  /**
   *
   * Returns a map of queryEmbedding and responseEmbedding
   *
   * @param input the ModelInput that contains queries and answers.
   */
  embed(input: ModelInput): ModelOutput {
    const embeddings = tf.tidy(() => {
      const queryEncoding = this.tokenizeStrings(input.queries, INPUT_LIMIT);
      const responseEncoding =
          this.tokenizeStrings(input.responses, INPUT_LIMIT);
      if (input.contexts != null) {
        if (input.contexts.length !== input.responses.length) {
          throw new Error(
              'The length of response strings ' +
              'and context strings need to match.');
        }
      }
      const contexts: string[] = input.contexts || [];
      if (input.contexts == null) {
        contexts.length = input.responses.length;
        contexts.fill('');
      }
      const contextEncoding = this.tokenizeStrings(contexts, INPUT_LIMIT);
      const modelInputs: {[key: string]: tf.Tensor} = {};
      modelInputs[QUERY_NODE_NAME] = queryEncoding;
      modelInputs[RESPONSE_NODE_NAME] = responseEncoding;
      modelInputs[RESPONSE_CONTEXT_NODE_NAME] = contextEncoding;

      return this.runtime.execute(
          modelInputs, [QUERY_RESULT_NODE_NAME, RESPONSE_RESULT_NODE_NAME]);
    });
    const queryEmbedding = embeddings[0];
    const responseEmbedding = embeddings[1];

    return {queryEmbedding, responseEmbedding};
  }

  dispose(): void {
    if (this.runtime != null) {
      this.runtime.dispose();
    }
  }

  private tokenizeStrings(strs: string[], limit: number): tf.Tensor2D {
    const tokens =
        strs.map(s => this.shiftTokens(this.tokenizer.encode(s), limit));
    return tf.tensor2d(tokens, [strs.length, limit], 'int32');
  }

  private shiftTokens(tokens: number[], limit: number): number[] {
    tokens.unshift(TOKEN_START_VALUE);
    for (let index = 0; index < limit; index++) {
      if (index >= tokens.length) {
        tokens[index] = TOKEN_PADDING;
      } else if (!SKIP_VALUES.includes(tokens[index])) {
        tokens[index] += OFFSET;
      }
    }
    return tokens.slice(0, limit);
  }
}
