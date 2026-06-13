import * as tf from '@tensorflow/tfjs-core';

import { UseSentencePieceTokenizer } from '../tokenizer/tokenizer';
import { TokenizerAdapter } from '../types';

/**
 * Sparse encoding consumed by the USE lite GraphModel: a `[nnz, 2]` matrix of
 * `[rowIndex, columnIndex]` pairs and a parallel `[nnz]` vector of token ids.
 */
export interface UseSparseEncoding {
  indices: tf.Tensor2D;
  values: tf.Tensor1D;
}

/**
 * Turns a batch of strings into the sparse `{indices, values}` representation
 * the USE lite GraphModel expects. This was previously inlined inside
 * `UniversalSentenceEncoder.embed()`; it is now isolated so the runtime and
 * tokenization concerns are separated.
 */
export class UseSparseInputAdapter implements
    TokenizerAdapter<UseSparseEncoding> {
  constructor(private readonly tokenizer: UseSentencePieceTokenizer) {}

  async encodeBatch(texts: string[]): Promise<UseSparseEncoding> {
    const encodings = texts.map(d => this.tokenizer.encode(d));

    const indicesArr =
        encodings.map((arr, i) => arr.map((_, index) => [i, index]));

    let flattenedIndicesArr: Array<[number, number]> = [];
    for (let i = 0; i < indicesArr.length; i++) {
      flattenedIndicesArr = flattenedIndicesArr.concat(indicesArr[i] as Array<[number, number]>);
    }

    const indices = tf.tensor2d(flattenedIndicesArr, [flattenedIndicesArr.length, 2], 'int32');
    const values = tf.tensor1d(tf.util.flatten(encodings), 'int32');

    return {indices, values};
  }
}
