/**
 * @license
 * Copyright 2019 Google LLC.
 * Modifications Copyright 2026 Nick G.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified from the original Google LLC source.
 *
 * =============================================
 */

import {stubbedTokenizerVocab} from './test_util';
import {UseSentencePieceTokenizer} from '../src/tokenizer/tokenizer';

describe('USE SentencePiece trie', () => {
  let tokenizer: UseSentencePieceTokenizer;
  beforeAll(() => {
    tokenizer = new UseSentencePieceTokenizer(
        stubbedTokenizerVocab as Array<[string, number]>);
  });

  it('Trie creates a child for each unique prefix', () => {
    const childKeys = Object.keys(tokenizer.trie.root.children);
    expect(childKeys).toEqual(['▁', 'a', '.', 'I', 'l', 'i', 'k', 'e', 't']);
  });

  it('Trie commonPrefixSearch basic usage', () => {
    const commonPrefixes =
        tokenizer.trie.commonPrefixSearch(['l', 'i', 'k', 'e'])
            .map(d => d[0].join(''));

    expect(commonPrefixes).toEqual(['l', 'like']);
  });
});
