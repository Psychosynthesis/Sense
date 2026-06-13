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

describe('USE SentencePiece tokenizer', () => {
  let tokenizer: UseSentencePieceTokenizer;
  beforeAll(() => {
    tokenizer = new UseSentencePieceTokenizer(
        stubbedTokenizerVocab as Array<[string, number]>);
  });

  it('basic usage', () => {
    expect(tokenizer.encode('Ilikeit.')).toEqual([11, 15, 16, 10]);
  });

  it('handles whitespace', () => {
    expect(tokenizer.encode('I like it.')).toEqual([11, 12, 13, 10]);
  });

  it('should normalize inputs', () => {
    expect(tokenizer.encode('ça')).toEqual(tokenizer.encode('c\u0327a'));
  });

  it('should handle unknown inputs', () => {
    expect(() => tokenizer.encode('😹')).not.toThrow();
  });

  it('should treat consecutive unknown inputs as a single word', () => {
    expect(tokenizer.encode('a😹😹')).toEqual([7, 0]);
  });
});
