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

import {load} from '../src/index';
import {TextEmbedder} from '../src/types';

describe('UseLiteEmbedder via load()', () => {
  let embedder: TextEmbedder;
  beforeAll(async () => {
    embedder = await load({weightsSource: 'tfhub'});
  });

  it('implements the TextEmbedder contract', () => {
    expect(embedder.id).toBe('use-lite');
    expect(embedder.embeddingDim).toBe(512);
    expect(embedder.outputType).toBe('tfjs-tensor2d');
  });

  it('embeds a batch into a [N, 512] tensor', async () => {
    const embeddings = await embedder.embed(['I like my phone.', 'Привет.']);
    expect(embeddings.shape).toEqual([2, 512]);
    const values = await embeddings.data();
    expect(values.every(v => Number.isFinite(v))).toBe(true);
    embeddings.dispose();
  });

  it('accepts a single string', async () => {
    const embeddings = await embedder.embed('How old are you?');
    expect(embeddings.shape).toEqual([1, 512]);
    embeddings.dispose();
  });
});
