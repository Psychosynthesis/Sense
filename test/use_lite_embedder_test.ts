/**
 * @license
 * Copyright 2024 Kurai-Nova. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
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
