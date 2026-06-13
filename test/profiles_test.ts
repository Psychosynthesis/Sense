import {MODEL_PROFILES, resolveProfile} from '../src/profiles';

describe('model profiles', () => {
  it('defaults to use-lite served from github', () => {
    const profile = resolveProfile();
    expect(profile.id).toBe('use-lite');
    expect(profile.runtime).toBe('tfjs-graph');
    expect(profile.embeddingDim).toBe(512);
    expect(profile.modelUrl).toContain('githubusercontent.com');
    expect(profile.modelUrl).toContain('use-lite/model.json');
    expect(profile.vocabUrl).toContain('use-lite/vocab.json');
  });

  it('resolves the multilingual MiniLM profile', () => {
    const profile = resolveProfile({model: 'paraphrase-multilingual-minilm'});
    expect(profile.runtime).toBe('transformers-js');
    expect(profile.embeddingDim).toBe(384);
    expect(profile.pooling).toBe('mean');
    expect(profile.normalize).toBe(true);
    expect(profile.hfModelId)
        .toBe('Xenova/paraphrase-multilingual-MiniLM-L12-v2');
  });

  it('switches the USE source to tfhub on request', () => {
    const profile = resolveProfile({weightsSource: 'tfhub'});
    expect(profile.modelUrl).toContain('tfhub.dev');
    expect(profile.vocabUrl).toContain('storage.googleapis.com');
  });

  it('honours explicit url overrides', () => {
    const profile = resolveProfile({
      modelUrl: 'https://example.com/model.json',
      vocabUrl: 'https://example.com/vocab.json',
    });
    expect(profile.modelUrl).toBe('https://example.com/model.json');
    expect(profile.vocabUrl).toBe('https://example.com/vocab.json');
  });

  it('allows overriding the hugging face model id', () => {
    const profile = resolveProfile(
        {model: 'paraphrase-multilingual-minilm', hfModelId: 'Custom/model'});
    expect(profile.hfModelId).toBe('Custom/model');
  });

  it('throws on unknown model ids', () => {
    // oxlint-disable-next-line typescript/no-explicit-any
    expect(() => resolveProfile({model: 'nope' as any})).toThrowError();
  });

  it('exposes a profile for every model id', () => {
    expect(Object.keys(MODEL_PROFILES).sort()).toEqual([
      'paraphrase-multilingual-minilm',
      'use-lite',
      'use-qna',
    ]);
  });
});
