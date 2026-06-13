import { TransformersJsEmbedder } from './embedders/transformers_js_embedder';
import { UseLiteEmbedder } from './embedders/use_lite_embedder';
import { LoadConfig, ModelProfile, TextEmbedder } from './types';

// Builds and loads the right `TextEmbedder` for a resolved `ModelProfile`.
export async function createEmbedder(
    profile: ModelProfile, config: LoadConfig = {}): Promise<TextEmbedder> {
  switch (profile.runtime) {
    case 'tfjs-graph': {
      const embedder = new UseLiteEmbedder(profile);
      await embedder.load();
      return embedder;
    }

    case 'transformers-js': {
      const embedder = new TransformersJsEmbedder(profile, config);
      await embedder.load();
      return embedder;
    }

    default:
      throw new Error(`Unsupported runtime: ${profile.runtime}`);
  }
}
