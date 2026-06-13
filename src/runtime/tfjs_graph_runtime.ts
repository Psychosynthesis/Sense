import * as tfconv from '@tensorflow/tfjs-converter';
import * as tf from '@tensorflow/tfjs-core';

import { isTfHubUrl } from '../profiles';

/**
 * Thin wrapper around a TensorFlow.js `GraphModel`. Centralises model loading
 * (including the tfhub.dev `fromTFHub` quirk) and execution so embedders do
 * not have to care about loader details.
 */
export class TfjsGraphRuntime {
  private model!: tfconv.GraphModel;

  async load(modelUrl: string): Promise<void> {
    this.model = isTfHubUrl(modelUrl) ?
        await tfconv.loadGraphModel(modelUrl, {fromTFHub: true}) :
        await tfconv.loadGraphModel(modelUrl);
  }

  async executeAsync(inputs: tf.NamedTensorMap): Promise<tf.Tensor> {
    return this.model.executeAsync(inputs) as Promise<tf.Tensor>;
  }

  execute(inputs: tf.NamedTensorMap, outputs: string[]): tf.Tensor[] {
    return this.model.execute(inputs, outputs) as tf.Tensor[];
  }

  get graphModel(): tfconv.GraphModel {
    return this.model;
  }

  dispose(): void {
    if (this.model != null) {
      this.model.dispose();
    }
  }
}
