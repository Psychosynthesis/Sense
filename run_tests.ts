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


// Use the CPU backend for running tests.
import '@tensorflow/tfjs-backend-cpu';
import * as tf from '@tensorflow/tfjs-core';

const jasmineCtor = require('jasmine');

Error.stackTraceLimit = Infinity;

process.on('unhandledRejection', e => {
  throw e;
});

const unitTests = 'test/**/*_test.ts';

tf.setBackend('cpu').then(() => {
  const runner = new jasmineCtor();
  runner.loadConfig({spec_files: [unitTests], random: false});
  runner.execute();
});
