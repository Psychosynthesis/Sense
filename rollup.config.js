import { readFileSync } from 'node:fs';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';
import replace from '@rollup/plugin-replace';

const pkg = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8')
);

const PREAMBLE = `////// Build Sense — Universal text embedding library //////`;

function minify() {
  return terser({ format: { preamble: PREAMBLE }});
}

function injectVersion() {
  return replace({
    preventAssignment: true,
    values: { __SENSE_PKG_VERSION__: pkg.version },
  });
}

function config({ plugins = [], output = {} }) {
  return {
    input: 'src/index.ts',
    plugins: [
      typescript({
        tsconfig: 'tsconfig.json',
        declaration: false,
        declarationMap: false,
      }),
      injectVersion(),
      nodeResolve(),
      ...plugins
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        '@tensorflow/tfjs-core': 'tf',
        '@tensorflow/tfjs-converter': 'tf',
      },
      ...output,
    },
    external: [
      '@tensorflow/tfjs-core',
      '@tensorflow/tfjs-converter',
      '@huggingface/transformers',
    ]
  };
}

export default [
  config({
    output: {
      format: 'umd',
      name: 'sense',
      file: 'dist/sense.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'umd',
      name: 'sense',
      file: 'dist/sense.min.js'
    }
  }),
  config({
    plugins: [minify()],
    output: {
      format: 'es',
      file: 'dist/sense.esm.js'
    }
  })
];
