// Assert the browser landmark classifier agrees with PyTorch.
//
// Two things are checked, because either failing alone would silently produce
// confident nonsense in the browser:
//
//   1. The forward pass, against probabilities exported from PyTorch.
//   2. The JS normaliser, against a feature vector exported from Python. The
//      network could be perfect and still be fed the wrong numbers.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { LandmarkClassifier, normalizeLandmarks } from '../site/landmark-model.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '..');

const manifest = JSON.parse(
  readFileSync(join(root, 'site/assets/landmark-model-manifest.json'), 'utf8'),
);
const buffer = readFileSync(join(root, 'site/assets/asl-landmark-mlp-v1.f32'));
const weights = new Float32Array(
  buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
);
const parity = JSON.parse(readFileSync(join(root, 'tools/landmark_model_parity.json'), 'utf8'));

const classifier = new LandmarkClassifier(manifest, weights);

const actual = classifier.forward(parity.features);
let forwardDifference = 0;
for (let index = 0; index < parity.expected.length; index += 1) {
  forwardDifference = Math.max(forwardDifference, Math.abs(actual[index] - parity.expected[index]));
}

let normaliserDifference = 0;
if (parity.normaliser) {
  const indices = {
    wrist: manifest.landmark_indices.wrist,
    indexMcp: manifest.landmark_indices.index_mcp,
    middleMcp: manifest.landmark_indices.middle_mcp,
    ringMcp: manifest.landmark_indices.ring_mcp,
    pinkyMcp: manifest.landmark_indices.pinky_mcp,
    fingertips: manifest.landmark_indices.fingertips,
  };
  const produced = normalizeLandmarks(
    parity.normaliser.landmarks.map(([x, y, z]) => ({ x, y, z })),
    parity.normaliser.handedness,
    indices,
  );
  if (produced.length !== parity.normaliser.features.length) {
    console.error(
      `LANDMARK_PARITY=fail normaliser produced ${produced.length} features, ` +
        `Python produced ${parity.normaliser.features.length}`,
    );
    process.exit(1);
  }
  for (let index = 0; index < produced.length; index += 1) {
    normaliserDifference = Math.max(
      normaliserDifference,
      Math.abs(produced[index] - parity.normaliser.features[index]),
    );
  }
}

const forwardTolerance = 1e-5;
const normaliserTolerance = 1e-6;
const passed = forwardDifference <= forwardTolerance && normaliserDifference <= normaliserTolerance;

console.log(
  `LANDMARK_PARITY=${passed ? 'pass' : 'fail'} ` +
    `forward_max_difference=${forwardDifference} ` +
    `normaliser_max_difference=${normaliserDifference}`,
);
process.exit(passed ? 0 : 1);
