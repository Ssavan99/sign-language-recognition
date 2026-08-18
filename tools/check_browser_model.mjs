import { readFile } from "node:fs/promises";

import { BrowserASLModel } from "../site/model.mjs";

const manifest = JSON.parse(await readFile(new URL("../site/assets/browser-model-manifest.json", import.meta.url)));
const expected = JSON.parse(await readFile(new URL("browser_model_parity.json", import.meta.url)));
const bytes = await readFile(new URL("../site/assets/asl-alphabet-cnn-v1.f32", import.meta.url));
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const model = BrowserASLModel.fromBuffers(manifest, buffer);
const input = new Float32Array(3 * 64 * 64);
let state = expected.seed >>> 0;
for (let index = 0; index < input.length; index += 1) {
  state = (Math.imul(state, 1_664_525) + 1_013_904_223) >>> 0;
  input[index] = (state / 4_294_967_296) * 2 - 1;
}
const actual = model.predict(input);
let maximumDifference = 0;
for (let index = 0; index < actual.length; index += 1) {
  maximumDifference = Math.max(maximumDifference, Math.abs(actual[index] - expected.expected[index]));
}
if (maximumDifference > 1e-5) {
  throw new Error(`Browser model diverged from PyTorch: ${maximumDifference}`);
}
console.log(`BROWSER_MODEL_PARITY=pass max_difference=${maximumDifference}`);
