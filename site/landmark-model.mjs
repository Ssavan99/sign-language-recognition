// Landmark classifier for the browser.
//
// The convolutional model reads pixels, so it carries every capture condition of
// the corpus it was trained on. This one reads twenty-one hand keypoints, which
// describe shape and nothing else. On the held-out external capture set that is
// the difference between 31.67% and 84.62%.
//
// The normalisation below must match src/asl_recognition/landmarks.py exactly.
// Landmark features are only meaningful under the transform that produced them,
// so tools/check_landmark_model.mjs asserts agreement against a vector generated
// by the Python side.

export function normalizeLandmarks(points, handedness, indices) {
  const { wrist, middleMcp, indexMcp, ringMcp, pinkyMcp, fingertips } = indices;

  const coordinates = points.map((point) => [point.x, point.y, point.z]);
  if (coordinates.length !== 21) {
    throw new Error(`expected 21 landmarks, got ${coordinates.length}`);
  }

  // 1. Mirror left hands onto right-hand form. The same letter made with either
  //    hand is the same sign, but mirrored coordinates.
  if (typeof handedness === 'string' && handedness.trim().toLowerCase().startsWith('l')) {
    for (const coordinate of coordinates) coordinate[0] = -coordinate[0];
  }

  // 2. Translate so the wrist is the origin.
  const origin = coordinates[wrist].slice();
  const centred = coordinates.map((point) => point.map((axis, index) => axis - origin[index]));

  // 3. Scale so the furthest landmark from the wrist sits at distance 1.
  let span = 0;
  for (const point of centred) {
    span = Math.max(span, Math.hypot(point[0], point[1], point[2]));
  }
  if (span <= 1e-9) throw new Error('degenerate landmarks');
  const scaled = centred.map((point) => point.map((axis) => axis / span));

  // 4. Rotate in XY so the middle-finger knuckle points up.
  const referenceX = scaled[middleMcp][0];
  const referenceY = scaled[middleMcp][1];
  const magnitude = Math.hypot(referenceX, referenceY);
  let rotated = scaled;
  if (magnitude > 1e-9) {
    const cosine = -referenceY / magnitude;
    const sine = -referenceX / magnitude;
    rotated = scaled.map((point) => [
      point[0] * cosine - point[1] * sine,
      point[0] * sine + point[1] * cosine,
      point[2],
    ]);
  }

  const features = [];
  for (const point of rotated) features.push(point[0], point[1], point[2]);

  const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  const thumbTip = rotated[fingertips[0]];
  for (const tip of fingertips) {
    const point = rotated[tip];
    features.push(Math.hypot(point[0], point[1], point[2]));
  }
  for (let index = 1; index < fingertips.length; index += 1) {
    features.push(distance(rotated[fingertips[index]], thumbTip));
  }
  for (let index = 1; index < fingertips.length - 1; index += 1) {
    features.push(distance(rotated[fingertips[index]], rotated[fingertips[index + 1]]));
  }
  const knuckles = [indexMcp, middleMcp, ringMcp, pinkyMcp];
  for (let index = 0; index < knuckles.length; index += 1) {
    features.push(distance(rotated[fingertips[index + 1]], rotated[knuckles[index]]));
  }
  return features;
}

export class LandmarkClassifier {
  constructor(manifest, weights) {
    this.manifest = manifest;
    this.classNames = manifest.class_names;
    this.featureDimension = manifest.feature_dimension;
    this.indices = {
      wrist: manifest.landmark_indices.wrist,
      indexMcp: manifest.landmark_indices.index_mcp,
      middleMcp: manifest.landmark_indices.middle_mcp,
      ringMcp: manifest.landmark_indices.ring_mcp,
      pinkyMcp: manifest.landmark_indices.pinky_mcp,
      fingertips: manifest.landmark_indices.fingertips,
    };
    this.tensors = new Map();
    for (const entry of manifest.tensors) {
      this.tensors.set(entry.name, weights.subarray(entry.offset, entry.offset + entry.length));
    }
  }

  static async load(manifestUrl, weightsUrl) {
    const [manifest, buffer] = await Promise.all([
      fetch(manifestUrl).then((response) => response.json()),
      fetch(weightsUrl).then((response) => response.arrayBuffer()),
    ]);
    const weights = new Float32Array(buffer);
    if (weights.length !== manifest.float_count) {
      throw new Error(
        `weight file holds ${weights.length} floats, manifest expects ${manifest.float_count}`,
      );
    }
    return new LandmarkClassifier(manifest, weights);
  }

  tensor(name) {
    const value = this.tensors.get(name);
    if (!value) throw new Error(`missing tensor ${name}`);
    return value;
  }

  // The exported network is Linear -> BatchNorm -> ReLU -> Dropout repeated, then
  // a final Linear. Dropout is identity at inference, so it has no weights and no
  // step here.
  forward(features) {
    if (features.length !== this.featureDimension) {
      throw new Error(`expected ${this.featureDimension} features, got ${features.length}`);
    }
    let activations = Float32Array.from(features);
    const linearIndices = [];
    for (const entry of this.manifest.tensors) {
      const match = /^network\.(\d+)\.weight$/.exec(entry.name);
      if (match && entry.shape.length === 2) linearIndices.push(Number(match[1]));
    }
    linearIndices.sort((a, b) => a - b);

    linearIndices.forEach((layerIndex, position) => {
      activations = this.linear(activations, layerIndex);
      const isFinal = position === linearIndices.length - 1;
      if (!isFinal) {
        activations = this.batchNormRelu(activations, layerIndex + 1);
      }
    });
    return this.softmax(activations);
  }

  linear(input, layerIndex) {
    const weight = this.tensor(`network.${layerIndex}.weight`);
    const bias = this.tensor(`network.${layerIndex}.bias`);
    const outputSize = bias.length;
    const inputSize = input.length;
    const output = new Float32Array(outputSize);
    for (let row = 0; row < outputSize; row += 1) {
      let total = bias[row];
      const base = row * inputSize;
      for (let column = 0; column < inputSize; column += 1) {
        total += weight[base + column] * input[column];
      }
      output[row] = total;
    }
    return output;
  }

  batchNormRelu(input, layerIndex) {
    const gamma = this.tensor(`network.${layerIndex}.weight`);
    const beta = this.tensor(`network.${layerIndex}.bias`);
    const mean = this.tensor(`network.${layerIndex}.running_mean`);
    const variance = this.tensor(`network.${layerIndex}.running_var`);
    const output = new Float32Array(input.length);
    for (let index = 0; index < input.length; index += 1) {
      const scale = gamma[index] / Math.sqrt(variance[index] + 1e-5);
      const value = (input[index] - mean[index]) * scale + beta[index];
      output[index] = value > 0 ? value : 0;
    }
    return output;
  }

  softmax(logits) {
    let largest = -Infinity;
    for (const value of logits) largest = Math.max(largest, value);
    let total = 0;
    const probabilities = new Float32Array(logits.length);
    for (let index = 0; index < logits.length; index += 1) {
      const value = Math.exp(logits[index] - largest);
      probabilities[index] = value;
      total += value;
    }
    for (let index = 0; index < probabilities.length; index += 1) probabilities[index] /= total;
    return probabilities;
  }

  classify(points, handedness) {
    const features = normalizeLandmarks(points, handedness, this.indices);
    const probabilities = this.forward(features);
    const ranked = Array.from(probabilities, (probability, index) => ({
      label: this.classNames[index],
      probability,
    })).sort((a, b) => b.probability - a.probability);
    return { ranked, top: ranked[0] };
  }
}
