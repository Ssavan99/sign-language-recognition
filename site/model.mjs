export class BrowserASLModel {
  constructor(manifest, weights) {
    this.manifest = manifest;
    this.weights = weights;
    this.classes = manifest.class_names;
    this.size = manifest.image_size;
    this.tensorIndex = new Map(manifest.tensors.map((tensor) => [tensor.name, tensor]));
  }

  static async load(basePath = "assets") {
    const [manifestResponse, weightsResponse] = await Promise.all([
      fetch(`${basePath}/browser-model-manifest.json`),
      fetch(`${basePath}/asl-alphabet-cnn-v1.f32`),
    ]);
    if (!manifestResponse.ok || !weightsResponse.ok) {
      throw new Error("The browser model files could not be loaded.");
    }
    const manifest = await manifestResponse.json();
    const weightsBuffer = await weightsResponse.arrayBuffer();
    return BrowserASLModel.fromBuffers(manifest, weightsBuffer);
  }

  static fromBuffers(manifest, weightsBuffer) {
    if (manifest.format !== "asl-browser-cnn-v1") {
      throw new Error("Unsupported browser model format.");
    }
    const weights = new Float32Array(weightsBuffer);
    if (weights.length !== manifest.float_count) {
      throw new Error("Browser model weight count does not match its manifest.");
    }
    return new BrowserASLModel(manifest, weights);
  }

  tensor(name) {
    const descriptor = this.tensorIndex.get(name);
    if (!descriptor) throw new Error(`Missing model tensor: ${name}`);
    return this.weights.subarray(descriptor.offset, descriptor.offset + descriptor.length);
  }

  convBlock(input, inChannels, outChannels, height, width, prefix) {
    let output = conv3x3(input, this.tensor(`${prefix}.0.weight`), inChannels, outChannels, height, width);
    output = batchNormRelu(output, this.tensor(`${prefix}.1.weight`), this.tensor(`${prefix}.1.bias`), this.tensor(`${prefix}.1.running_mean`), this.tensor(`${prefix}.1.running_var`));
    output = conv3x3(output, this.tensor(`${prefix}.3.weight`), outChannels, outChannels, height, width);
    output = batchNormRelu(output, this.tensor(`${prefix}.4.weight`), this.tensor(`${prefix}.4.bias`), this.tensor(`${prefix}.4.running_mean`), this.tensor(`${prefix}.4.running_var`));
    return maxPool2(output, outChannels, height, width);
  }

  predict(input) {
    if (!(input instanceof Float32Array) || input.length !== 3 * this.size * this.size) {
      throw new Error("Expected a normalized 3×64×64 Float32 input.");
    }
    let value = this.convBlock(input, 3, 24, 64, 64, "features.0");
    value = this.convBlock(value, 24, 48, 32, 32, "features.1");
    value = this.convBlock(value, 48, 96, 16, 16, "features.2");
    const pooled = globalAverage(value, 96, 8, 8);
    const logits = linear(pooled, this.tensor("classifier.2.weight"), this.tensor("classifier.2.bias"), 26, 96);
    return softmax(logits);
  }

  topK(probabilities, count = 3) {
    return [...probabilities]
      .map((probability, index) => ({ label: this.classes[index], probability }))
      .sort((left, right) => right.probability - left.probability)
      .slice(0, count);
  }
}

function conv3x3(input, weights, inChannels, outChannels, height, width) {
  const plane = height * width;
  const output = new Float32Array(outChannels * plane);
  for (let outputChannel = 0; outputChannel < outChannels; outputChannel += 1) {
    const outputOffset = outputChannel * plane;
    const filterOffset = outputChannel * inChannels * 9;
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        for (let inputChannel = 0; inputChannel < inChannels; inputChannel += 1) {
          const inputOffset = inputChannel * plane;
          const kernelOffset = filterOffset + inputChannel * 9;
          for (let kernelY = 0; kernelY < 3; kernelY += 1) {
            const sourceY = y + kernelY - 1;
            if (sourceY < 0 || sourceY >= height) continue;
            const rowOffset = inputOffset + sourceY * width;
            const weightRowOffset = kernelOffset + kernelY * 3;
            for (let kernelX = 0; kernelX < 3; kernelX += 1) {
              const sourceX = x + kernelX - 1;
              if (sourceX >= 0 && sourceX < width) {
                sum += input[rowOffset + sourceX] * weights[weightRowOffset + kernelX];
              }
            }
          }
        }
        output[outputOffset + y * width + x] = sum;
      }
    }
  }
  return output;
}

function batchNormRelu(values, gamma, beta, mean, variance) {
  const plane = values.length / gamma.length;
  for (let channel = 0; channel < gamma.length; channel += 1) {
    const scale = gamma[channel] / Math.sqrt(variance[channel] + 1e-5);
    const shift = beta[channel] - mean[channel] * scale;
    const offset = channel * plane;
    for (let index = 0; index < plane; index += 1) {
      values[offset + index] = Math.max(0, values[offset + index] * scale + shift);
    }
  }
  return values;
}

function maxPool2(input, channels, height, width) {
  const outputHeight = height / 2;
  const outputWidth = width / 2;
  const inputPlane = height * width;
  const outputPlane = outputHeight * outputWidth;
  const output = new Float32Array(channels * outputPlane);
  for (let channel = 0; channel < channels; channel += 1) {
    const inputOffset = channel * inputPlane;
    const outputOffset = channel * outputPlane;
    for (let y = 0; y < outputHeight; y += 1) {
      for (let x = 0; x < outputWidth; x += 1) {
        const source = inputOffset + (y * 2) * width + x * 2;
        output[outputOffset + y * outputWidth + x] = Math.max(input[source], input[source + 1], input[source + width], input[source + width + 1]);
      }
    }
  }
  return output;
}

function globalAverage(input, channels, height, width) {
  const plane = height * width;
  const output = new Float32Array(channels);
  for (let channel = 0; channel < channels; channel += 1) {
    let sum = 0;
    const offset = channel * plane;
    for (let index = 0; index < plane; index += 1) sum += input[offset + index];
    output[channel] = sum / plane;
  }
  return output;
}

function linear(input, weights, bias, outFeatures, inFeatures) {
  const output = new Float32Array(outFeatures);
  for (let outputIndex = 0; outputIndex < outFeatures; outputIndex += 1) {
    let sum = bias[outputIndex];
    const offset = outputIndex * inFeatures;
    for (let inputIndex = 0; inputIndex < inFeatures; inputIndex += 1) sum += weights[offset + inputIndex] * input[inputIndex];
    output[outputIndex] = sum;
  }
  return output;
}

function softmax(logits) {
  let maximum = -Infinity;
  for (const value of logits) maximum = Math.max(maximum, value);
  const output = new Float32Array(logits.length);
  let total = 0;
  for (let index = 0; index < logits.length; index += 1) {
    output[index] = Math.exp(logits[index] - maximum);
    total += output[index];
  }
  for (let index = 0; index < output.length; index += 1) output[index] /= total;
  return output;
}
