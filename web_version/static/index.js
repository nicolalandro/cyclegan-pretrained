async function runExample() {
  const session = new onnx.InferenceSession();
  await session.loadModel("static/cezanne.onnx");

  const x = new Float32Array(1 * 3 * 224 * 224).fill(1);
  const tensorX = new onnx.Tensor(x, 'float32', [1, 3, 224, 224]);

  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output');

  const max_val = Math.max.apply(Math, outputData.data);
  const class_id = outputData.data.indexOf(max_val)

  const predictions = document.getElementById('predictions');
  predictions.innerHTML = `Output Prob: ${max_val}, Output Class: ${class_id}`;
}
