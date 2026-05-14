tf.setBackend("cpu");
let model;
const trainButton = document.getElementById("trainButton");
const predictButton = document.getElementById("predictButton");
const predictionResult = document.querySelector(".prediction-result");

trainButton.addEventListener("click", async () => {
  trainButton.innerText = "Training...";

  const response = await fetch("data/titanic.json");

  const dataSet = await response.json();
  const inputs = dataSet.map((data) => {
    const sexNumeric = data.sex === "female" ? 1 : 0;
    return [Number(data.age), sexNumeric, Number(data.pclass)];
  });

  const labels = dataSet.map((data) => [Number(data.survived)]);
  const xs = tf.tensor2d(inputs);
  const ys = tf.tensor2d(labels);

  model = tf.sequential();

  model.add(
    tf.layers.dense({ units: 10, inputShape: [3], activation: "relu" }),
  );

  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.05),

    loss: "binaryCrossentropy",

    metrics: ["accuracy"],
  });

  await model.fit(xs, ys, {
    epochs: 50,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        const currentAcc =
          logs.accuracy !== undefined ? logs.accuracy : logs.acc;
        console.log(`Epoch ${epoch + 1}: accuracy = ${currentAcc.toFixed(2)}`);
      },
    },
  });

  trainButton.innerText = "Model Ready!";
  predictButton.disabled = false;
  predictButton.style.display = "block";
});

predictButton.addEventListener("click", () => {
  predictionResult.style.display = "block";
  if (!model) {
    alert("Please train the model first!");
    return;
  }
  const userAge = parseFloat(document.getElementById("ageInput").value);
  const userGender =
    document.getElementById("sexSelect").value === "female" ? 1 : 0;
  const userPclass = parseInt(document.getElementById("pclassSelect").value);

  if (
    isNaN(userAge) ||
    isNaN(userPclass) ||
    document.getElementById("sexSelect").value === ""
  ) {
    alert("Please fill in all fields!");
    return;
  }
  const predictionInput = tf.tensor2d([[userAge, userGender, userPclass]]);
  const prediction = model.predict(predictionInput);

  const probability = (prediction.dataSync()[0] * 100).toFixed(2);
  document.getElementById("predictionResult").textContent =
    `Survival Probability: %${probability}`;
});
