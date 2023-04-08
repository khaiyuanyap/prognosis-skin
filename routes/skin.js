const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const sharp = require('sharp');

const upload = multer({ dest: 'uploads/' });


const TARGET_CLASSES = {
  0: "Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)",
  1: "Basal Cell Carcinoma",
  2: "Benign Keratosis",
  3: "Dermatofibroma",
  4: "Melanoma",
  5: "Melanocytic Nevi",
  6: "Vascular skin lesion"
};

// Load the model only once when the server starts
const modelPromise = tf.loadLayersModel("https://raw.githubusercontent.com/khaiyuanyap/prognosis/main/public/model/model.json");

const predict = async (imageBuffer, model) => {
  // Pre-process the image without the browser
  let tensor = tf.node.decodeImage(imageBuffer, 3);

  let offset = tf.scalar(127.5);

  tensor = tensor.sub(offset).div(offset).expandDims();

  let predictions = await model.predict(tensor).data();

  let top3 = Array.from(predictions)
    .map(function (p, i) {
      // this is Array.map
      return {
        probability: p,
        className: TARGET_CLASSES[i] // we are selecting the value from the obj
      }
    })
    .sort(function (a, b) {
      return b.probability - a.probability
    })
    .slice(0, 3);

  console.log(top3);
  return top3;
}

/* POST image and get prediction */
const router = express.Router();

router.post('/', upload.single('file'), async (req, res) => {
  try {
    // Load the model from the promise
    const model = await modelPromise;
    // Get the image file
    const imageFileRef = req.file.path;

    // Resize and convert image to PNG
    const image = await sharp(imageFileRef)
      .resize(224, 224)
      .toFormat('png') // Convert to PNG format
      .toBuffer();

    // Make a prediction
    const prediction = await predict(image, model);

    // Delete the file after prediction
    fs.unlinkSync(imageFileRef);

    res.send(prediction);
  } catch (error) {
    console.error(error);
    res.status(500).send('Error occurred during prediction');
  }
});

module.exports = router;
