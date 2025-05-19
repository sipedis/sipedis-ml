const express = require("express");
const bodyParser = require("body-parser");
const { Pinecone } = require("@pinecone-database/pinecone");
const tf = require("@tensorflow/tfjs"); // Better use @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu but it has building isues
const multer = require("multer");
const sharp = require("sharp");
require("dotenv").config();

// Inisialisasi Express
const app = express();
const port = 3000;

// Middleware
app.use(bodyParser.json());

// Konfigurasi Pinecone
const pinecone = new Pinecone({
  apiKey:
    process.env.PINECONE_API
});

const index = pinecone.Index("skin-cancer-qa");

// Inisialisasi model embedding
let embedder;
const upload = multer({ storage: multer.memoryStorage() });
app.use("/model", express.static("model"));

(async () => {
  try {
    model = await tf.loadGraphModel("http://localhost:3000/model/model.json");
    console.log("Classifier model loaded");

  } catch (err) {
    console.error("Classifier error:", err);
  }

  try {
    const { pipeline } = await import("@xenova/transformers");

    console.log("Loads embedding model...")

    embedder = await pipeline(
      "feature-extraction",
      "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
    );

    console.log("Embedding model loaded");

  } catch (err) {
    console.error("Embedder error:", err);
  }

})();

// Fungsi retrieve dari Pinecone
async function retrieveRelevantDocsFromPinecone(query) {
  if (!embedder) throw new Error("Embedder belum siap");

  // Buat query embedding
  const queryEmbedding = await embedder(query, {
    pooling: "mean",
    normalize: true,
  });

  const vector = Object.values(queryEmbedding.data); // ubah dari object ke array

  // Panggil Pinecone
  const result = await index.query({
    vector: vector,
    topK: 3, // panggil 3 teratas
    includeMetadata: true,
  });

  // Ambil dokumen dari metadata
  return result.matches.map((match) => ({
    id: match.id,
    score: match.score,
    content: match.metadata?.text || "Tidak ada konten",
  }));
}

async function chat_image(imageBuffer, res) {
  try {
    // Use sharp to resize the image and convert it to raw pixel data (RGB)
    const image = await sharp(imageBuffer).resize(224, 224).raw().toBuffer();

    // Convert raw image data into a Tensor
    const imgArray = new Uint8Array(image);

    // Check if the image has the correct number of values (224 * 224 * 3)
    if (imgArray.length !== 224 * 224 * 3) {
      return res.status(400).send("Image size is not correct.");
    }

    // Convert to tensor, shape [1, 224, 224, 3]
    let imgTensor = tf.tensor4d(Array.from(imgArray), [1, 224, 224, 3]);
    imgTensor = imgTensor.div(tf.scalar(255));

    const prediction = model.predict(imgTensor);
    const result = await prediction.data();

    const class_names = [
      "Actinic Keratoses (AK)",  
      "Basal Cell Carcinoma (BCC)",  
      "Benign Keratosis-like Lesions (BKL)",  
      "Dermatofibroma (DF)",  
      "Melanoma (Mel)",  
      "Melanocytic Nevi (NV)",  
      "Vascular Lesions (VASC)"
    ];

    const class_name = class_names[result.indexOf(Math.max(...result))];

    const query = `apa itu ${class_name}`

    try {
      const docs = await retrieveRelevantDocsFromPinecone(query);
      const maxIndex = (
        result[result.indexOf(Math.max(...result))] * 100
      ).toFixed(2);

      if (maxIndex < 0.5) res.send("Tidak dapat memprediksi gambar")

      let randomContext = docs[Math.floor(Math.random() * docs.length)];
      randomContext =
        randomContext.score > 0.5
          ? randomContext?.content
          : "Tidak ada hasil relevan";

      res.status(200).json({
        class_name: class_name,
        confidence: maxIndex,
        assistant: randomContext,
      });

    } catch (err) {
      console.error("RAG error:", err);
      res.status(500).send("Server error: " + err.message);
    }

  } catch (err) {
    console.error(err);
    res.status(500).send("Error processing image.");
  }
}

async function chat_query(query, res) {
  if (!query) return res.status(400).send("Query diperlukan");

  if (query.length <= 5)
    return res
      .status(400)
      .send({
        assistant:
          "Konteks terlalu kecil, tidak menemukan informasi yang relevan",
      });

  try {
    const docs = await retrieveRelevantDocsFromPinecone(
      query.replace(/[^\w\s]/g, "").toLowerCase()
    );
    // res.send(docs)

    let randomContext = docs[Math.floor(Math.random() * docs.length)];
    randomContext =
      randomContext.score > 0.5
        ? randomContext?.content
        : "Tidak ada hasil relevan";

    res.send({ assistant: randomContext });
  } catch (err) {
    console.error("RAG error:", err);
    res.status(500).send("Server error: " + err.message);
  }
}

async function chat_all(imageBuffer, query, res){
  // Use sharp to resize the image and convert it to raw pixel data (RGB)
  const image = await sharp(imageBuffer).resize(224, 224).raw().toBuffer();

  // Convert raw image data into a Tensor
  const imgArray = new Uint8Array(image);

  // Check if the image has the correct number of values (224 * 224 * 3)
  if (imgArray.length !== 224 * 224 * 3) {
    return res.status(400).send("Image size is not correct.");
  }

  // Convert to tensor, shape [1, 224, 224, 3]
  let imgTensor = tf.tensor4d(Array.from(imgArray), [1, 224, 224, 3]);
  imgTensor = imgTensor.div(tf.scalar(255));

  const prediction = model.predict(imgTensor);
  const result = await prediction.data();

  const class_names = [
    "Actinic Keratoses (AK)",
    "Basal Cell Carcinoma (BCC)",
    "Benign Keratosis-like Lesions (BKL)",
    "Dermatofibroma (DF)",
    "Melanoma (Mel)",
    "Melanocytic Nevi (NV)",
    "Vascular Lesions (VASC)",
  ];

  const class_name = class_names[result.indexOf(Math.max(...result))];
  query = `${class_name} ${query}`;

  try {
    const docs = await retrieveRelevantDocsFromPinecone(query);
    const maxIndex = (
      result[result.indexOf(Math.max(...result))] * 100
    ).toFixed(2);

    if (maxIndex < 0.5) return res.send("Tidak dapat memprediksi gambar");

    let randomContext = docs[Math.floor(Math.random() * docs.length)];
    randomContext =
      randomContext.score > 0.5
        ? randomContext?.content
        : "Tidak ada hasil relevan";

    res.status(200).json({
      class_name: class_name,
      confidence: maxIndex,
      assistant: randomContext,
    });
  } catch (err) {
    console.error("RAG error:", err);
    res.status(500).send("Server error: " + err.message);
  }
}


// Endpoint Retriever dengan gambar dalam format JSON (base64)
app.post("/chat", async (req, res) => {
  try {

    if (req.body.image && req.body.query) {

      const imageBase64 = req.body.image.replace(/^data:image\/\w+;base64,/, ""); // Hapus prefix base64
      const imageBuffer = Buffer.from(imageBase64, 'base64');
      let query = req.body.query
  
      await chat_all(imageBuffer, query, res);
    } 
    
    
    else if (req.body.image && !req.body.query) {

      const imageBase64 = req.body.image.replace(/^data:image\/\w+;base64,/, ""); // Hapus prefix base64
      const imageBuffer = Buffer.from(imageBase64, "base64");

      await chat_image(imageBuffer, res);
    }

    
    else if (!req.body.image && req.body.query) {

      const query = req.body.query;

      await chat_query(query, res);
    }

    
    else {
      return res.status(400).send("No image or query data provided.");
    }
  } 
  
  catch (err) {
    console.error(err);
    res.status(500).send("Error processing data");
  }

});

app.listen(port, () => {
  console.log(`Server jalan di http://localhost:${port}`);
});
