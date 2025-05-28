const tf = require("@tensorflow/tfjs");
const sharp = require("sharp");
const { Pinecone } = require("@pinecone-database/pinecone");

// Konfigurasi Pinecone
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_APIKEY,
});
const index = pinecone.Index("skin-cancer-qa");

const classNames = [
  "Actinic Keratoses (AK)",
  "Basal Cell Carcinoma (BCC)",
  "Benign Keratosis-like Lesions (BKL)",
  "Dermatofibroma (DF)",
  "Melanoma (Mel)",
  "Melanocytic Nevi (NV)",
  "Vascular Lesions (VASC)",
];

let model = null;
let embedder = null;

(async () => {

  if (!model || !embedder) {
    try {
      model = await tf.loadGraphModel(process.env.MODEL_ENDPOINT);
      console.log("✅ Classifier model loaded");
    } catch (err) {
      console.error("Classifier error:", err);
    }

    try {
      const { pipeline } = await import("@xenova/transformers");

      console.log("Loads embedding model...");

      embedder = await pipeline(
        "feature-extraction",
        "Xenova/paraphrase-multilingual-MiniLM-L12-v2"
      );

      console.log("✅ Embedding model loaded");
    } catch (err) {
      console.error("Embedder error:", err);
    }
  }

  // return { model: model, embedder: embedder };
})();


// Fungsi retrieve dari Pinecone
async function retrieveRelevantDocsFromPinecone(query) {
  if (!embedder) throw new Error("Embedder is not ready");

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





async function chat(image, query) {
  /*
    error: ["Confidence score really low, cannot classified image", "Cannot processed image", "RAG error"]
  */

  let className = null;
  let confidence = null;
  let randomContext = null;
  
  const imageBase64 = image.replace(/^data:image\/\w+;base64,/, ""); // Hapus prefix base64
  const imageBuffer = Buffer.from(imageBase64, "base64");

  let data = {};

  if (imageBuffer) {
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

      confidence = (result[result.indexOf(Math.max(...result))] * 100).toFixed(
        2
      );
      if (confidence < 0.5)
        throw new Error("Confidence score really low, cannot classified image");

      className = classNames[result.indexOf(Math.max(...result))];

      if (query) {
        query = `${className} ${query}`;
      } else {
        query = `apa itu ${className}`;
      }

      data = { ...data, class_name: className, confidence: confidence };
    } catch (err) {
      console.log({ error: err });
      throw new Error("Cannot processed image");
    }
  }

  if (query) {
    if (query.length <= 7)
      return {
        assistant:
          "Konteks terlalu kecil, tidak menemukan informasi yang relevan",
      };

    try {
      const docs = await retrieveRelevantDocsFromPinecone(query);

      randomContext = docs[Math.floor(Math.random() * docs.length)];
      randomContext =
        randomContext.score > 0.5
          ? randomContext?.content
          : "Tidak ada hasil relevan";

      data = { ...data, assistant: randomContext };
    } catch (err) {
      console.error({ error: err });
      throw new Error("RAG error");
    }
  }

  if (data) return data;
}

module.exports = {
  chat,
};
