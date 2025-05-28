const express = require("express");
const bodyParser = require("body-parser");
const tf = require("@tensorflow/tfjs"); // Better use @tensorflow/tfjs-node or @tensorflow/tfjs-node-gpu but it has building isues
require("dotenv").config();

const { chat } = require("./repositories/chatRepository");

// Inisialisasi Express
const app = express();
const port = process.env.POST || 3000;

// Middleware
app.use(bodyParser.json());
app.use("/model", express.static("model"));

// Endpoint Retriever dengan gambar dalam format JSON (base64)
app.post("/chat", async (req, res) => {
  try {
    const { image, query } = req.body;
    const result = await chat(image, query);

    res.status(200).json({
      message: "Chat is delivered",
      data: result,
    });
  } catch (err) {
    const ERROR_MESSAGE = [
      "Confidence score really low, cannot classified image",
      "Cannot processed image",
      "RAG error",
    ];

    if (ERROR_MESSAGE.includes(err.message)) {
      res.status(500).json({ error: err.message });
    } else {
      res.status(500).json({error: "Internal server error"})
    }
  }
});

app.listen(port, () => {
  console.log(`Server is running`);
});
