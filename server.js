const express = require("express");
const bodyParser = require("body-parser");
require("dotenv").config();
const cors = require("cors);


// Inisialisasi Express
const app = express();
const port = process.env.PORT || 3000;
app.use(cors());

// Middleware
app.use(bodyParser.json());

app.use("/model", express.static("model"));
const { chat } = require("./repositories/chatRepository");

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
      console.log({error: err})
      res.status(500).json({error: "Internal server error"})
    }
  }
});

app.listen(port, () => {
  console.log(`Server is running`);
});
