require('dotenv').config();
const express = require('express');
const path = require('path');
const cors = require('cors');
const KJUR = require('jsrsasign');
const fetch = require('node-fetch');
const app = express();

app.use(cors());
app.use(express.json());
app.use(express.static('public')); // Serve static files

// Environment variables
const {
    KID,
    ISS,
    SUB,
    PROJECT_ID,
    ENDPOINT_ID,
    PRIVATE_KEY
} = process.env;

const endpoint = `https://northamerica-northeast2-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/northamerica-northeast2/endpoints/${ENDPOINT_ID}:predict`;
const aud = "https://aiplatform.googleapis.com/";

console.log("Environment Variables:", {
    KID: KID?.substring(0, 4) + '...',
    ISS: ISS?.substring(0, 4) + '...',
    SUB: SUB?.substring(0, 4) + '...',
    PROJECT_ID: PROJECT_ID,
    ENDPOINT_ID: ENDPOINT_ID,
    PRIVATE_KEY: PRIVATE_KEY ? "Exists" : "Missing"
  });

// Serve index.html for root route
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/predict', async (req, res) => {
    try {
        const { ba, obp, slg } = req.body;
        
        // Log incoming request data
        console.log("Incoming Request Data:", { ba, obp, slg });

        // Log environment variables
        console.log("Environment Variables:", {
            KID: KID,
            ISS: ISS,
            SUB: SUB,
            PROJECT_ID: PROJECT_ID,
            ENDPOINT_ID: ENDPOINT_ID,
            PRIVATE_KEY: PRIVATE_KEY ? "Exists" : "Missing"
        });

        // Replace escaped newlines with actual newlines
        const formattedPrivateKey = PRIVATE_KEY.replace(/\\n/g, '\n');

        // Log part of the PRIVATE_KEY to ensure it's read correctly
        console.log("PRIVATE_KEY (partial):", formattedPrivateKey.substring(0, 30) + '...');

        // Generate JWT
        const header = { alg: "RS256", typ: "JWT", kid: KID };
        const now = Math.floor(Date.now() / 1000);
        const payload = {
            iss: ISS,
            sub: SUB,
            aud: aud,
            iat: now,
            exp: now + 3600
        };

        console.log("JWT Header:", header);
        console.log("JWT Payload:", payload);
        
        let token;
        try {
            token = KJUR.jws.JWS.sign(
                "RS256",
                JSON.stringify(header),
                JSON.stringify(payload),
                formattedPrivateKey
            );
            console.log("Generated Token:", token);
        } catch (tokenError) {
            console.error("Error Generating Token:", tokenError);
            throw new Error("Token generation failed");
        }

        console.log("Request Details:", {
            method: "POST",
            url: endpoint,
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ instances: [[ba, obp, slg]] })
        });

        // Make request to Vertex AI
        const vertexResponse = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            body: JSON.stringify({ instances: [[ba, obp, slg]] })
        });

        console.log("Vertex AI Response Status:", vertexResponse.status);
        
        if (!vertexResponse.ok) {
            const errorBody = await vertexResponse.text();
            console.error("Vertex AI Error Response:", errorBody);
            throw new Error(`Vertex AI Error: ${vertexResponse.status} - ${errorBody}`);
        }

        const data = await vertexResponse.json();
        console.log("Vertex AI Response Data:", data);
        
        res.json(data);
        
    } catch (error) {
        // Preserve full error details
        console.error("Complete Error Object:", {
            message: error.message,
            stack: error.stack,
            code: error.code,
            type: typeof error
        });
        
        res.status(500).json({ 
            error: error.message,
            stack: process.env.NODE_ENV === 'production' ? undefined : error.stack
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));