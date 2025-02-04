// const { ChromaClient } = require("chromadb");
import { ChromaClient } from "chromadb";

// Connect to ChromaDB
const chroma = new ChromaClient({ path: "http://localhost:8000" }); // Update with your path

// Default number of clusters
const k_clusters = 4;

async function fetchData() {
    const collection = await chroma.getCollection({name: "app_reviews"});
    const results = await collection.get({ include: ["embeddings", "documents"] });

    return {
        embeddings: results.embeddings,  // Array of vectors
        reviews: results.documents       // Corresponding review texts
    };
}

// const kmeans = require("ml-kmeans");
import { kmeans } from "ml-kmeans";

async function clusterData(embeddings, k = k_clusters) {
    const { clusters, centroids } = kmeans(embeddings, k, { initialization: "kmeans++" });
    return clusters;  // Array of cluster indices
}

// const natural = require("natural");
// const stopword = require("stopword");
import natural from "natural";
import stopword from "stopword";

function extractKeywords(reviews, clusterLabels) {
    let clusterKeywords = {};

    // Group reviews by cluster
    reviews.forEach((review, i) => {
        const cluster = clusterLabels[i];
        if (!clusterKeywords[cluster]) clusterKeywords[cluster] = [];
        clusterKeywords[cluster].push(review);
    });

    // Extract keywords
    for (let cluster in clusterKeywords) {
        let text = clusterKeywords[cluster].join(" ");
        let tokenizer = new natural.WordTokenizer();
        let words = tokenizer.tokenize(text.toLowerCase());

        // Remove stopwords & count word frequency
        let filteredWords = stopword.removeStopwords(words);
        let wordFreq = filteredWords.reduce((acc, word) => {
            acc[word] = (acc[word] || 0) + 1;
            return acc;
        }, {});

        // Sort & select top keywords
        let sortedWords = Object.entries(wordFreq).sort((a, b) => b[1] - a[1]);
        clusterKeywords[cluster] = sortedWords.slice(0, 10).map(([word]) => word);
    }

    return clusterKeywords;
}

async function main() {
    const { embeddings, reviews } = await fetchData();
    const clusterLabels = await clusterData(embeddings, k_clusters);
    const keywords = extractKeywords(reviews, clusterLabels);

    console.log("\n📊 Cluster Keywords:");
    for (let cluster in keywords) {
        console.log(`Cluster ${cluster}: ${keywords[cluster].join(", ")}`);
    }
}

main();
