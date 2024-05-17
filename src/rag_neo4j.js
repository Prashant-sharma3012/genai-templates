import { Neo4jVectorStore } from "@langchain/community/vectorstores/neo4j_vector";
import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const config = {
    url: "bolt://localhost:7687", // URL for the Neo4j instance
    username: "neo4j", // Username for Neo4j authentication
    password: "password", // Password for Neo4j authentication
    indexName: "movies-embedding", // Name of the vector index
    searchType: "vector", // Type of search (e.g., vector, hybrid)
    nodeLabel: "Movie", // Label for the nodes in the graph
    embeddingNodeProperty: "embedding", // Property of the node containing embedding
    textNodeProperties: ["released", "title", "tagline"]
};

const ollamaEmbeddings = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434",
    model: 'llama3'
});

const model = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: 'llama3'
});

const vectorStore = await Neo4jVectorStore.fromExistingGraph(
    ollamaEmbeddings,
    config
)

let vectorResult = await vectorStore.similaritySearch("what movies were released in 1992?")
let context = ``

vectorResult.map(d => {
    context = context + d.pageContent + Object.keys(d.metadata).reduce((res, k) =>  {
        res = res + ` ${k}:${d.metadata[k]} `
        return res
    }, '') + "\n"
})

console.log("############## Context provided to LLM ###################")
console.log(context)
console.log("############## Context provided to LLM ###################")

const prompt = ChatPromptTemplate.fromMessages([
    [
        "human",
        `Answer the question based on only the following context, do not use any other source of data. Only use this context, reply not found if data is not in context: 
         {context}`,
    ],
    ["human", `what movies were released in 1992?`],
]);

const outputParser = new StringOutputParser();

const chain2 = prompt.pipe(model).pipe(outputParser);

const response = await chain2.invoke({
    context
});

console.log(`LLM responded - ${response}`);