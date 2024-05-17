import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { ChatOllama } from "@langchain/community/chat_models/ollama"
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama"
import { StringOutputParser } from "@langchain/core/output_parsers";
import { formatDocumentsAsString } from "langchain/util/document";

const ollamaEmbeddings = new OllamaEmbeddings({
    baseUrl: "http://localhost:11434",
    model: 'llama3'
});

const model = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: 'llama3'
});

const vectorStore = await HNSWLib.fromTexts(
    ["mitochondria is the powerhouse of the cell"],
    [{ id: 1 }],
    ollamaEmbeddings
);

const retriever = vectorStore.asRetriever();

const prompt = ChatPromptTemplate.fromMessages([
    [
        "ai",
        `Answer the question based on only the following context:
  
{context}`,
    ],
    ["human", "{question}"],
]);

const chain = RunnableSequence.from([
    {
        context: retriever.pipe(formatDocumentsAsString),
        question: new RunnablePassthrough(),
    },
    prompt,
    model,
    new StringOutputParser(),
]);

const result = await chain.invoke("What is power house of cell?");

console.log(result);
/**
Harrison worked at Kensho.
 */