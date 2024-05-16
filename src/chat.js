import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatOllama } from "@langchain/community/chat_models/ollama" 

const prompt = ChatPromptTemplate.fromMessages([
  ["human", `Tell me a joke about "{input}"`],
]);

const model = new ChatOllama({
    baseUrl: "http://localhost:11434",
    model: 'llama3'
});

const outputParser = new StringOutputParser();

const chain = prompt.pipe(model).pipe(outputParser);

const response = await chain.invoke({
  input: "Ice cream"
});

console.log(response);