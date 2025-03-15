# EVM Wallet Analyst Agent

🎞️ [Presentation Link](https://www.figma.com/slides/PcWsiQpix0KChZHZNPAI6w/EVM-Wallet-Analyst-Agent?node-id=1-26&t=eDDNrXCedliQGqrm-1)

## Problem Statement
Crypto traders and investors face challenges managing their portfolios across the fragmented EVM ecosystem: With hundreds of EVM-compatible blockchains and thousands of tokens, traders struggle to maintain a comprehensive view of their holdings across chains. While portfolio trackers show basic balances, they fail to deliver actionable insights about the specific assets in a wallet or relevant market developments.

This project aims to solve these problems by creating an intelligent wallet analyst that provides a unified view of cross-chain holdings while delivering personalized news, insights, and intelligence specifically relevant to the assets in a user's wallet.

## Features
- Get real-time multi-chain portfolio of any EVM wallet
- Provide updates and insights about crypto assets held in the wallet
- Personalized news related to the wallet portfolio
- 20+ EVM blockchains are supported thanks to Zerion
- Accessible with any MCP client (Claude Desktop, Cursor, ...)

## Components
- [Heurist Mesh](https://github.com/heurist-network/heurist-agent-framework/tree/main/mesh) agents are used to fetch data from external data sources
  - [Zerion Wallet Analysis Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/zerion_wallet_analysis_agent.py) (implemented from scratch during the hackathon)
  - [Exa Search Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/exa_search_agent.py) (improved during the hackathon)
  - [Elfa Twitter Intelligence Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/elfa_twitter_intelligence_agent.py) (improved during the hackathon)
  - [Firecrawl Search Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/firecrawl_search_agent.py) (improved during the hackathon)
  - [CoinGecko Token Info Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/coingecko_token_info_agent.py)
- Utility agents are deployed on Near
  - https://app.near.ai/agents/swimsmoke.near/coingecko-token-info-agent/latest
  - https://app.near.ai/agents/swimsmoke.near/elfa-twitter-intelligence-agent/latest
  - https://app.near.ai/agents/swimsmoke.near/exa-search-agent/latest
  - https://app.near.ai/agents/swimsmoke.near/firecrawl-search-agent/latest
  - https://app.near.ai/agents/swimsmoke.near/zerion-wallet-analysis-agent/latest
- [Claude MCP server](https://github.com/heurist-network/heurist-mesh-mcp-server/blob/main/README.md)

## Setup Instructions
The above repositories have detailed setup instructions. For a limited time, you can use the following configuration to connect to the MCP server directly.

Claude Desktop MCP config file
```
{
    "mcpServers": {
      "mcp-proxy-agent-1": {
          "command": "mcp-proxy",
          "args": ["https://sequencer-v2.heurist.xyz/tool6271b/sse"]
      }
    }
  }
```

You need to install https://github.com/sparfenyuk/mcp-proxy to use MCP SSE endpoint URL on Claude Desktop. 

Alternatively, Cursor can use SSE directly. Next, you need to enter the following system prompt. You can do that by putting it inside a Claude project if you're using Claude Desktop, or putting it in the user input if you're using other MCP clients.

### Claude system prompt
```
Requirements:
- Use as many tools available to get a comprehensive result
- Be curious and explore different angles
- Discover relevant information even if user doesn't directly ask for it. For example, if a crypto token is mentioned in the result, do more research about it
- Today's date is March 15, 2025. Prioritize the sources that are more recent

Tool use rules:
- Prefer EXA search to Firecrawl for general-purpose web search
- Firecrawl is good at getting full web page contents but it's slow. Only use it if you want to deep dive into a specific web page if EXA search does not provide sufficient details
- Do not use web search for asset price because it's not accurate. Use the price from specialized market data tools
- Do not search for wallet address or token address on Twitter
- Use Twitter search for altcoin discussions to get latest news and sentiment around a project or token

Response rules:
- Pick the essential information that's worth paying attention to, and ignore minor uninteresting details
- Present news and updates related to the query
- Write like an email marketing newsletter targeting cryptocurrency users
- Don't write like a Goldman Sachs analyst
```

## Benchmark Method
While benchmarking the whole system end-to-end requires subjective evaluation of the final response quality of Claude and the data collection cannot be completed during the hackathon, we ran a benchmark of [Exa Search Agent](https://github.com/heurist-network/heurist-agent-framework/blob/main/mesh/exa_search_agent.py) which fetches and interprets data from Exa.ai and compared it with Perplexity results as the baseline.

### Benchmark command 
```
nearai benchmark run swimsmoke.near/perplexity_qa_dataset/1.0.0 MMLUSolverStrategy --agent ~/.nearai/registry/swimsmoke.near/exa-search-agent/0.0.1 --force
```

Local MMLUSolverStrategy class was override with the following code
```
from pydantic import BaseModel
from datasets import Dataset
from nearai.solvers import SolverStrategy
import openai
import json
import nearai
from typing import Dict, List

class PerplexityQADatum(BaseModel):
    input: str
    output: str

class MMLUSolverStrategy(SolverStrategy):
    """Solver for evaluating AI model responses against Perplexity QA dataset."""

    def __init__(self, dataset_ref: Dataset, model: str = "llama-v3p1-70b-instruct", agent: str = ""):
        super().__init__(model, agent)
        self.dataset_ref = dataset_ref
        
        # Setup OpenAI client with Near AI endpoint
        hub_url = "https://api.near.ai/v1"
        auth = nearai.config.load_config_file()["auth"]
        signature = json.dumps(auth)
        self.client = openai.OpenAI(base_url=hub_url, api_key=signature)
        self.evaluator_model = "llama-v3p1-70b-instruct"

    def evaluation_name(self) -> str:
        return "perplexity_qa_evaluation"

    def compatible_datasets(self) -> List[str]:
        return ["perplexity_qa_dataset"]

    def solve(self, datum: Dict[str, str]) -> bool:
        datum = PerplexityQADatum(**datum)
        question = datum.input
        reference_answer = datum.output
        
        # Get your model's answer using the inference session
        session = self.start_inference_session(question)
        model_answer = session.run_task(f"Answer this question: {question}").strip()
        
        # Evaluate the quality using the evaluator model
        evaluation_prompt = f"""You are an expert evaluator of AI responses. Compare the following two answers to the question and determine if the first answer is equal or better quality than the reference answer.

Question: {question}

Model Answer:
{model_answer}

Reference Answer:
{reference_answer}

Evaluate the model answer based on:
1. Factual accuracy
2. Completeness
3. Clarity and coherence
4. Relevance to the question

First, analyze both answers point by point. Then conclude with VERDICT: followed by exactly one word - BETTER, EQUAL, or WORSE.

IMPORTANT: Your final verdict MUST be in the format "VERDICT: X" where X is exactly one of these three words: BETTER, EQUAL, or WORSE.

Here are examples of how to evaluate different pairs of answers:

Example 1:
Question: What is blockchain technology?
Model Answer: Blockchain is a distributed, immutable ledger technology that records transactions across many computers. Each block contains a timestamp and link to the previous block, creating a chain. This design makes the blockchain resistant to modification of data. It's the foundation for cryptocurrencies like Bitcoin, but has many other applications including supply chain tracking, digital identity, and smart contracts.
Reference Answer: Blockchain is a type of database that stores data in blocks that are linked together. It's used for cryptocurrencies.

Analysis: The model answer provides a more comprehensive explanation of blockchain technology, covering its distributed nature, immutability, structure, security features, and various applications. The reference answer is factually correct but much more limited in scope and detail. The model answer is more educational and provides greater context.
VERDICT: BETTER

Example 2:
Question: How does photosynthesis work?
Model Answer: Photosynthesis is the process where plants convert sunlight into energy.
Reference Answer: Photosynthesis is the process by which green plants and certain other organisms transform light energy into chemical energy. During photosynthesis, light energy transfers electrons from water to carbon dioxide to produce carbohydrates. In this process, the carbon dioxide is reduced, or receives electrons, and the water is oxidized, or loses electrons. Ultimately, oxygen is produced along with carbohydrates like glucose, which stores energy.

Analysis: The reference answer provides a more detailed explanation of the chemical processes involved in photosynthesis, including the electron transfer mechanism. While the model answer covers the basic concept correctly, it lacks the depth and scientific precision of the reference answer.
VERDICT: WORSE

Example 3:
Question: What are smart contracts?
Model Answer: Smart contracts are self-executing contracts with the terms directly written into code. They run on blockchain networks and automatically execute when predetermined conditions are met. This eliminates the need for intermediaries and increases efficiency and transparency.
Reference Answer: Smart contracts are programs stored on a blockchain that run when predetermined conditions are met. They typically are used to automate the execution of an agreement so that all participants can be immediately certain of the outcome, without any intermediary's involvement or time loss.

Analysis: Both answers accurately explain what smart contracts are, their self-executing nature, blockchain foundation, and purpose of eliminating intermediaries. They cover similar points with slightly different wording but equivalent content and clarity.
VERDICT: EQUAL

Remember, I need your final line to be exactly "VERDICT: BETTER", "VERDICT: EQUAL", or "VERDICT: WORSE" - nothing else.
"""

        response = self.client.chat.completions.create(
            model=self.evaluator_model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        
        evaluation = response.choices[0].message.content
        
        # Extract the verdict
        if "VERDICT: BETTER" in evaluation or "VERDICT: EQUAL" in evaluation:
            return True
        else:
            return False
```

### Dataset Generation Script
We generated 20 Q&A and search results using Perplexity. The dataset should be ideally much larger, but we only included 20 samples due to the time limit. The dataset is uploaded to `namespace='swimsmoke.near' name='perplexity_qa_dataset' version='1.0.0'`
```
import requests
from datasets import Dataset
import json
import os

# --- Perplexity API configuration ---
API_KEY = ""
API_URL = "https://api.perplexity.ai/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_answer(question):
    """
    Call the Perplexity API with the given question to retrieve the answer.
    """
    payload = {
        "model": "sonar",  # Adjust the model name if needed.
        "messages": [{"role": "user", "content": question}],
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        # Assumes the first choice contains the generated message.
        answer = data['choices'][0]['message']['content']
        return answer.strip()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# --- Manually defined English questions ---
crypto_questions = [
    "What is Bitcoin and how does it work?",
    "How does blockchain technology ensure security?",
    "What are smart contracts and what are their uses?",
    "What is Ethereum and how is it different from Bitcoin?",
    "How do decentralized finance (DeFi) platforms operate?",
    "risks associated with investing in cryptocurrencies",
    "types of crypto wallets",
    "POW mining",
    "blockchain consensus mechanisms",
    "crypto regulations in the US"
]

general_questions = [
    "What causes the seasons to change?",
    "How does photosynthesis work in plants?",
    "What are the major causes of climate change?",
    "How do governments create economic policies?",
    "What is the significance of the Renaissance period?",
    "computer vision applications",
    "benefits of renewable energy",
    "AI in agriculture",
    "key factors in a successful startup",
    "human brain memory mechanisms"
]

# --- Build dataset entries ---
entries = {
    "id": [],
    "category": [],
    "input": [],   # The question
    "output": []   # The answer from Perplexity
}

# Process crypto-related questions
for i, question in enumerate(crypto_questions):
    answer = generate_answer(question)
    if answer is not None:
        entries["id"].append(f"crypto_{i+1}")
        entries["category"].append("crypto")
        entries["input"].append(question)
        entries["output"].append(answer)

# Process general domain questions
for i, question in enumerate(general_questions):
    answer = generate_answer(question)
    if answer is not None:
        entries["id"].append(f"general_{i+1}")
        entries["category"].append("general")
        entries["input"].append(question)
        entries["output"].append(answer)

# --- Create a Hugging Face dataset ---
dataset = Dataset.from_dict(entries)

# Optionally, register the dataset with Hugging Face Hub or save it locally.
# For local saving:
dataset.save_to_disk("perplexity_qa_dataset")

# Create metadata.json file for NearAI registry
metadata = {
    "name": "perplexity_qa_dataset",
    "description": "A dataset of questions and answers from Perplexity API covering crypto and general knowledge topics",
    "version": "1.0.0",
    "license": "MIT",
    "tags": ["perplexity", "crypto", "general_knowledge", "qa"],
    "author": "Heurist AI",
    "created_at": "2025-03-14",
    "category": "question-answering",
    "details": "This dataset contains Q&A pairs generated using the Perplexity API, covering cryptocurrency topics and general knowledge questions.",
    "show_entry": True
}

# Write the metadata file
with open(os.path.join("perplexity_qa_dataset", "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("Dataset created with", len(dataset), "entries.")
print("Metadata file created for NearAI registry.")
```

### Benchmark Result
```
Correct/Seen - 2/20 - 10.00%: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [02:29<00:00,  7.49s/it]
Final score: 2/20 - 10.00%
```
This indicates that 10% of the Exa Search Agent response is better than Perplexity. On average, it takes the agent 7.49s to generate a response.

## Future Plans
- Use MCP to send email updates (https://github.com/resend/mcp-send-email)
- Use an open source MCP client other than Claude Desktop
- Turn it into a paid agent-as-a-service
