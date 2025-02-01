import { CdpAgentkit } from "@coinbase/cdp-agentkit-core";
import { CdpToolkit } from "@coinbase/cdp-langchain";
import { HumanMessage } from "@langchain/core/messages";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { DallEAPIWrapper } from "@langchain/openai";
import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from "dotenv";
import * as fs from "fs";
import * as readline from "readline";
import { Wallet } from "@coinbase/coinbase-sdk";
import { CdpTool } from "@coinbase/cdp-langchain";
import { z } from "zod";

dotenv.config();

/**
 * Validates that required environment variables are set
 *
 * @throws {Error} - If required environment variables are missing
 * @returns {void}
 */
function validateEnvironment(): void {
  const missingVars: string[] = [];

  // Check required variables
  const requiredVars = [
    "OPENAI_API_KEY",
    "CDP_API_KEY_NAME",
    "CDP_API_KEY_PRIVATE_KEY",
  ];
  requiredVars.forEach((varName) => {
    if (!process.env[varName]) {
      missingVars.push(varName);
    }
  });

  // Exit if any required variables are missing
  if (missingVars.length > 0) {
    console.error("Error: Required environment variables are not set");
    missingVars.forEach((varName) => {
      console.error(`${varName}=your_${varName.toLowerCase()}_here`);
    });
    process.exit(1);
  }

  // Warn about optional NETWORK_ID
  if (!process.env.NETWORK_ID) {
    console.warn(
      "Warning: NETWORK_ID not set, defaulting to base-sepolia testnet"
    );
  }
}

// Add this right after imports and before any other code
validateEnvironment();

const INVOKE_CONTRACT_PROMPT = `
This tool allows publishing an ad to the decentralized advertising contract. It requires:
- The recipient address for the ad
- The text content of the ad

The tool will execute the contract call and return the transaction result.
You are expected to assist with coming up with the content of the ad.
Ask user questions about his had and the content of the ad.
And Use the final text as the content of the ad.
`;
const AD_CONTRACT_ADDRESS = "0x1f169173e8E54b65b4cd321217443E1919728e3c";
const AD_CONTRACT_ABI = [
  {
    inputs: [
      { internalType: "address", name: "to", type: "address" },
      { internalType: "string", name: "text", type: "string" },
    ],
    name: "publishAd",
    outputs: [{ internalType: "uint256", name: "", type: "uint256" }],
    stateMutability: "nonpayable",
    type: "function",
  },
];
const AbiInputOutput = z.object({
  name: z.string(),
  type: z.string(),
});

const AbiFunctionDefinition = z.object({
  name: z.string(),
  type: z.string(),
  inputs: z.array(AbiInputOutput),
  outputs: z.array(AbiInputOutput).optional(),
  stateMutability: z.string().optional(),
});

const InvokeContractInput = z.object({
  args: z.object({
    to: z
      .string()
      .describe("The recipient address for the ad. e.g. '0x1234...'"),
    text: z
      .string()
      .describe(
        "The text content of the ad. e.g. 'Check out our new product!'"
      ),
  }),
});

type AbiFunction = z.infer<typeof AbiFunctionDefinition>;

async function invokeContract(
  wallet: Wallet,
  contractAddress: string,
  method: string,
  abi: Array<AbiFunction>,
  args: Record<string, any>
): Promise<string> {
  try {
    const methodAbi = abi.find((func) => func.name === method);
    if (!methodAbi) {
      throw new Error(`Method ${method} not found in ABI`);
    }

    const contractInvocation = await wallet.invokeContract({
      contractAddress,
      method,
      args,
      abi,
    });

    const receipt = await contractInvocation.wait();
    const txHash = receipt.getTransactionHash();

    if (!txHash) {
      return "Contract invocation completed, but transaction hash is not available";
    }

    return `Ad published successfully. Transaction hash: https://sepolia.basescan.org/tx/${txHash}`;
  } catch (error) {
    if (error instanceof Error) {
      return `Failed to publish ad: ${error.message}`;
    }
    return "Failed to publish ad due to an unknown error";
  }
}

// Configure a file to persist the agent's CDP MPC Wallet Data
const WALLET_DATA_FILE = "wallet_data.txt";

/**
 * Initialize the agent with CDP Agentkit
 *
 * @returns Agent executor and config
 */
async function initializeAgent() {
  try {
    // Initialize LLM
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
    });

    let walletDataStr: string | null = null;

    // Read existing wallet data if available
    if (fs.existsSync(WALLET_DATA_FILE)) {
      try {
        walletDataStr = fs.readFileSync(WALLET_DATA_FILE, "utf8");
      } catch (error) {
        console.error("Error reading wallet data:", error);
        // Continue without wallet data
      }
    }

    // Configure CDP AgentKit
    const config = {
      cdpWalletData: walletDataStr || undefined,
      networkId: "base-sepolia",
    };

    // Initialize CDP AgentKit
    const agentkit = await CdpAgentkit.configureWithWallet(config);
    const dallETool = new DallEAPIWrapper({
      n: 1,
      model: "dall-e-3",
      apiKey: process.env.OPENAI_API_KEY,
    });
    // Initialize CDP AgentKit Toolkit and get tools
    const cdpToolkit = new CdpToolkit(agentkit);
    const tools = cdpToolkit.getTools();
    const allTools = [...cdpToolkit.getTools(), dallETool];
    // console.log("Tools:", allTools);

    const invokeContractTool = new CdpTool(
      {
        name: "publish_ad",
        description: INVOKE_CONTRACT_PROMPT,
        argsSchema: InvokeContractInput,
        func: async (
          wallet: Wallet,
          params: z.infer<typeof InvokeContractInput>
        ) => {
          return invokeContract(
            wallet,
            AD_CONTRACT_ADDRESS,
            "publishAd",
            AD_CONTRACT_ABI,
            params.args
          );
        },
      },
      agentkit
    );

    allTools.push(invokeContractTool);
    // Store buffered conversation history in memory
    const memory = new MemorySaver();
    const agentConfig = {
      configurable: { thread_id: "CDP AgentKit Chatbot Example!" },
    };

    // Create React Agent using the LLM and CDP AgentKit tools
    const agent = createReactAgent({
      llm,
      tools: allTools,
      checkpointSaver: memory,
      messageModifier: `
      You are specifically configured to publish ads to the decentralized advertising network named Placeholder ${AD_CONTRACT_ADDRESS}. And help out the Publishers of Placeholder.
      Placeholder is an advertising platform that allows users to publish ads on the blockchain.
      When asked to publish an ad or create the ad , you ask follow up question get to know about the user and his requirements.  
        You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit.
        You are empowered to interact onchain using your tools.
        Howwver you will not tell the user other details only when asked explcitly.
        When you start always request funds if balance is low, If you ever need funds, you can request them from the 
        faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet details and request 
        funds from the user. Before executing your first action, get the wallet details to see what network 
        you're on. If there is a 5XX (internal) HTTP error code, ask the user to try again later. 
        If someone  asks you to do something you can't do with your currently available tools, you must say so. Refrain from 
        restating your tools' descriptions unless it is explicitly requested.
        `,
    });

    // Save wallet data
    const exportedWallet = await agentkit.exportWallet();
    fs.writeFileSync(WALLET_DATA_FILE, exportedWallet);

    return { agent, config: agentConfig };
  } catch (error) {
    console.error("Failed to initialize agent:", error);
    throw error; // Re-throw to be handled by caller
  }
}

/**
 * Run the agent autonomously with specified intervals
 *
 * @param agent - The agent executor
 * @param config - Agent configuration
 * @param interval - Time interval between actions in seconds
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function runAutonomousMode(agent: any, config: any, interval = 10) {
  console.log("Starting autonomous mode...");

  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      const thought =
        "Be creative and do something interesting on the blockchain. " +
        "Choose an action or set of actions and execute it that highlights your abilities.";

      const stream = await agent.stream(
        { messages: [new HumanMessage(thought)] },
        config
      );

      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }

      await new Promise((resolve) => setTimeout(resolve, interval * 1000));
    } catch (error) {
      if (error instanceof Error) {
        console.error("Error:", error.message);
      }
      process.exit(1);
    }
  }
}

/**
 * Run the agent interactively based on user input
 *
 * @param agent - The agent executor
 * @param config - Agent configuration
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function runChatMode(agent: any, config: any) {
  console.log("Starting chat mode... Type 'exit' to end.");

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  try {
    // eslint-disable-next-line no-constant-condition
    while (true) {
      const userInput = await question("\nPrompt: ");

      if (userInput.toLowerCase() === "exit") {
        break;
      }

      const stream = await agent.stream(
        { messages: [new HumanMessage(userInput)] },
        config
      );

      for await (const chunk of stream) {
        if ("agent" in chunk) {
          console.log(chunk.agent.messages[0].content);
        } else if ("tools" in chunk) {
          console.log(chunk.tools.messages[0].content);
        }
        console.log("-------------------");
      }
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  } finally {
    rl.close();
  }
}

/**
 * Choose whether to run in autonomous or chat mode based on user input
 *
 * @returns Selected mode
 */
async function chooseMode(): Promise<"chat" | "auto"> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  const question = (prompt: string): Promise<string> =>
    new Promise((resolve) => rl.question(prompt, resolve));

  // eslint-disable-next-line no-constant-condition
  while (true) {
    console.log("\nAvailable modes:");
    console.log("1. chat    - Interactive chat mode");
    console.log("2. auto    - Autonomous action mode");

    const choice = (await question("\nChoose a mode (enter number or name): "))
      .toLowerCase()
      .trim();

    if (choice === "1" || choice === "chat") {
      rl.close();
      return "chat";
    } else if (choice === "2" || choice === "auto") {
      rl.close();
      return "auto";
    }
    console.log("Invalid choice. Please try again.");
  }
}

/**
 * Start the chatbot agent
 */
async function main() {
  try {
    const { agent, config } = await initializeAgent();
    const mode = await chooseMode();

    if (mode === "chat") {
      await runChatMode(agent, config);
    } else {
      await runAutonomousMode(agent, config);
    }
  } catch (error) {
    if (error instanceof Error) {
      console.error("Error:", error.message);
    }
    process.exit(1);
  }
}

if (require.main === module) {
  console.log("Starting Agent...");
  main().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
  });
}
