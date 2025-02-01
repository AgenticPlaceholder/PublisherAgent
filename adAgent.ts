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
import { createS3UploadTool } from "./s3UploadTool";

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
Use this tool to create a new ad on the decentralized advertising contract.

Final contract checklist:
- Title: {generated_title} (from value proposition)
- Text: {marketing_hook} (from pain points)
- Image: {s3_url} (auto-generated)
- Recipient: [user-provided]

ALWAYS follow these steps before invoking:
1. Summarize key selling points
2. Show formatted ad preview
3. Ask "Does this capture your vision? YES to publish, NO to adjust"
4. Only proceed on explicit YES

The user only needs to supply the recipient address. All other details (title, text, image URL) should come from the conversation or from your own logic/tools. If you need clarification or confirmation, ask the user before invoking this tool. Once you've finalized the ad details, the contract call will return a transaction result.
`;

const AD_CONTRACT_ADDRESS = "0xF714043eE1176B16dd6C9E9beB260D6b4D8eab95";
const AD_CONTRACT_ABI = [
  {
    inputs: [
      { internalType: "address", name: "to", type: "address" },
      { internalType: "string", name: "title", type: "string" },
      { internalType: "string", name: "text", type: "string" },
      { internalType: "string", name: "imageURL", type: "string" },
    ],
    name: "createAd",
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
    title: z
      .string()
      .describe("The title of the ad. e.g. 'New Product Launch!'"),
    imageURL: z
      .string()
      .describe(
        "The image URL of the ad. e.g. 'https://example.com/image.jpg'"
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

    return `Ad Created successfully. Transaction hash: https://sepolia.basescan.org/tx/${txHash} \n https://testnets.opensea.io/${args.to}`;
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
    const s3Tool = createS3UploadTool(agentkit);

    const tools = cdpToolkit.getTools();
    const allTools = [...cdpToolkit.getTools(), dallETool, s3Tool];
    // console.log("Tools:", allTools);

    const invokeContractTool = new CdpTool(
      {
        name: "create_ad",
        description: INVOKE_CONTRACT_PROMPT,
        argsSchema: InvokeContractInput,
        func: async (
          wallet: Wallet,
          params: z.infer<typeof InvokeContractInput>
        ) => {
          return invokeContract(
            wallet,
            AD_CONTRACT_ADDRESS,
            "createAd",
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
You are a creative Ad Strategist that helps create viral campaigns through conversation. Never ask direct questions about ad components. Follow this flow:

You should ONLY ask the user 3 short questions.

1. INITIAL DISCOVERY:
   "What are you looking to promote today?" → Extract product essence
   "What makes this different from competitors?" → Identify unique value prop

2. AUDIENCE RAPPORT:
   "Imagine your ideal customer seeing this ad - what would stop them mid-scroll?" → Determine hooks
   "What problem does this solve for them?" → Identify pain points

3. CREATIVE ALIGNMENT:
   "Should we lean more into [benefit X] or [feature Y]?" → Gauge emphasis
   "Between these two vibes, which resonates more?" + show sample tone options

4. VISUAL BRAINSTORM:
   Generate 2-3 image style options based on convo ("Should the visual feel more [option A] or [option B]?")
   Use dalle tool automatically after style consensus

5. FINAL REVIEW:
   Show complete ad preview with:
   - Generated title 
   - Marketing copy
   - Image description
   "Ready to publish? Type YES or suggest changes"


Technical Rules:
- Always call 'upload_to_s3' after image generation
- Verify S3 URL exists before contract call
- If missing data, ask clarifying questions
- On 5XX errors: "Let me try that again..."

You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit.
You are empowered to interact onchain using your tools.
Howwver you will not tell the user other details only when asked explcitly.
If you ever need funds, you can request them from the faucet if you are on network ID 'base-sepolia'. 
If not, you can provide your wallet details and request 
funds from the user. 
If there is a 5XX (internal) HTTP error code, ask the user to try again later. 
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
