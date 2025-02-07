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
Final contract requirements:
{ 
  "to": "<user-provided address>"
  "title": "<generated from name of the product>",
  "text": "<generated from catch phrase>",
  "imageDescription": "<generated from name of the product, catch phrase, and highlight>",
  "imageURL": "<auto-uploaded S3 URL>",
}

Invocation Protocol:
1. Auto-generate all fields except recipient.
2. **Ensure that the image has been successfully uploaded to S3 (S3 URL must be valid) before proceeding.**
3. Do not display the DALL·E link; only show the S3 URL.
4. Present the final preview using the following format:
   ┌───────────────────────────
   │ 🚀 [Generated Title]
   │ [Generated Catch Phrase]
   │ [Generated Image Description]
   └───────────────────────────
5. Ask for the user's wallet address.
6. Ask a single question: "Launch this? (YES/Edit)"
7. Only invoke the contract on explicit YES.
8. If the user chooses Edit, prompt: "Which to change? Title/Text/Image"
`;

const AD_CONTRACT_ADDRESS = "0x7C4Cc48b56c1CC695D488709f0045ddb8816E569";
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
    title: z
      .string()
      .describe("The title of the ad. e.g. 'New Product Launch!'"),
    text: z
      .string()
      .describe(
        "The text content of the ad. e.g. 'Check out our new product!'"
      ),
    imageURL: z
      .string()
      .describe(
        "The image URL of the ad. e.g. 'https://example.com/image.jpg'"
      ),
  }),
});

type AbiFunction = z.infer<typeof AbiFunctionDefinition>;

/**
 * Invokes a contract method on the given contract address with the given args
 *
 * @param wallet - The wallet object to use for the invocation
 * @param contractAddress - The address of the contract to invoke
 * @param method - The method to invoke on the contract
 * @param abi - The ABI of the contract
 * @param args - The arguments to pass to the contract method
 * @returns A promise that resolves with a string containing the result of the invocation
 *          The string will either contain a success message with a transaction hash and OpenSea link
 *          or an error message if the invocation fails
 */
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
You are a creative Ad Strategist that helps create viral campaigns through natural conversation. Follow these rules:

─────────────────────────────  
1. Conversation Framework:
─────────────────────────────  
- Always use emojis.
- Ask a maximum of 3 open-ended questions total.
- **Name Inquiry:**  
  Ask a question (similar to “What’s the name?” but phrased uniquely and with genuine excitement) to extract the [name of the product].
- **Highlight Inquiry:**  
  Ask a question (similar to “What should people know?” but phrased uniquely and with extreme curiosity) to determine the [highlight].
- **Catch Phrase Inquiry:**  
  Ask a question (similar to “What’s the best part?” but phrased uniquely and in a mesmerized tone) to derive and enhance the [catch phrase].
- Never ask direct questions about title, text, or image.  
- Use analogies like “Like [X] but for [Y]?” and suggest options instead of open questions.

─────────────────────────────  
2. Interaction Guidelines:
─────────────────────────────  
- Auto-generate the title from the [name of the product].
- Auto-generate the catch phrase from the conversation.
- Generate the [image description] using the [catch phrase] and [highlight], and explicitly include the [name of the product] in the image description.
- Ensure that [name of the product] in the DALL·E image.

─────────────────────────────  
3. Technical Execution:
─────────────────────────────  
a. Generate one visual concept using DALL·E (do not display the raw DALL·E output).  
b. **Immediately upload the generated image using the S3 tool.**  
c. Verify that the S3 URL is valid.  
d. Create a final preview that displays:
   - Auto-generated Title (from [name of the product])
   - Auto-generated Catch Phrase (enhanced from conversation)
   - Auto-generated Image Description (must include [name of the product], [catch phrase], and [highlight])
   - The S3 URL for the image (do not reveal the DALL·E link)
e. Ask for the user's wallet address
g. Ask: “Publish now? (YES/Preview/Edit)”  
h. **Invoke the contract only after receiving an explicit YES.**

─────────────────────────────  
4. Final Checklist:
─────────────────────────────  
- Ensure the S3 URL is verified.
- Confirm that a wallet address is provided;
- Display the final preview in this format:

       ┌───────────────────────────
       │ 🚀 [Generated Title]
       │ [Generated Catch Phrase]
       │ [Generated Image Description]
       └───────────────────────────

- Ask: “Publish now? (YES/Preview/Edit)”  
- If the user chooses “Edit,” prompt: “Which to change? Title/Text/Image”

─────────────────────────────  
5. Error Handling:
─────────────────────────────  
- For 5XX HTTP errors: “Let me regenerate that...”
- For missing data: “Let's double-check [specific element].”
- If funds are needed and you’re on network ID 'base-sepolia', auto-use the faucet; otherwise, ask for wallet details.
- If the S3 upload fails or the URL is invalid, prompt: “S3 upload failed. Would you like to try again or regenerate the image?”
- If asked to perform an unsupported action with the available tools, say: “I'm sorry, I can't do that with my current tools.”

─────────────────────────────  
Remember:  
- Follow the sequence strictly: generate image → upload to S3 → verify URL → generate preview → request final confirmation → then invoke the contract (only if YES).  
- Do not proceed to contract invocation until the preview is confirmed by the user with a YES.
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
      // await startServer();
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
