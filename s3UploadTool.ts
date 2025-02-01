import { z } from "zod";
import axios from "axios";
import { s3 } from "./s3Client";
import { Wallet } from "@coinbase/coinbase-sdk";
import { CdpTool } from "@coinbase/cdp-langchain";

export const S3UploadInput = z.object({
  dallEImageUrl: z
    .string()
    .url()
    .describe(
      "The URL returned by DALL·E that should be fetched and uploaded to S3."
    ),
});

// A helper function to upload a PNG from a URL to S3
export async function uploadImageUrlToS3(imageUrl: string): Promise<string> {
  try {
    // 1. Fetch the image from the DALL·E-provided URL
    const response = await axios.get(imageUrl, { responseType: "arraybuffer" });

    // 2. The buffer of the PNG file
    const fileBuffer = Buffer.from(response.data);

    // 3. Generate a unique S3 object key (e.g., date/time + random)
    const objectKey = `ad-images/${Date.now()}-${Math.floor(
      Math.random() * 10000
    )}.png`;

    // 4. Upload to S3
    const uploadResult = await s3
      .upload({
        Bucket: "placeholderads",
        Key: objectKey,
        Body: fileBuffer,
        ContentType: "image/png",
        //   ACL: "public-read", // if you want a public URL
      })
      .promise();
    const s3Url = `https://placeholderads.s3.ap-south-1.amazonaws.com/${objectKey}`;

    // 5. Return the final S3 URL
    return s3Url;
    // e.g. https://my-awesome-bucket.s3.amazonaws.com/dalle-images/...
  } catch (error) {
    console.error("Error uploading to S3:", error);
    throw new Error(
      `S3 upload failed: ${
        error instanceof Error ? error.message : "Unknown error"
      }`
    );
  }
}
export function createS3UploadTool(agentkit: any) {
  return new CdpTool(
    {
      name: "upload_to_s3",
      description:
        "Uploads a PNG image from a DALL·E URL to S3 and returns the S3 URL.",
      argsSchema: S3UploadInput,
      func: async (wallet: Wallet, params) => {
        // 'wallet' is passed automatically by AgentKit, but we don't need it here
        const { dallEImageUrl } = params;
        const s3Url = await uploadImageUrlToS3(dallEImageUrl);
        return s3Url;
      },
    },
    agentkit
  );
}
