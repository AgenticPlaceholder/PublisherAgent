// s3Client.ts
import * as dotenv from "dotenv";
dotenv.config();

import AWS from "aws-sdk";

export const s3 = new AWS.S3({
  apiVersion: "2006-03-01",
  region: "ap-south-1",
  credentials: new AWS.Credentials({
    accessKeyId: process.env.NEXT_PUBLIC_AWS_ACCESS_KEY_ID!,
    secretAccessKey: process.env.NEXT_PUBLIC_AWS_SECRET_ACCESS_KEY!,
  }),
  signatureVersion: "v4",
});
