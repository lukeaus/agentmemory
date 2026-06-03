import type { EmbeddingProvider } from "../../types.js";
import { getEnvVar } from "../../config.js";
import { fetchWithTimeout } from "../_fetch.js";

const API_URL = "https://openrouter.ai/api/v1/embeddings";

export class OpenRouterEmbeddingProvider implements EmbeddingProvider {
  readonly name = "openrouter";
  readonly dimensions: number;
  private apiKey: string;
  private model: string;

  constructor(apiKey?: string) {
    this.apiKey = apiKey || getEnvVar("OPENROUTER_API_KEY") || "";
    if (!this.apiKey) throw new Error("OPENROUTER_API_KEY is required");
    this.dimensions = resolveDimensions();
    this.model =
      getEnvVar("OPENROUTER_EMBEDDING_MODEL") ||
      "openai/text-embedding-3-small";
  }

  async embed(text: string): Promise<Float32Array> {
    const [result] = await this.embedBatch([text]);
    return result;
  }

  async embedBatch(texts: string[]): Promise<Float32Array[]> {
    const response = await fetchWithTimeout(API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: this.model,
        input: texts,
        dimensions: this.dimensions,
      }),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(
        `OpenRouter embedding failed (${response.status}): ${err}`,
      );
    }

    const data = (await response.json()) as {
      data: Array<{ embedding: number[] }>;
    };

    return data.data.map((d) => new Float32Array(d.embedding));
  }
}

function resolveDimensions(): number {
  const raw = getEnvVar("OPENROUTER_EMBEDDING_DIMENSIONS");
  if (!raw) return 1536;
  const trimmed = raw.trim();
  if (!/^\d+$/.test(trimmed)) {
    throw new Error("OPENROUTER_EMBEDDING_DIMENSIONS must be a positive integer");
  }
  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) {
    throw new Error("OPENROUTER_EMBEDDING_DIMENSIONS must be a positive integer");
  }
  return parsed;
}
