import type { MemoryProvider } from "../types.js";
import { getEnvVar } from "../config.js";
import { fetchWithTimeout } from "./_fetch.js";

type OpenRouterRequestBody = {
  model: string;
  max_tokens: number;
  messages: Array<{ role: "system" | "user"; content: string }>;
  reasoning?: { effort: string };
  include_reasoning?: true;
};

export class OpenRouterProvider implements MemoryProvider {
  name: string;
  private apiKey: string;
  private model: string;
  private maxTokens: number;
  private baseUrl: string;
  private reasoningEffort?: string;
  private includeReasoning: boolean;

  constructor(
    apiKey: string,
    model: string,
    maxTokens: number,
    baseUrl: string,
  ) {
    this.apiKey = apiKey;
    this.model = model;
    this.maxTokens = maxTokens;
    this.baseUrl = baseUrl;
    this.name = baseUrl.includes("openrouter") ? "openrouter" : "gemini";
    this.reasoningEffort =
      this.name === "openrouter"
        ? getEnvVar("OPENROUTER_REASONING_EFFORT")?.trim() || undefined
        : undefined;
    this.includeReasoning =
      this.name === "openrouter" &&
      getEnvVar("OPENROUTER_INCLUDE_REASONING") === "true";
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt);
  }

  private async call(
    systemPrompt: string,
    userPrompt: string,
  ): Promise<string> {
    const body: OpenRouterRequestBody = {
      model: this.model,
      max_tokens: this.maxTokens,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    };
    if (this.reasoningEffort) {
      body.reasoning = { effort: this.reasoningEffort };
    }
    if (this.includeReasoning) {
      body.include_reasoning = true;
    }

    const response = await fetchWithTimeout(this.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
        ...(this.baseUrl.includes("openrouter")
          ? { "HTTP-Referer": "https://github.com/rohitg00/agentmemory" }
          : {}),
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`${this.name} API error (${response.status}): ${text}`);
    }

    const data = (await response.json()) as {
      choices?: Array<{
        message?: { content?: string; reasoning?: string; reasoning_content?: string };
      }>;
    };
    const message = data.choices?.[0]?.message;
    const content = message?.content;
    if (content) {
      return content;
    }
    const reasoning = message?.reasoning ?? message?.reasoning_content;
    if (reasoning) {
      return reasoning;
    }
    throw new Error(
      `${this.name} returned unexpected response: ${JSON.stringify(data).slice(0, 200)}`,
    );
  }
}
