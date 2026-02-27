import axios from "axios";

const GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";

export const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.0-flash";
export const GEMINI_MODEL_MINI = process.env.GEMINI_MODEL_MINI || "gemini-1.5-flash";

// Gemini has a 1M token context window, so we can afford much larger scans
// than Groq's practical ~6K limit. These limits scale every fetch accordingly.
export const PROVIDER_LIMITS = {
  gemini: {
    maxInputTokens: 100000,
    maxOutputTokens: 8192,
    maxFilesFullScan: 50,
    maxLinesPerFile: 500,
    maxChangedFiles: 20,
    maxChangedFileLines: 300,
    maxPatchFiles: 15,
    maxPatchFileLines: 200,
    contextOptimizeAt: 80000,
    maxPatchSections: 10,
  },
  groq: {
    maxInputTokens: 6000,
    maxOutputTokens: 6000,
    maxFilesFullScan: 25,
    maxLinesPerFile: 200,
    maxChangedFiles: 10,
    maxChangedFileLines: 150,
    maxPatchFiles: 10,
    maxPatchFileLines: 100,
    contextOptimizeAt: 8000,
    maxPatchSections: 5,
  },
};

// Used by git.worker.js at job start to decide how much data to fetch
// before we even know which provider key will succeed.
export function getActiveLimits() {
  const hasGemini = !!(
    process.env.GEMINI_API_KEY1 ||
    process.env.GEMINI_API_KEY2 ||
    process.env.GEMINI_API_KEY3
  );
  return hasGemini ? PROVIDER_LIMITS.gemini : PROVIDER_LIMITS.groq;
}

// Gemini's API uses a different message format from OpenAI:
// system prompt goes into `systemInstruction`, conversation turns into `contents`,
// and assistant role is called "model" instead of "assistant".
function toGeminiPayload(messages) {
  const systemMsg = messages.find((m) => m.role === "system");
  const nonSystem = messages.filter((m) => m.role !== "system");

  const contents = nonSystem.map((m) => ({
    role: m.role === "assistant" ? "model" : "user",
    parts: [{ text: m.content }],
  }));

  const payload = { contents };
  if (systemMsg) {
    payload.systemInstruction = { parts: [{ text: systemMsg.content }] };
  }
  return payload;
}

export async function callGeminiAPI(
  messages,
  { model, maxTokens, temperature },
  apiKey,
  timeout = 60000,
) {
  const { contents, systemInstruction } = toGeminiPayload(messages);

  const body = {
    contents,
    generationConfig: { temperature, maxOutputTokens: maxTokens },
  };
  if (systemInstruction) body.systemInstruction = systemInstruction;

  const response = await axios.post(
    `${GEMINI_API_BASE}/${model}:generateContent?key=${apiKey}`,
    body,
    { headers: { "Content-Type": "application/json" }, timeout },
  );

  const candidate = response.data?.candidates?.[0];
  if (!candidate) throw new Error("No candidates in Gemini response");

  // Gemini blocks certain content instead of erroring — treat it like an error
  if (candidate.finishReason === "SAFETY") {
    throw new Error("Gemini content filtered for safety reasons");
  }

  const text = candidate?.content?.parts?.[0]?.text;
  if (!text) throw new Error("Invalid response from Gemini API");
  return text;
}
