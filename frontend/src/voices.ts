export type VoiceGender = "original" | "female" | "male";
export type VoiceFamily = "minicpm" | "qwen";

export type VoiceInfo = {
  id: string;
  name: string;
  gender: VoiceGender;
  blurb: string;
  featured: boolean;
  source?: string;
  license?: string;
};

export const DEFAULT_VOICE_ID = "original";
export const DEFAULT_QWEN_SPEAKER = "chelsie";
export const VOICE_STORAGE_KEY_MINICPM = "speechagent.voice_id.minicpm";
export const VOICE_STORAGE_KEY_QWEN = "speechagent.voice_id.qwen";

// Legacy key for migration
const LEGACY_VOICE_KEY = "speechagent.voice_id";

export const FALLBACK_VOICES_MINICPM: VoiceInfo[] = [
  {
    id: "original",
    name: "Original",
    gender: "original",
    blurb: "the MiniCPM studio clone — our first voice",
    featured: true,
    source: "MiniCPM HT_ref_audio",
    license: "Apache-2.0",
  },
  {
    id: "heart",
    name: "Heart",
    gender: "female",
    blurb: "warm sparkle, like a pocket sun",
    featured: true,
    source: "Kokoro Heart",
    license: "Apache-2.0",
  },
  {
    id: "bella",
    name: "Bella",
    gender: "female",
    blurb: "bright and bouncy",
    featured: true,
    source: "Kokoro Bella",
    license: "unverified",
  },
  {
    id: "nicole",
    name: "Nicole",
    gender: "female",
    blurb: "soft evening tea",
    featured: false,
    source: "Kokoro Nicole",
    license: "unverified",
  },
  {
    id: "michael",
    name: "Michael",
    gender: "male",
    blurb: "calm and clear",
    featured: true,
    source: "Kokoro Michael",
    license: "unverified",
  },
  {
    id: "ryan",
    name: "Ryan",
    gender: "male",
    blurb: "friendly radio host",
    featured: true,
    source: "Piper Ryan",
    license: "CC BY-NC-SA 4.0",
  },
  {
    id: "joe",
    name: "Joe",
    gender: "male",
    blurb: "easygoing neighbor",
    featured: false,
    source: "Piper Joe",
    license: "CC0 / Lessac-derived",
  },
  {
    id: "lessac",
    name: "Lessac",
    gender: "female",
    blurb: "storybook narrator",
    featured: false,
    source: "Piper Lessac",
    license: "research",
  },
  {
    id: "amy",
    name: "Amy",
    gender: "female",
    blurb: "gentle and even",
    featured: false,
    source: "Piper Amy",
    license: "see upstream",
  },
];

export const FALLBACK_VOICES_QWEN: VoiceInfo[] = [
  {
    id: "chelsie",
    name: "Chelsie",
    gender: "female",
    blurb: "honeyed warmth with luminous clarity",
    featured: true,
  },
  {
    id: "ethan",
    name: "Ethan",
    gender: "male",
    blurb: "bright energy with an approachable vibe",
    featured: true,
  },
  {
    id: "aiden",
    name: "Aiden",
    gender: "male",
    blurb: "warm laid-back American charm",
    featured: true,
  },
];

// Legacy alias
export const FALLBACK_VOICES = FALLBACK_VOICES_MINICPM;

export const GENDERS_MINICPM: { id: VoiceGender; label: string; hint: string }[] = [
  { id: "original", label: "Original", hint: "studio" },
  { id: "female", label: "Female", hint: "she / her" },
  { id: "male", label: "Male", hint: "he / him" },
];

export const GENDERS_QWEN: { id: VoiceGender; label: string; hint: string }[] = [
  { id: "female", label: "Female", hint: "she / her" },
  { id: "male", label: "Male", hint: "he / him" },
];

// Legacy alias
export const GENDERS = GENDERS_MINICPM;

export function loadStoredVoiceId(family: VoiceFamily = "minicpm"): string {
  try {
    const key = family === "qwen" ? VOICE_STORAGE_KEY_QWEN : VOICE_STORAGE_KEY_MINICPM;
    const stored = localStorage.getItem(key);
    if (stored) return stored;
    // Try legacy key for migration (MiniCPM only)
    if (family === "minicpm") {
      const legacy = localStorage.getItem(LEGACY_VOICE_KEY);
      if (legacy) return legacy;
    }
    return family === "qwen" ? DEFAULT_QWEN_SPEAKER : DEFAULT_VOICE_ID;
  } catch {
    return family === "qwen" ? DEFAULT_QWEN_SPEAKER : DEFAULT_VOICE_ID;
  }
}

export function storeVoiceId(id: string, family: VoiceFamily = "minicpm"): void {
  try {
    const key = family === "qwen" ? VOICE_STORAGE_KEY_QWEN : VOICE_STORAGE_KEY_MINICPM;
    localStorage.setItem(key, id);
  } catch {
    /* ignore quota / private mode */
  }
}

export function stickerFor(voice: VoiceInfo): string {
  // MiniCPM voices
  if (voice.id === "original") return "★";
  if (voice.id === "heart") return "♥";
  if (voice.id === "bella") return "✦";
  if (voice.id === "nicole") return "☾";
  if (voice.id === "michael") return "◆";
  if (voice.id === "ryan") return "♪";
  if (voice.id === "joe") return "•";
  if (voice.id === "lessac") return "❧";
  if (voice.id === "amy") return "❀";
  // Qwen speakers
  if (voice.id === "chelsie") return "✿";
  if (voice.id === "ethan") return "◈";
  if (voice.id === "aiden") return "◉";
  return "○";
}
