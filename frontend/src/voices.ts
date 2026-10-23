export type VoiceGender = "original" | "female" | "male";

export type VoiceInfo = {
  id: string;
  name: string;
  gender: VoiceGender;
  blurb: string;
  featured: boolean;
  source: string;
  license: string;
};

export const DEFAULT_VOICE_ID = "original";
export const VOICE_STORAGE_KEY = "speechagent.voice_id";

export const FALLBACK_VOICES: VoiceInfo[] = [
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

export const GENDERS: { id: VoiceGender; label: string; hint: string }[] = [
  { id: "original", label: "Original", hint: "studio" },
  { id: "female", label: "Female", hint: "she / her" },
  { id: "male", label: "Male", hint: "he / him" },
];

export function loadStoredVoiceId(): string {
  try {
    return localStorage.getItem(VOICE_STORAGE_KEY) || DEFAULT_VOICE_ID;
  } catch {
    return DEFAULT_VOICE_ID;
  }
}

export function storeVoiceId(id: string): void {
  try {
    localStorage.setItem(VOICE_STORAGE_KEY, id);
  } catch {
    /* ignore quota / private mode */
  }
}

export function stickerFor(voice: VoiceInfo): string {
  if (voice.id === "original") return "★";
  if (voice.id === "heart") return "♥";
  if (voice.id === "bella") return "✦";
  if (voice.id === "nicole") return "☾";
  if (voice.id === "michael") return "◆";
  if (voice.id === "ryan") return "♪";
  if (voice.id === "joe") return "•";
  if (voice.id === "lessac") return "❧";
  if (voice.id === "amy") return "❀";
  return "○";
}
