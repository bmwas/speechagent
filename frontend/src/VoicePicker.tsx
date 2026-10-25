import { GENDERS_MINICPM, GENDERS_QWEN, stickerFor, type VoiceFamily, type VoiceGender, type VoiceInfo } from "./voices";

type Props = {
  voices: VoiceInfo[];
  selectedId: string;
  disabled: boolean;
  family: VoiceFamily;
  onSelect: (id: string) => void;
};

function sortVoices(list: VoiceInfo[]): VoiceInfo[] {
  return [...list].sort((a, b) => Number(b.featured) - Number(a.featured) || a.name.localeCompare(b.name));
}

export default function VoicePicker({ voices, selectedId, disabled, family, onSelect }: Props) {
  const selected = voices.find((v) => v.id === selectedId) || voices[0];
  // For Qwen, default to "female" if no selection; for MiniCPM default to "original"
  const defaultGender: VoiceGender = family === "qwen" ? "female" : "original";
  const gender: VoiceGender = selected?.gender || defaultGender;
  const visible = sortVoices(voices.filter((v) => v.gender === gender));
  const genders = family === "qwen" ? GENDERS_QWEN : GENDERS_MINICPM;
  const kicker = family === "qwen" ? "pick a built-in speaker" : "pick a speaking buddy";

  const boothTitle = family === "qwen" ? "Speaker" : "Voice closet";
  const defaultLabel = family === "qwen" ? "Chelsie" : "Original";

  return (
    <section className="voice-booth" aria-labelledby="voice-booth-title">
      <div className="voice-booth-head">
        <div>
          <p className="voice-kicker">{kicker}</p>
          <h2 id="voice-booth-title">{boothTitle}</h2>
        </div>
        <p className="voice-selected" aria-live="polite">
          <span className="voice-sticker" aria-hidden="true">
            {selected ? stickerFor(selected) : "★"}
          </span>
          {selected ? `${selected.name} · ${selected.blurb}` : defaultLabel}
        </p>
      </div>
      <div className="voice-tabs" role="tablist" aria-label="Voice gender">
        {genders.map((tab) => {
          const active = tab.id === gender;
          const first = sortVoices(voices.filter((v) => v.gender === tab.id))[0];
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={active}
              className={active ? "voice-tab on" : "voice-tab"}
              disabled={disabled || !first}
              onClick={() => first && onSelect(first.id)}
            >
              <span>{tab.label}</span>
              <small>{tab.hint}</small>
            </button>
          );
        })}
      </div>
      <div className="voice-grid" role="listbox" aria-label={`${gender} voices`} aria-disabled={disabled}>
        {visible.map((voice) => {
          const on = voice.id === selectedId;
          return (
            <button
              key={voice.id}
              type="button"
              role="option"
              aria-selected={on}
              disabled={disabled}
              className={on ? "voice-card on" : "voice-card"}
              onClick={() => onSelect(voice.id)}
            >
              <span className="voice-sticker" aria-hidden="true">
                {stickerFor(voice)}
              </span>
              <span className="voice-name">
                {voice.name}
                {voice.featured ? <em>favorite</em> : null}
              </span>
              <span className="voice-blurb">{voice.blurb}</span>
            </button>
          );
        })}
      </div>
      {disabled ? (
        <p className="voice-lock">Voice is locked while you are connected. Disconnect to swap.</p>
      ) : null}
    </section>
  );
}
