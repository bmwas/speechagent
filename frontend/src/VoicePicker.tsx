import { GENDERS, stickerFor, type VoiceGender, type VoiceInfo } from "./voices";

type Props = {
  voices: VoiceInfo[];
  selectedId: string;
  disabled: boolean;
  onSelect: (id: string) => void;
};

function sortVoices(list: VoiceInfo[]): VoiceInfo[] {
  return [...list].sort((a, b) => Number(b.featured) - Number(a.featured) || a.name.localeCompare(b.name));
}

export default function VoicePicker({ voices, selectedId, disabled, onSelect }: Props) {
  const selected = voices.find((v) => v.id === selectedId) || voices[0];
  const gender: VoiceGender = selected?.gender || "original";
  const visible = sortVoices(voices.filter((v) => v.gender === gender));

  return (
    <section className="voice-booth" aria-labelledby="voice-booth-title">
      <div className="voice-booth-head">
        <div>
          <p className="voice-kicker">pick a speaking buddy</p>
          <h2 id="voice-booth-title">Voice closet</h2>
        </div>
        <p className="voice-selected" aria-live="polite">
          <span className="voice-sticker" aria-hidden="true">
            {selected ? stickerFor(selected) : "★"}
          </span>
          {selected ? `${selected.name} · ${selected.blurb}` : "Original"}
        </p>
      </div>
      <div className="voice-tabs" role="tablist" aria-label="Voice gender">
        {GENDERS.map((tab) => {
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
