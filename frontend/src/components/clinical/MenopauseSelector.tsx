import React, { useState, useEffect } from 'react';
import { Calendar, AlertCircle, Info, ShieldAlert } from 'lucide-react';

export type MenopausalStatus = 'post' | 'pre_peri' | 'unknown';

export interface MenopauseSelectorProps {
  isFemale: boolean;
  status: MenopausalStatus;
  onsetAge: number | null | undefined;
  onStatusChange: (status: MenopausalStatus) => void;
  onOnsetAgeChange: (age: number | null) => void;
  disabled?: boolean;
}

export const MenopauseSelector: React.FC<MenopauseSelectorProps> = ({
  isFemale,
  status,
  onsetAge,
  onStatusChange,
  onOnsetAgeChange,
  disabled = false,
}) => {
  // Local text input state for onset age to decouple typing from numeric state
  const [ageText, setAgeText] = useState<string>(() =>
    onsetAge !== null && onsetAge !== undefined && !isNaN(onsetAge) ? String(onsetAge) : ''
  );
  const [ageError, setAgeError] = useState<string | null>(null);

  // Sync internal age text when external onsetAge changes (e.g. Presets)
  useEffect(() => {
    if (onsetAge === null || onsetAge === undefined) {
      if (ageText !== '') setAgeText('');
    } else {
      const parsed = parseFloat(ageText);
      if (isNaN(parsed) || parsed !== onsetAge) {
        setAgeText(String(onsetAge));
      }
    }
  }, [onsetAge]);

  const handleAgeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setAgeText(raw);
    setAgeError(null);

    if (raw.trim() === '') {
      // Empty input is valid: indicates missing/unknown (will be imputed with training median)
      // CRITICAL: NEVER convert empty string to 0!
      onOnsetAgeChange(null);
      return;
    }

    const parsed = parseFloat(raw);
    if (!isNaN(parsed)) {
      onOnsetAgeChange(parsed);
    }
  };

  const handleAgeBlur = () => {
    const trimmed = ageText.trim();
    if (trimmed === '') {
      onOnsetAgeChange(null);
      setAgeError(null);
      return;
    }

    const parsed = parseFloat(trimmed);
    if (isNaN(parsed)) {
      setAgeError('Please enter a valid age in years.');
      onOnsetAgeChange(null);
      return;
    }

    if (parsed < 20.0 || parsed > 80.0) {
      setAgeError('Menopause onset age must be between 20 and 80 years.');
      return;
    }

    setAgeError(null);
    onOnsetAgeChange(parsed);
  };

  // If patient is Male: menopause age is strictly not applicable per contract
  if (!isFemale) {
    return (
      <div className="p-3 rounded-xl bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px] text-slate-500 dark:text-slate-400 space-y-1">
        <div className="font-semibold text-slate-700 dark:text-slate-300 flex items-center gap-1.5">
          <Info className="w-3.5 h-3.5 text-slate-400" />
          <span>5. Menopause Age (num_menopause_age)</span>
          <span className="px-1.5 py-0.2 rounded bg-slate-200 dark:bg-slate-800 text-slate-600 dark:text-slate-400 text-[10px]">
            Not Applicable
          </span>
        </div>
        <p className="leading-relaxed">
          Not applicable for male subjects per original research dataset semantics. Transmitted as{' '}
          <code className="text-teal-600 font-mono">null</code> (safely imputed with cohort median 47.0y by backend preprocessor).
        </p>
      </div>
    );
  }

  // Patient is Female
  return (
    <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-3">
      {/* Status Radio / Selector */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs font-medium text-slate-700 dark:text-slate-300">
          <span className="flex items-center gap-1.5">
            <Calendar className="w-3.5 h-3.5 text-teal-600" />
            <span>5. Menopausal Status &amp; Onset Age</span>
          </span>
          {status === 'post' && onsetAge !== null && onsetAge !== undefined && !isNaN(onsetAge) && (
            <span className="font-mono text-teal-600 font-bold text-xs">{onsetAge}y</span>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <button
            type="button"
            disabled={disabled}
            onClick={() => {
              onStatusChange('post');
            }}
            className={`py-2 px-3 rounded-lg border font-medium text-center transition ${
              status === 'post'
                ? 'bg-teal-600 text-white border-teal-600 shadow-xs'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-750'
            }`}
          >
            Post-Menopausal
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => {
              onStatusChange('pre_peri');
              onOnsetAgeChange(null);
            }}
            className={`py-2 px-3 rounded-lg border font-medium text-center transition ${
              status === 'pre_peri'
                ? 'bg-teal-600 text-white border-teal-600 shadow-xs'
                : 'bg-white dark:bg-slate-800 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-750'
            }`}
          >
            Pre- / Peri-Menopausal
          </button>
        </div>
      </div>

      {/* Conditional Onset Age Input when Post-Menopausal */}
      {status === 'post' ? (
        <div className="space-y-1 pt-1 border-t border-slate-200 dark:border-slate-800">
          <div className="flex justify-between items-center text-[11px] text-slate-600 dark:text-slate-400">
            <label htmlFor="input-menopause-age" className="cursor-pointer font-medium">
              Menopause Onset Age (years):
            </label>
            <span className="text-[10px] text-slate-400 italic">
              Optional (leave blank for median 47.0y)
            </span>
          </div>

          <div className="relative">
            <input
              id="input-menopause-age"
              type="text"
              inputMode="decimal"
              disabled={disabled}
              placeholder="e.g. 47 (typical 40-55y)"
              value={ageText}
              onChange={handleAgeChange}
              onBlur={handleAgeBlur}
              className={`w-full px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors focus:outline-none focus:ring-2 ${
                ageError
                  ? 'border-rose-400 bg-rose-50/30 text-rose-900 focus:ring-rose-500/30'
                  : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
              }`}
            />
            <span className="absolute right-3 top-1.5 text-xs text-slate-400 pointer-events-none">
              years
            </span>
          </div>

          {ageError ? (
            <div className="flex items-center gap-1 text-[11px] text-rose-600 font-medium">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{ageError}</span>
            </div>
          ) : (
            <p className="text-[10px] text-slate-400 leading-tight">
              Values must be between 20 and 80 years. If unknown or left blank, the research cohort median (47.0y) is safely applied.
            </p>
          )}
        </div>
      ) : (
        <div className="p-2 rounded-lg bg-teal-50/50 dark:bg-teal-950/20 text-[10px] text-slate-500 dark:text-slate-400 border border-teal-200/40 dark:border-teal-900/40">
          <span>Menopause onset age is not applicable for pre-/peri-menopausal patients. Feature will be passed as null (safely median-imputed).</span>
        </div>
      )}
    </div>
  );
};
