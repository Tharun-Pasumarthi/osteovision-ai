import React, { useState, useEffect } from 'react';
import { Footprints, AlertCircle, ArrowRightLeft, Info, HelpCircle } from 'lucide-react';
import {
  stepsToKm,
  metresToKm,
  kmToSteps,
  computeDistanceEquivalents,
  DEFAULT_STEP_LENGTH_M,
} from '../../utils/clinicalCalculators';

export type DistanceUnit = 'km' | 'metres' | 'steps';

export interface WalkingDistanceConverterProps {
  valueKm: number | null | undefined;
  onChangeKm: (val: number | null) => void;
  required?: boolean;
  disabled?: boolean;
  customError?: string | null;
}

export const WalkingDistanceConverter: React.FC<WalkingDistanceConverterProps> = ({
  valueKm,
  onChangeKm,
  required = true,
  disabled = false,
  customError,
}) => {
  const [activeUnit, setActiveUnit] = useState<DistanceUnit>('km');
  const [stepLengthM, setStepLengthM] = useState<number>(DEFAULT_STEP_LENGTH_M);
  const [showConfig, setShowConfig] = useState<boolean>(false);

  // Local text input states
  const [textKm, setTextKm] = useState<string>(() =>
    valueKm !== null && valueKm !== undefined && !isNaN(valueKm) ? String(valueKm) : ''
  );
  const [textMetres, setTextMetres] = useState<string>(() =>
    valueKm !== null && valueKm !== undefined && !isNaN(valueKm) ? String(Math.round(valueKm * 1000)) : ''
  );
  const [textSteps, setTextSteps] = useState<string>(() =>
    valueKm !== null && valueKm !== undefined && !isNaN(valueKm)
      ? String(kmToSteps(valueKm, DEFAULT_STEP_LENGTH_M))
      : ''
  );

  const [inlineError, setInlineError] = useState<string | null>(null);

  // Synchronize when valueKm prop changes (e.g., Presets)
  useEffect(() => {
    if (valueKm === null || valueKm === undefined || isNaN(valueKm)) {
      setTextKm('');
      setTextMetres('');
      setTextSteps('');
    } else {
      const parsedKm = parseFloat(textKm);
      if (isNaN(parsedKm) || Math.abs(parsedKm - valueKm) > 0.01) {
        setTextKm(String(Number(valueKm.toFixed(2))));
        setTextMetres(String(Math.round(valueKm * 1000)));
        setTextSteps(String(kmToSteps(valueKm, stepLengthM)));
      }
    }
  }, [valueKm, stepLengthM]);

  // Handle direct km input
  const handleKmChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setTextKm(raw);
    setInlineError(null);

    if (raw.trim() === '') {
      onChangeKm(null);
      setTextMetres('');
      setTextSteps('');
      return;
    }

    const val = parseFloat(raw);
    if (!isNaN(val)) {
      onChangeKm(val);
      setTextMetres(String(Math.round(val * 1000)));
      setTextSteps(String(kmToSteps(val, stepLengthM)));
    }
  };

  // Handle metres input
  const handleMetresChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setTextMetres(raw);
    setInlineError(null);

    if (raw.trim() === '') {
      onChangeKm(null);
      setTextKm('');
      setTextSteps('');
      return;
    }

    const valM = parseFloat(raw);
    if (!isNaN(valM)) {
      const km = metresToKm(valM);
      onChangeKm(km);
      setTextKm(String(km));
      setTextSteps(String(kmToSteps(km, stepLengthM)));
    }
  };

  // Handle steps input
  const handleStepsChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setTextSteps(raw);
    setInlineError(null);

    if (raw.trim() === '') {
      onChangeKm(null);
      setTextKm('');
      setTextMetres('');
      return;
    }

    const valSteps = parseFloat(raw);
    if (!isNaN(valSteps)) {
      const km = stepsToKm(valSteps, stepLengthM);
      onChangeKm(km);
      setTextKm(String(km));
      setTextMetres(String(Math.round(km * 1000)));
    }
  };

  const handleBlur = () => {
    if (valueKm === null || valueKm === undefined || isNaN(valueKm)) {
      if (required) {
        setInlineError('Max walking distance is required.');
      } else {
        setInlineError(null);
      }
      return;
    }

    if (valueKm < 0.0 || valueKm > 50.0) {
      setInlineError('Walking distance must be between 0.0 and 50.0 km.');
      return;
    }

    setInlineError(null);
  };

  // Quick preset steps
  const applyPresetSteps = (steps: number) => {
    const km = stepsToKm(steps, stepLengthM);
    onChangeKm(km);
    setTextKm(String(km));
    setTextMetres(String(Math.round(km * 1000)));
    setTextSteps(String(steps));
    setInlineError(null);
  };

  const equivalents = valueKm !== null && valueKm !== undefined ? computeDistanceEquivalents(valueKm, stepLengthM) : null;
  const activeError = customError || inlineError;

  return (
    <div className="space-y-2">
      {/* Label and Unit Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-1 text-xs font-medium text-slate-700 dark:text-slate-300">
        <label htmlFor="input-walking" className="flex items-center gap-1.5 cursor-pointer">
          <span className="font-semibold text-slate-900 dark:text-slate-100">6. Max Walking Distance</span>
          <span className="text-teal-600 font-bold">
            ({activeUnit === 'km' ? 'kilometres' : activeUnit === 'metres' ? 'metres' : 'estimated steps'})
          </span>
          {required && <span className="text-rose-500 font-bold">*</span>}
        </label>

        {/* Unit Selector Tabs */}
        <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-700 p-0.5 bg-slate-100 dark:bg-slate-800 text-[11px]">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('km')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'km'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Kilometres (km)
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('metres')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'metres'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Metres (m)
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('steps')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'steps'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Steps (est.)
          </button>
        </div>
      </div>

      {/* Input Field according to Active Unit */}
      {activeUnit === 'km' && (
        <div className="relative">
          <input
            id="input-walking"
            type="text"
            inputMode="decimal"
            disabled={disabled}
            placeholder="e.g. 2.5"
            value={textKm}
            onChange={handleKmChange}
            onBlur={handleBlur}
            className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
              activeError
                ? 'border-rose-400 bg-rose-50/30 text-rose-900 focus:ring-rose-500/30'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
            }`}
          />
          <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
            km
          </span>
        </div>
      )}

      {activeUnit === 'metres' && (
        <div className="relative">
          <input
            id="input-walking-metres"
            type="text"
            inputMode="decimal"
            disabled={disabled}
            placeholder="e.g. 2500"
            value={textMetres}
            onChange={handleMetresChange}
            onBlur={handleBlur}
            className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
              activeError
                ? 'border-rose-400 bg-rose-50/30 text-rose-900 focus:ring-rose-500/30'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
            }`}
          />
          <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
            metres
          </span>
        </div>
      )}

      {activeUnit === 'steps' && (
        <div className="space-y-1.5">
          <div className="relative">
            <input
              id="input-walking-steps"
              type="text"
              inputMode="numeric"
              disabled={disabled}
              placeholder="e.g. 3333"
              value={textSteps}
              onChange={handleStepsChange}
              onBlur={handleBlur}
              className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
                activeError
                  ? 'border-rose-400 bg-rose-50/30 text-rose-900 focus:ring-rose-500/30'
                  : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
              }`}
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              steps
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500">
            <span>Quick Step Presets:</span>
            <div className="flex items-center gap-1">
              {[1000, 3000, 5000, 10000].map((s) => (
                <button
                  key={s}
                  type="button"
                  onClick={() => applyPresetSteps(s)}
                  className="px-1.5 py-0.5 rounded bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-[10px] font-mono transition"
                >
                  {s.toLocaleString()}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Inline Validation Error */}
      {activeError && (
        <div className="flex items-center gap-1 text-[11px] text-rose-600 dark:text-rose-400 font-medium">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>{activeError}</span>
        </div>
      )}

      {/* Walking Distance Converter Scale & Step Assumption Disclosure */}
      <div className="p-2.5 rounded-lg bg-teal-50/60 dark:bg-teal-950/30 border border-teal-200/60 dark:border-teal-900/50 text-[11px] space-y-1.5">
        <div className="flex items-center justify-between text-teal-900 dark:text-teal-200 font-medium">
          <span className="flex items-center gap-1">
            <Footprints className="w-3.5 h-3.5 text-teal-600" />
            <span>Walking Distance Converter Scale</span>
          </span>
          <span className="text-[10px] text-teal-700 dark:text-teal-300 font-mono">
            Canonical: {valueKm !== null && valueKm !== undefined ? `${valueKm.toFixed(2)} km` : '--'}
          </span>
        </div>

        {equivalents ? (
          <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-[11px] text-teal-800 dark:text-teal-300">
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Kilometres</span>
              <span className="font-semibold">{equivalents.km.toFixed(2)} km</span>
            </div>
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Metres</span>
              <span className="font-semibold">{equivalents.metres.toLocaleString()} m</span>
            </div>
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Steps (est.)</span>
              <span className="font-semibold">~{equivalents.steps.toLocaleString()}</span>
            </div>
          </div>
        ) : (
          <p className="text-[10px] text-slate-500 italic">
            Enter a walking distance above to preview metric and step conversions.
          </p>
        )}

        <div className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight pt-1 border-t border-teal-200/40 dark:border-teal-800/40 flex items-start gap-1">
          <Info className="w-3 h-3 shrink-0 mt-0.5 text-teal-600" />
          <span>
            <strong>Clinical Note:</strong> Assumes ~{stepLengthM} m/step. Step count is an estimate; maximum continuous walking distance measures musculoskeletal mobility and may differ from all-day cumulative step tracking.
          </span>
        </div>
      </div>
    </div>
  );
};

