import React, { useState, useEffect } from 'react';
import { Ruler, ArrowRightLeft, AlertCircle, Info } from 'lucide-react';
import {
  cmToMetres,
  metresToCm,
  ftInToMetres,
  metresToFtIn,
  computeHeightEquivalents,
} from '../../utils/clinicalCalculators';

export type HeightUnit = 'm' | 'cm' | 'ft_in';

export interface HeightConverterProps {
  valueMetres: number | null | undefined;
  onChangeMetres: (val: number | null) => void;
  required?: boolean;
  disabled?: boolean;
  customError?: string | null;
}

export const HeightConverter: React.FC<HeightConverterProps> = ({
  valueMetres,
  onChangeMetres,
  required = true,
  disabled = false,
  customError,
}) => {
  const [activeUnit, setActiveUnit] = useState<HeightUnit>('m');
  const [showEquivalents, setShowEquivalents] = useState<boolean>(true);

  // Local text input states for each mode
  const [textMetres, setTextMetres] = useState<string>(() =>
    valueMetres ? String(valueMetres) : ''
  );
  const [textCm, setTextCm] = useState<string>(() =>
    valueMetres ? String(metresToCm(valueMetres)) : ''
  );
  const [textFeet, setTextFeet] = useState<string>(() => {
    if (!valueMetres) return '';
    const { feet } = metresToFtIn(valueMetres);
    return String(feet);
  });
  const [textInches, setTextInches] = useState<string>(() => {
    if (!valueMetres) return '';
    const { inches } = metresToFtIn(valueMetres);
    return String(inches);
  });

  const [inlineError, setInlineError] = useState<string | null>(null);

  // Sync with incoming canonical metres (e.g. preset change)
  useEffect(() => {
    if (valueMetres === null || valueMetres === undefined || isNaN(valueMetres)) {
      setTextMetres('');
      setTextCm('');
      setTextFeet('');
      setTextInches('');
    } else {
      const parsedM = parseFloat(textMetres);
      if (isNaN(parsedM) || Math.abs(parsedM - valueMetres) > 0.001) {
        setTextMetres(String(Number(valueMetres.toFixed(2))));
        setTextCm(String(Math.round(valueMetres * 100)));
        const { feet, inches } = metresToFtIn(valueMetres);
        setTextFeet(String(feet));
        setTextInches(String(inches));
      }
    }
  }, [valueMetres]);

  // Handle Metres Input Change
  const handleMetresChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setTextMetres(raw);
    setInlineError(null);

    if (raw.trim() === '') {
      onChangeMetres(null);
      setTextCm('');
      setTextFeet('');
      setTextInches('');
      return;
    }

    const val = parseFloat(raw);
    if (!isNaN(val)) {
      onChangeMetres(val);
      setTextCm(String(Number((val * 100).toFixed(1))));
      const { feet, inches } = metresToFtIn(val);
      setTextFeet(String(feet));
      setTextInches(String(inches));
    }
  };

  // Handle Centimetres Input Change
  const handleCmChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setTextCm(raw);
    setInlineError(null);

    if (raw.trim() === '') {
      onChangeMetres(null);
      setTextMetres('');
      setTextFeet('');
      setTextInches('');
      return;
    }

    const valCm = parseFloat(raw);
    if (!isNaN(valCm)) {
      const m = cmToMetres(valCm);
      onChangeMetres(m);
      setTextMetres(String(m));
      const { feet, inches } = metresToFtIn(m);
      setTextFeet(String(feet));
      setTextInches(String(inches));
    }
  };

  // Handle Feet & Inches Input Change
  const handleFtInChange = (newFeetStr: string, newInchesStr: string) => {
    setTextFeet(newFeetStr);
    setTextInches(newInchesStr);
    setInlineError(null);

    if (newFeetStr.trim() === '' && newInchesStr.trim() === '') {
      onChangeMetres(null);
      setTextMetres('');
      setTextCm('');
      return;
    }

    const f = parseFloat(newFeetStr) || 0;
    const inc = parseFloat(newInchesStr) || 0;
    if (f > 0 || inc > 0) {
      const m = ftInToMetres(f, inc);
      onChangeMetres(m);
      setTextMetres(String(Number(m.toFixed(2))));
      setTextCm(String(Math.round(m * 100)));
    }
  };

  // Blur validation on canonical height in metres
  const handleBlur = () => {
    if (valueMetres === null || valueMetres === undefined || isNaN(valueMetres)) {
      if (required) {
        setInlineError('Patient height is required.');
      } else {
        setInlineError(null);
      }
      return;
    }

    if (valueMetres < 0.50 || valueMetres > 2.50) {
      setInlineError('Height must be between 0.50 m (50 cm) and 2.50 m (250 cm).');
      return;
    }

    setInlineError(null);
  };

  const equivalents = valueMetres ? computeHeightEquivalents(valueMetres) : null;
  const activeError = customError || inlineError;

  return (
    <div className="space-y-2">
      {/* Label and Unit Selector */}
      <div className="flex flex-wrap items-center justify-between gap-1 text-xs font-medium text-slate-700 dark:text-slate-300">
        <label htmlFor="input-height" className="flex items-center gap-1.5 cursor-pointer">
          <span className="font-semibold text-slate-900 dark:text-slate-100">2. Height</span>
          <span className="text-teal-600 font-bold">
            ({activeUnit === 'm' ? 'metres' : activeUnit === 'cm' ? 'cm' : 'ft/in'})
          </span>
          {required && <span className="text-rose-500 font-bold">*</span>}
        </label>

        {/* Unit Selector Tabs */}
        <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-700 p-0.5 bg-slate-100 dark:bg-slate-800 text-[11px]">
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('m')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'm'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Metres (m)
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('cm')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'cm'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Centimetres (cm)
          </button>
          <button
            type="button"
            disabled={disabled}
            onClick={() => setActiveUnit('ft_in')}
            className={`px-2 py-0.5 rounded-md font-medium transition ${
              activeUnit === 'ft_in'
                ? 'bg-white dark:bg-slate-900 text-teal-700 dark:text-teal-300 shadow-xs'
                : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
            }`}
          >
            Feet &amp; In (ft/in)
          </button>
        </div>
      </div>

      {/* Input Field according to Active Unit */}
      {activeUnit === 'm' && (
        <div className="relative">
          <input
            id="input-height"
            type="text"
            inputMode="decimal"
            disabled={disabled}
            placeholder="e.g. 1.63"
            value={textMetres}
            onChange={handleMetresChange}
            onBlur={handleBlur}
            className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
              activeError
                ? 'border-rose-400 dark:border-rose-600 bg-rose-50/30 dark:bg-rose-950/20 text-rose-900 dark:text-rose-200 focus:ring-rose-500/30'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
            }`}
          />
          <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
            metres
          </span>
        </div>
      )}

      {activeUnit === 'cm' && (
        <div className="relative">
          <input
            id="input-height-cm"
            type="text"
            inputMode="decimal"
            disabled={disabled}
            placeholder="e.g. 163"
            value={textCm}
            onChange={handleCmChange}
            onBlur={handleBlur}
            className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
              activeError
                ? 'border-rose-400 dark:border-rose-600 bg-rose-50/30 dark:bg-rose-950/20 text-rose-900 dark:text-rose-200 focus:ring-rose-500/30'
                : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:ring-teal-500/30 focus:border-teal-500'
            }`}
          />
          <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
            cm
          </span>
        </div>
      )}

      {activeUnit === 'ft_in' && (
        <div className="grid grid-cols-2 gap-2">
          <div className="relative">
            <input
              id="input-height-feet"
              type="text"
              inputMode="numeric"
              disabled={disabled}
              placeholder="Feet (e.g. 5)"
              value={textFeet}
              onChange={(e) => handleFtInChange(e.target.value, textInches)}
              onBlur={handleBlur}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500"
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              ft
            </span>
          </div>
          <div className="relative">
            <input
              id="input-height-inches"
              type="text"
              inputMode="decimal"
              disabled={disabled}
              placeholder="Inches (e.g. 8)"
              value={textInches}
              onChange={(e) => handleFtInChange(textFeet, e.target.value)}
              onBlur={handleBlur}
              className="w-full px-3 py-2 rounded-lg border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500"
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              in
            </span>
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

      {/* Live Unit Converter Scale / Ruler */}
      <div className="p-2.5 rounded-lg bg-teal-50/60 dark:bg-teal-950/30 border border-teal-200/60 dark:border-teal-900/50 text-[11px]">
        <div className="flex items-center justify-between text-teal-900 dark:text-teal-200 font-medium mb-1">
          <span className="flex items-center gap-1">
            <Ruler className="w-3.5 h-3.5 text-teal-600" />
            <span>Unit Conversion Scale</span>
          </span>
          <span className="text-[10px] text-teal-700 dark:text-teal-300 font-mono">
            Canonical: {valueMetres ? `${valueMetres.toFixed(2)} m` : '--'}
          </span>
        </div>

        {equivalents ? (
          <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-[11px] text-teal-800 dark:text-teal-300">
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Metres</span>
              <span className="font-semibold">{equivalents.metres.toFixed(2)} m</span>
            </div>
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Centimetres</span>
              <span className="font-semibold">{Math.round(equivalents.centimetres)} cm</span>
            </div>
            <div className="p-1 rounded bg-white/80 dark:bg-slate-900/80 border border-teal-200/40 dark:border-teal-800/40">
              <span className="block text-[9px] text-slate-400 uppercase">Imperial</span>
              <span className="font-semibold">{equivalents.formattedFtIn}</span>
            </div>
          </div>
        ) : (
          <p className="text-[10px] text-slate-500 italic">
            Enter a height value above to preview live equivalents across metric and imperial units.
          </p>
        )}
      </div>
    </div>
  );
};

