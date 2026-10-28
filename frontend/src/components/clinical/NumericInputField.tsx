import React, { useState, useEffect } from 'react';
import { AlertCircle } from 'lucide-react';

export interface NumericInputFieldProps {
  id: string;
  label: string;
  unit?: string;
  value: number | null | undefined;
  onChange: (val: number | null) => void;
  min?: number;
  max?: number;
  step?: number | string;
  placeholder?: string;
  helperText?: string;
  required?: boolean;
  disabled?: boolean;
  className?: string;
  customError?: string | null;
}

export const NumericInputField: React.FC<NumericInputFieldProps> = ({
  id,
  label,
  unit,
  value,
  onChange,
  min,
  max,
  step = 'any',
  placeholder,
  helperText,
  required = false,
  disabled = false,
  className = '',
  customError,
}) => {
  // Separate editing state from numeric value
  const [text, setText] = useState<string>(() =>
    value !== null && value !== undefined && !isNaN(value) ? String(value) : ''
  );
  const [blurError, setBlurError] = useState<string | null>(null);

  // Sync internal text if external value changed significantly (e.g., presets applied)
  useEffect(() => {
    if (value === null || value === undefined) {
      if (text !== '') setText('');
    } else {
      const parsedCurrent = parseFloat(text);
      if (isNaN(parsedCurrent) || parsedCurrent !== value) {
        setText(String(value));
      }
    }
  }, [value]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setText(raw);
    setBlurError(null);

    if (raw.trim() === '') {
      onChange(null);
      return;
    }

    // Allow typing valid numeric patterns (including single dot like "1.")
    const parsed = parseFloat(raw);
    if (!isNaN(parsed)) {
      onChange(parsed);
    }
  };

  const handleBlur = () => {
    const trimmed = text.trim();
    if (trimmed === '') {
      if (required) {
        setBlurError('This field is required.');
      } else {
        setBlurError(null);
      }
      onChange(null);
      return;
    }

    const parsed = parseFloat(trimmed);
    if (isNaN(parsed)) {
      setBlurError('Please enter a valid numeric value.');
      onChange(null);
      return;
    }

    if (min !== undefined && parsed < min) {
      setBlurError(`Value cannot be less than ${min}${unit ? ` ${unit}` : ''}.`);
      return;
    }
    if (max !== undefined && parsed > max) {
      setBlurError(`Value cannot exceed ${max}${unit ? ` ${unit}` : ''}.`);
      return;
    }

    setBlurError(null);
    onChange(parsed);
  };

  const activeError = customError || blurError;

  return (
    <div className={`space-y-1 ${className}`}>
      <div className="flex justify-between items-center text-xs font-medium text-slate-700 dark:text-slate-300">
        <label htmlFor={id} className="flex items-center gap-1 cursor-pointer">
          <span>{label}</span>
          {unit && <span className="text-teal-600 font-semibold">({unit})</span>}
          {required && <span className="text-rose-500 font-bold">*</span>}
        </label>
        {value !== null && value !== undefined && (
          <span className="font-mono text-xs text-slate-500 dark:text-slate-400">
            {value} {unit}
          </span>
        )}
      </div>

      <div className="relative">
        <input
          id={id}
          type="text"
          inputMode="decimal"
          disabled={disabled}
          placeholder={placeholder}
          value={text}
          onChange={handleChange}
          onBlur={handleBlur}
          className={`w-full px-3 py-2 rounded-lg border text-sm font-mono transition-colors focus:outline-none focus:ring-2 ${
            disabled
              ? 'bg-slate-100 dark:bg-slate-800 text-slate-400 border-slate-200 dark:border-slate-800 cursor-not-allowed'
              : activeError
              ? 'border-rose-400 dark:border-rose-600 bg-rose-50/30 dark:bg-rose-950/20 text-rose-900 dark:text-rose-200 focus:ring-rose-500/30 focus:border-rose-500'
              : 'border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-teal-500/30 focus:border-teal-500'
          }`}
        />
      </div>

      {activeError ? (
        <div className="flex items-center gap-1 text-[11px] text-rose-600 dark:text-rose-400 font-medium">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>{activeError}</span>
        </div>
      ) : helperText ? (
        <p className="text-[10px] text-slate-400 leading-tight">{helperText}</p>
      ) : null}
    </div>
  );
};

