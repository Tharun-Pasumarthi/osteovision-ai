import React, { useState, useEffect } from 'react';
import { Scale, Info, Check, AlertCircle, Edit3 } from 'lucide-react';
import { calculateBmi, BmiResult } from '../../utils/clinicalCalculators';
import { Badge } from '../ui/Badge';

export interface BmiCalculatorProps {
  heightM: number | null | undefined;
  weightKg: number | null | undefined;
  onBmiChange: (bmi: number | null, isOverridden: boolean) => void;
  disabled?: boolean;
}

export const BmiCalculator: React.FC<BmiCalculatorProps> = ({
  heightM,
  weightKg,
  onBmiChange,
  disabled = false,
}) => {
  const [isManualOverride, setIsManualOverride] = useState<boolean>(false);
  const [manualBmiText, setManualBmiText] = useState<string>('');
  const [manualError, setManualError] = useState<string | null>(null);

  // Auto-calculated BMI from canonical height (m) and weight (kg)
  const autoResult: BmiResult = calculateBmi(heightM, weightKg);

  // Propagate BMI update whenever inputs change or override toggles
  useEffect(() => {
    if (!isManualOverride) {
      if (autoResult.isValid && autoResult.bmi !== null) {
        onBmiChange(autoResult.bmi, false);
      } else {
        onBmiChange(null, false);
      }
    } else {
      const parsed = parseFloat(manualBmiText);
      if (!isNaN(parsed) && parsed >= 10.0 && parsed <= 70.0) {
        onBmiChange(parsed, true);
      } else {
        onBmiChange(null, true);
      }
    }
  }, [heightM, weightKg, isManualOverride, manualBmiText]);

  const handleManualChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    setManualBmiText(raw);
    setManualError(null);

    if (raw.trim() === '') {
      onBmiChange(null, true);
      return;
    }

    const val = parseFloat(raw);
    if (!isNaN(val)) {
      onBmiChange(val, true);
    }
  };

  const handleManualBlur = () => {
    const parsed = parseFloat(manualBmiText);
    if (isNaN(parsed) || parsed < 10.0 || parsed > 70.0) {
      setManualError('Manual BMI must be between 10.0 and 70.0 kg/m².');
    } else {
      setManualError(null);
    }
  };

  const toggleManualOverride = () => {
    if (disabled) return;
    if (!isManualOverride) {
      // Transitioning to manual: preload auto calculated value if valid
      if (autoResult.isValid && autoResult.bmi !== null) {
        setManualBmiText(String(autoResult.bmi));
      } else {
        setManualBmiText('');
      }
      setIsManualOverride(true);
    } else {
      // Revert to auto calculation
      setIsManualOverride(false);
      setManualError(null);
      setManualBmiText('');
    }
  };

  return (
    <div className="p-3.5 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-2.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Scale className="w-4 h-4 text-teal-600" />
          <span className="font-semibold text-slate-800 dark:text-slate-200 text-xs">
            4. Body Mass Index (num_bmi)
          </span>
          <span className="text-[10px] text-teal-600 font-mono font-bold">(kg/m²)</span>
        </div>

        {/* Source of Truth / Manual Override Toggle */}
        <button
          type="button"
          onClick={toggleManualOverride}
          disabled={disabled}
          className="text-[11px] text-slate-500 hover:text-teal-600 dark:text-slate-400 dark:hover:text-teal-400 flex items-center gap-1 transition"
          title="Toggle clinical manual BMI entry"
        >
          <Edit3 className="w-3 h-3" />
          <span>{isManualOverride ? 'Use Auto Calculation' : 'Manual Override'}</span>
        </button>
      </div>

      {!isManualOverride ? (
        /* Automatic Calculation Display */
        <div className="flex items-center justify-between pt-0.5">
          <div>
            <div className="text-[11px] text-slate-500 dark:text-slate-400 flex items-center gap-1.5">
              <span>Formula: weight (kg) / height² (m²)</span>
              <span className="px-1.5 py-0.2 rounded bg-teal-100 dark:bg-teal-950 text-teal-800 dark:text-teal-300 text-[9px] font-semibold">
                Auto-calculated
              </span>
            </div>
            {autoResult.isValid ? (
              <div className="flex items-center gap-2 mt-1">
                <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${autoResult.categoryColor}`}>
                  {autoResult.category}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">
                  ({weightKg} kg ÷ {heightM ? `${heightM}² m²` : '--'})
                </span>
              </div>
            ) : (
              <p className="text-[11px] text-amber-600 dark:text-amber-400 italic mt-0.5">
                {autoResult.message}
              </p>
            )}
          </div>

          <div className="text-right">
            <span className="text-lg font-bold font-mono text-teal-600 dark:text-teal-400">
              {autoResult.isValid && autoResult.bmi !== null ? `${autoResult.bmi.toFixed(2)} kg/m²` : '--'}
            </span>
          </div>
        </div>
      ) : (
        /* Manual Override Mode */
        <div className="space-y-1.5 pt-0.5">
          <div className="flex items-center justify-between text-[11px]">
            <span className="text-amber-700 dark:text-amber-400 font-medium flex items-center gap-1">
              <Info className="w-3 h-3" />
              <span>Manual Override Active: Value will be transmitted directly to API model.</span>
            </span>
          </div>
          <div className="relative">
            <input
              id="input-manual-bmi"
              type="text"
              inputMode="decimal"
              disabled={disabled}
              placeholder="e.g. 24.50"
              value={manualBmiText}
              onChange={handleManualChange}
              onBlur={handleManualBlur}
              className={`w-full px-3 py-1.5 rounded-lg border text-sm font-mono transition ${
                manualError
                  ? 'border-rose-400 bg-rose-50/30 text-rose-900 focus:ring-rose-500/30'
                  : 'border-amber-300 dark:border-amber-700 bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 focus:ring-amber-500/30 focus:border-amber-500'
              }`}
            />
            <span className="absolute right-3 top-2 text-xs text-slate-400 pointer-events-none">
              kg/m²
            </span>
          </div>
          {manualError && (
            <div className="flex items-center gap-1 text-[11px] text-rose-600 font-medium">
              <AlertCircle className="w-3.5 h-3.5 shrink-0" />
              <span>{manualError}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

