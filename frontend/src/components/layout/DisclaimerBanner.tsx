import React from 'react';
import { AlertTriangle, ShieldCheck } from 'lucide-react';

export const DisclaimerBanner: React.FC = () => {
  return (
    <div className="bg-amber-50 dark:bg-amber-950/40 border-b border-amber-200 dark:border-amber-800/60 px-4 py-2 text-xs text-amber-900 dark:text-amber-200 flex items-center justify-between">
      <div className="flex items-center gap-2 max-w-5xl mx-auto">
        <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0" />
        <p>
          <span className="font-semibold uppercase tracking-wider text-[11px] bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-100 px-1.5 py-0.5 rounded mr-1.5">
            Research Use Only
          </span>
          Model predictions are experimental research outputs and are not an authorized substitute for clinical DXA bone densitometry or medical consultation.
        </p>
      </div>
      <div className="hidden md:flex items-center gap-1.5 text-amber-700 dark:text-amber-300 shrink-0 text-[11px]">
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
        <span>IRB & Ethics Compliant • Zero PHI Retained</span>
      </div>
    </div>
  );
};
