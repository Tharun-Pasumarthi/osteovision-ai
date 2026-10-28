import React from 'react';

interface BadgeProps {
  children: React.ReactNode;
  variant?: 'normal' | 'osteopenia' | 'osteoporosis' | 'teal' | 'blue' | 'slate' | 'outline';
  size?: 'sm' | 'md';
  className?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  children,
  variant = 'slate',
  size = 'md',
  className = '',
}) => {
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-2.5 py-1 text-xs';

  const variantClasses = {
    normal: 'bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-800',
    osteopenia: 'bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border border-amber-200 dark:border-amber-800',
    osteoporosis: 'bg-rose-50 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 border border-rose-200 dark:border-rose-800',
    teal: 'bg-teal-50 dark:bg-teal-950/50 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800',
    blue: 'bg-sky-50 dark:bg-sky-950/50 text-sky-700 dark:text-sky-300 border border-sky-200 dark:border-sky-800',
    slate: 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700',
    outline: 'bg-transparent text-slate-600 dark:text-slate-400 border border-slate-300 dark:border-slate-700',
  }[variant];

  return (
    <span className={`inline-flex items-center font-medium rounded-full ${sizeClasses} ${variantClasses} ${className}`}>
      {children}
    </span>
  );
};
