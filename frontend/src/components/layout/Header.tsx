import React, { useEffect, useState } from 'react';
import { Activity, Sun, Moon, Cpu, AlertCircle, Menu, X } from 'lucide-react';
import { api } from '../../services/api';
import { HealthResponse } from '../../types/api';

interface HeaderProps {
  darkMode: boolean;
  setDarkMode: (val: boolean) => void;
  onToggleMobileMenu?: () => void;
  isMobileMenuOpen?: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  darkMode,
  setDarkMode,
  onToggleMobileMenu,
  isMobileMenuOpen,
}) => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const data = await api.getHealth();
        if (isMounted) {
          setHealth(data);
          setLoading(false);
        }
      } catch (err) {
        if (isMounted) setLoading(false);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="sticky top-0 z-40 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border-b border-slate-200 dark:border-slate-800 px-3 sm:px-6 py-2.5 sm:py-3.5 flex items-center justify-between">
      <div className="flex items-center gap-2.5 sm:gap-3">
        {/* Mobile Hamburger Menu Button */}
        {onToggleMobileMenu && (
          <button
            type="button"
            onClick={onToggleMobileMenu}
            className="md:hidden p-1.5 rounded-lg text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-none"
            aria-label="Toggle navigation drawer"
          >
            {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </button>
        )}

        <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-lg bg-teal-600 text-white flex items-center justify-center shadow-xs shrink-0">
          <Activity className="w-4 h-4 sm:w-5 sm:h-5" />
        </div>
        <div>
          <div className="flex items-center gap-1.5 sm:gap-2">
            <h1 className="text-sm sm:text-base font-bold text-slate-900 dark:text-slate-100 tracking-tight">
              OsteoVision AI
            </h1>
            <span className="text-[9px] sm:text-[10px] font-semibold tracking-wider uppercase px-1.5 sm:px-2 py-0.5 rounded-full bg-teal-50 dark:bg-teal-950 text-teal-700 dark:text-teal-300 border border-teal-200 dark:border-teal-800">
              v1.0
            </span>
          </div>
          <p className="hidden sm:block text-xs text-slate-500 dark:text-slate-400">
            Knee Radiograph Osteoporosis, Osteopenia &amp; Normal Deep Learning Platform
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        {/* Backend health status badge */}
        <div className="flex items-center gap-1.5 sm:gap-2 px-2 sm:px-3 py-1 sm:py-1.5 rounded-lg bg-slate-50 dark:bg-slate-800/80 border border-slate-200 dark:border-slate-700 text-[11px] sm:text-xs">
          {health?.status === 'healthy' ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </span>
              <span className="font-medium text-slate-700 dark:text-slate-200">Live</span>
              <span className="hidden sm:inline text-slate-400">|</span>
              <span className="hidden sm:flex items-center gap-1 text-slate-500 dark:text-slate-400 text-[11px]">
                <Cpu className="w-3 h-3 text-teal-600 dark:text-teal-400" />
                {health.device.toUpperCase()}
              </span>
            </>
          ) : loading ? (
            <span className="text-slate-400 text-[10px] sm:text-xs">Connecting...</span>
          ) : (
            <>
              <AlertCircle className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-amber-500" />
              <span className="font-medium text-amber-700 dark:text-amber-400 text-[10px] sm:text-xs">Offline</span>
            </>
          )}
        </div>

        {/* Theme toggle */}
        <button
          onClick={() => setDarkMode(!darkMode)}
          className="p-1.5 sm:p-2 rounded-lg text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 hover:bg-slate-100 dark:hover:bg-slate-750 cursor-pointer"
          aria-label="Toggle theme"
          title="Toggle light/dark theme"
        >
          {darkMode ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-600" />}
        </button>
      </div>
    </header>
  );
};
