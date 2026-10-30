import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanEye,
  Stethoscope,
  Sparkles,
  ShieldCheck,
  BarChart3,
  ChevronRight,
  X,
  Microscope,
} from 'lucide-react';

export const NAV_ITEMS = [
  { path: '/', label: 'Overview', icon: LayoutDashboard, badge: null },
  { path: '/analysis', label: 'Image Analysis', icon: ScanEye, badge: 'IQA' },
  { path: '/clinical', label: 'Clinical Analysis', icon: Stethoscope, badge: '13 Feat' },
  { path: '/explainable-ai', label: 'Explainable AI', icon: Sparkles, badge: 'CAM+SHAP' },
  { path: '/reliability', label: 'Model Reliability', icon: ShieldCheck, badge: 'Calibrated' },
  { path: '/evaluation', label: 'Model Evaluation', icon: BarChart3, badge: '11 Models' },
];

export interface SidebarProps {
  isOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isOpen = false, onClose }) => {
  return (
    <>
      {/* Mobile Backdrop Blur Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/60 backdrop-blur-xs z-40 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-50 md:z-auto w-64 bg-white dark:bg-slate-900 border-r border-slate-200 dark:border-slate-800 flex flex-col shrink-0 select-none transform transition-transform duration-200 ease-in-out md:translate-x-0 ${
          isOpen ? 'translate-x-0 shadow-2xl' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Mobile Header inside drawer */}
        <div className="flex md:hidden items-center justify-between p-3.5 border-b border-slate-200 dark:border-slate-800">
          <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
            OsteoVision Menu
          </span>
          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-3.5 sm:p-4 border-b border-slate-100 dark:border-slate-800/80">
          <div className="bg-gradient-to-br from-teal-500/10 via-slate-100 to-sky-500/10 dark:from-teal-950/40 dark:via-slate-800/50 dark:to-sky-950/40 border border-teal-500/20 rounded-xl p-2.5 sm:p-3">
            <div className="flex items-center gap-2 text-teal-700 dark:text-teal-300 font-bold text-xs">
              <Microscope className="w-4 h-4" />
              <span>OsteoVision AI Platform</span>
            </div>
            <p className="text-[11px] text-slate-600 dark:text-slate-400 mt-1 leading-relaxed">
              Explainable &amp; reliability-aware knee X-ray radiograph and clinical risk screening.
            </p>
          </div>
        </div>

        {/* Navigation Link List */}
        <nav aria-label="Sidebar Navigation" className="flex-1 p-2.5 sm:p-3 space-y-1 overflow-y-auto">
          <span className="px-3 text-[10px] font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider block mb-1">
            Research Navigation
          </span>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                onClick={onClose}
                className={({ isActive }) =>
                  `flex items-center justify-between px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
                    isActive
                      ? 'bg-teal-50 dark:bg-teal-950/60 text-teal-700 dark:text-teal-300 font-semibold shadow-2xs'
                      : 'text-slate-600 dark:text-slate-400 hover:bg-slate-100/80 dark:hover:bg-slate-800/60 hover:text-slate-900 dark:hover:text-slate-200'
                  }`
                }
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <Icon className="w-4 h-4 shrink-0" />
                  <span className="truncate">{item.label}</span>
                </div>
                {item.badge ? (
                  <span className="text-[9px] px-1.5 py-0.5 rounded-full font-semibold uppercase tracking-wider bg-teal-100/80 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300">
                    {item.badge}
                  </span>
                ) : (
                  <ChevronRight className="w-3.5 h-3.5 opacity-40 shrink-0" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Footer Info */}
        <div className="p-3 border-t border-slate-100 dark:border-slate-800 text-[10px] text-slate-400 space-y-1">
          <p className="font-semibold text-slate-600 dark:text-slate-300">Model Registry v2.1</p>
          <p>DenseNet121 &bull; Random Forest (Platt)</p>
          <p className="text-[9px] text-slate-400 dark:text-slate-500 mt-0.5">
            Research Use Only &bull; Zero PHI Retained
          </p>
        </div>
      </aside>
    </>
  );
};
