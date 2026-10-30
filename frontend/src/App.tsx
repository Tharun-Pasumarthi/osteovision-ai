import React, { useState, useEffect } from 'react';
import { BrowserRouter, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import { DisclaimerBanner } from './components/layout/DisclaimerBanner';
import { Header } from './components/layout/Header';
import { Sidebar, NAV_ITEMS } from './components/layout/Sidebar';
import { OverviewPage } from './pages/OverviewPage';
import { ImageAnalysisPage } from './pages/ImageAnalysisPage';
import { ClinicalAnalysisPage } from './pages/ClinicalAnalysisPage';
import { ExplainableAIPage } from './pages/ExplainableAIPage';
import { ModelReliabilityPage } from './pages/ModelReliabilityPage';
import { ModelEvaluationPage } from './pages/ModelEvaluationPage';
import { DatasetPage } from './pages/DatasetPage';

export const App: React.FC = () => {
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    return localStorage.getItem('theme') === 'dark';
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState<boolean>(false);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 transition-colors">
        {/* Research Disclaimer Banner */}
        <DisclaimerBanner />

        {/* Global Navigation Header */}
        <Header
          darkMode={darkMode}
          setDarkMode={setDarkMode}
          isMobileMenuOpen={mobileMenuOpen}
          onToggleMobileMenu={() => setMobileMenuOpen((prev) => !prev)}
        />

        {/* Main Content Area with Responsive Sidebar & Mobile Drawer */}
        <div className="flex-1 flex overflow-hidden relative">
          <Sidebar
            isOpen={mobileMenuOpen}
            onClose={() => setMobileMenuOpen(false)}
          />

          <main className="flex-1 overflow-y-auto p-3.5 sm:p-6 lg:p-8 pb-18 md:pb-8">
            <Routes>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/analysis" element={<ImageAnalysisPage />} />
              <Route path="/image-analysis" element={<Navigate to="/analysis" replace />} />
              <Route path="/clinical" element={<ClinicalAnalysisPage />} />
              <Route path="/explainable-ai" element={<ExplainableAIPage />} />
              <Route path="/reliability" element={<ModelReliabilityPage />} />
              <Route path="/evaluation" element={<ModelEvaluationPage />} />
              <Route path="/comparison" element={<Navigate to="/evaluation" replace />} />
              <Route path="/dataset" element={<DatasetPage />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </main>
        </div>

        {/* Mobile Compact Bottom Navigation Bar */}
        <nav
          aria-label="Mobile Bottom Navigation"
          className="md:hidden fixed bottom-0 inset-x-0 z-40 bg-white/95 dark:bg-slate-900/95 backdrop-blur-md border-t border-slate-200 dark:border-slate-800 px-1 py-1.5 flex items-center justify-around shadow-lg"
        >
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const shortLabel = item.path === '/'
              ? 'Home'
              : item.path === '/analysis'
              ? 'Images'
              : item.path === '/clinical'
              ? 'Clinical'
              : item.path === '/explainable-ai'
              ? 'XAI'
              : item.path === '/reliability'
              ? 'Reliability'
              : 'Eval';

            return (
              <NavLink
                key={item.path}
                to={item.path}
                end={item.path === '/'}
                className={({ isActive }) =>
                  `flex flex-col items-center justify-center py-1 px-1.5 rounded-lg text-[10px] font-medium transition ${
                    isActive
                      ? 'text-teal-600 dark:text-teal-400 font-bold'
                      : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                  }`
                }
              >
                <Icon className="w-4 h-4 mb-0.5" />
                <span className="truncate max-w-[50px]">{shortLabel}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>
    </BrowserRouter>
  );
};

export default App;
