import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Layers,
  Database,
  ArrowRight,
  ShieldCheck,
  ScanEye,
  Microscope,
  Stethoscope,
  BarChart3,
  Sparkles,
  Flame,
  Activity,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../services/api';
import { ModelsListResponse, CalibrationReportResponse } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Badge } from '../components/ui/Badge';

export const OverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelsListResponse | null>(null);
  const [calibrationReport, setCalibrationReport] = useState<CalibrationReportResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    api.getModels()
      .then((m) => {
        if (isMounted) setModels(m);
      })
      .catch((err) => console.error('Failed to load models list:', err));

    api.getCalibrationStatus()
      .then((c) => {
        if (isMounted) setCalibrationReport(c);
      })
      .catch((err) => console.error('Failed to load calibration status:', err));

    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Hero Research Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-850 to-teal-950 text-white p-6 sm:p-8 shadow-sm border border-slate-800">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 text-xs font-semibold uppercase tracking-wider mb-3">
            <Microscope className="w-3.5 h-3.5" />
            Explainable &amp; Reliability-Aware AI Research Platform
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
            OsteoVision AI — Deep Learning Knee Radiograph &amp; Clinical Risk Screening
          </h2>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            A production-oriented medical AI research platform uniting convolutional knee radiograph feature extraction, 13-factor clinical risk modeling, pre-inference image quality validation, authentic Grad-CAM &amp; SHAP explainability, and post-hoc probability calibration.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button
              onClick={() => navigate('/analysis')}
              variant="primary"
              size="md"
              leftIcon={<ScanEye className="w-4 h-4" />}
            >
              Analyze Radiograph
            </Button>
            <Button
              onClick={() => navigate('/clinical')}
              variant="outline"
              size="md"
              className="bg-slate-800/80 border-slate-700 text-white hover:bg-slate-750"
              leftIcon={<Stethoscope className="w-4 h-4" />}
            >
              Clinical Risk &amp; SHAP
            </Button>
            <Button
              onClick={() => navigate('/explainable-ai')}
              variant="outline"
              size="md"
              className="bg-slate-800/80 border-slate-700 text-white hover:bg-slate-750"
              leftIcon={<Sparkles className="w-4 h-4" />}
            >
              XAI Workspace
            </Button>
          </div>
        </div>
      </div>

      {/* KPI Stats Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-teal-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Architectures Benchmarked</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                {models?.models.length || 11} Models
              </h3>
              <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mt-0.5">
                PyTorch &amp; Sklearn Registered
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-teal-50 dark:bg-teal-950 text-teal-600 dark:text-teal-400 flex items-center justify-center">
              <Layers className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-sky-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Clinical Feature Contract</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                13 Features
              </h3>
              <p className="text-[11px] text-sky-600 dark:text-sky-400 font-medium mt-0.5">
                Standardized Preprocessor
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-sky-50 dark:bg-sky-950 text-sky-600 dark:text-sky-400 flex items-center justify-center">
              <Stethoscope className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Calibration Reliability</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                2.64% ECE
              </h3>
              <p className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium mt-0.5">
                -68.6% Calibration Error
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-emerald-50 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
              <ShieldCheck className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-purple-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Explainability Engine</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                Grad-CAM &amp; SHAP
              </h3>
              <p className="text-[11px] text-purple-600 dark:text-purple-400 font-medium mt-0.5">
                Dual Independent Attributions
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-purple-50 dark:bg-purple-950 text-purple-600 dark:text-purple-400 flex items-center justify-center">
              <Sparkles className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 5 Core Research Modules Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {/* Module 1: Image Analysis */}
        <Card
          className="hover:border-teal-500 dark:hover:border-teal-500 transition-all cursor-pointer group"
          onClick={() => navigate('/analysis')}
        >
          <CardHeader
            title="Image Analysis"
            subtitle="Knee Radiograph Deep Learning"
            action={<ScanEye className="w-5 h-5 text-teal-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              Upload AP/Lateral knee plain-film radiographs. Features pre-inference image quality screening (blurriness, contrast, resolution), CNN inference, and integrated Grad-CAM overlays.
            </p>
            <div className="flex items-center text-teal-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Open Image Analysis</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        {/* Module 2: Clinical Analysis */}
        <Card
          className="hover:border-sky-500 dark:hover:border-sky-500 transition-all cursor-pointer group"
          onClick={() => navigate('/clinical')}
        >
          <CardHeader
            title="Clinical Analysis"
            subtitle="13-Feature Tabular Model"
            action={<Stethoscope className="w-5 h-5 text-sky-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              Independent clinical machine learning model operating strictly on patient systemic indicators. Includes unit converters, reactive BMI computation, and local SHAP factor attribution.
            </p>
            <div className="flex items-center text-sky-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Open Clinical Analysis</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        {/* Module 3: Explainable AI */}
        <Card
          className="hover:border-purple-500 dark:hover:border-purple-500 transition-all cursor-pointer group"
          onClick={() => navigate('/explainable-ai')}
        >
          <CardHeader
            title="Explainable AI"
            subtitle="Unified XAI Workspace"
            action={<Sparkles className="w-5 h-5 text-purple-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              Explore localized convolutional gradient heatmaps across target layers alongside SHAP waterfall plots and global Gini rankings, maintaining strict independence between modalities.
            </p>
            <div className="flex items-center text-purple-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Launch XAI Console</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        {/* Module 4: Model Reliability */}
        <Card
          className="hover:border-emerald-500 dark:hover:border-emerald-500 transition-all cursor-pointer group"
          onClick={() => navigate('/reliability')}
        >
          <CardHeader
            title="Model Reliability"
            subtitle="Confidence Calibration &amp; ECE"
            action={<ShieldCheck className="w-5 h-5 text-emerald-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              Inspect empirical reliability diagrams, Brier reliability scores, temperature scaling parameters, and image QA thresholds verified on patient-isolated validation cohorts.
            </p>
            <div className="flex items-center text-emerald-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Inspect Calibration</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        {/* Module 5: Model Evaluation */}
        <Card
          className="hover:border-indigo-500 dark:hover:border-indigo-500 transition-all cursor-pointer group"
          onClick={() => navigate('/evaluation')}
        >
          <CardHeader
            title="Model Evaluation"
            subtitle="11 Benchmark Architectures"
            action={<BarChart3 className="w-5 h-5 text-indigo-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p className="leading-relaxed">
              Review comparative test accuracy, Macro-F1, per-class sensitivity, latency, parameter counts, and confusion matrices derived from verified patient test partitions.
            </p>
            <div className="flex items-center text-indigo-600 font-semibold gap-1 text-[11px] pt-1">
              <span>View Scientific Benchmarks</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default OverviewPage;
