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
} from 'lucide-react';
import { api } from '../services/api';
import { ModelsListResponse } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export const OverviewPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelsListResponse | null>(null);

  useEffect(() => {
    let isMounted = true;
    const loadOverviewData = async () => {
      try {
        const modelsRes = await api.getModels();
        if (isMounted) {
          setModels(modelsRes);
        }
      } catch (err) {
        console.error('Failed to load models data:', err);
      }
    };
    loadOverviewData();
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-slate-900 via-slate-850 to-teal-950 text-white p-8 shadow-sm border border-slate-800">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-teal-500/20 text-teal-300 border border-teal-500/30 text-xs font-semibold uppercase tracking-wider mb-3">
            <Microscope className="w-3.5 h-3.5" />
            Medical Radiography Research Platform
          </div>
          <h2 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
            Multimodal Knee X-Ray Osteoporosis Classification
          </h2>
          <p className="mt-3 text-sm text-slate-300 leading-relaxed">
            Integrating mandatory systemic clinical risk factors with optional knee radiograph deep learning
            to deliver robust bone mineral density classification across Normal, Osteopenia, and Osteoporosis.
          </p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button
              onClick={() => navigate('/analysis')}
              variant="primary"
              size="md"
              leftIcon={<ScanEye className="w-4 h-4" />}
            >
              Open Diagnostic Console
            </Button>
            <Button
              onClick={() => navigate('/comparison')}
              variant="outline"
              size="md"
              className="bg-slate-800/80 border-slate-700 text-white hover:bg-slate-750"
              leftIcon={<BarChart3 className="w-4 h-4" />}
            >
              Model Benchmarks
            </Button>
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-teal-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Models Evaluated</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                {models?.models.length || 14}
              </h3>
              <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mt-0.5">
                PyTorch &amp; Sklearn Active
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
                Mandatory Clinical Schema
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
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Deduplicated Cohort</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                865
              </h3>
              <p className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium mt-0.5">
                Clean Patient Examinations
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-emerald-50 dark:bg-emerald-950 text-emerald-600 dark:text-emerald-400 flex items-center justify-center">
              <Database className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-purple-500">
          <CardContent className="p-4 flex items-center justify-between">
            <div>
              <p className="text-xs font-medium text-slate-500 dark:text-slate-400">Leakage Audit Status</p>
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100 mt-1">
                0% Leakage
              </h3>
              <p className="text-[11px] text-purple-600 dark:text-purple-400 font-medium mt-0.5">
                Patient-Grouped Isolation
              </p>
            </div>
            <div className="w-11 h-11 rounded-xl bg-purple-50 dark:bg-purple-950 text-purple-600 dark:text-purple-400 flex items-center justify-center">
              <ShieldCheck className="w-6 h-6" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Quick Launch Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <Card className="hover:border-teal-500 dark:hover:border-teal-500 transition-all cursor-pointer group" onClick={() => navigate('/analysis')}>
          <CardHeader
            title="Multimodal Diagnostic Console"
            subtitle="Clinical data mandatory • Knee radiograph optional"
            action={<ScanEye className="w-5 h-5 text-teal-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p leading-relaxed>
              Enter patient systemic risk variables and optionally attach a knee radiograph to evaluate bone mineral status using deep multimodal fusion.
            </p>
            <div className="flex items-center text-teal-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Launch Diagnostic Console</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        <Card className="hover:border-sky-500 dark:hover:border-sky-500 transition-all cursor-pointer group" onClick={() => navigate('/clinical')}>
          <CardHeader
            title="Clinical Feature Profiler"
            subtitle="Explore predictor significance"
            action={<Stethoscope className="w-5 h-5 text-sky-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p leading-relaxed>
              Analyze the permutation importance of individual clinical indicators including age, height loss, BMI, and hormonal history across cohorts.
            </p>
            <div className="flex items-center text-sky-600 font-semibold gap-1 text-[11px] pt-1">
              <span>View Clinical Predictors</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>

        <Card className="hover:border-purple-500 dark:hover:border-purple-500 transition-all cursor-pointer group" onClick={() => navigate('/comparison')}>
          <CardHeader
            title="14-Model Benchmarks"
            subtitle="Performance metrics &amp; comparisons"
            action={<BarChart3 className="w-5 h-5 text-purple-600 transition-transform group-hover:scale-110" />}
          />
          <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
            <p leading-relaxed>
              Review verified Macro-F1, accuracy, and confusion matrices across CNN backbones, clinical classifiers, and multimodal fusion architectures.
            </p>
            <div className="flex items-center text-purple-600 font-semibold gap-1 text-[11px] pt-1">
              <span>Inspect Benchmarks</span>
              <ArrowRight className="w-3.5 h-3.5 transition-transform group-hover:translate-x-1" />
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default OverviewPage;
