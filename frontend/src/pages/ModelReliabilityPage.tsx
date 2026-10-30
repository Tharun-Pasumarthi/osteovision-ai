import React, { useState, useEffect } from 'react';
import {
  ShieldCheck,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Info,
  Scale,
  Sliders,
  BarChart3,
  HelpCircle,
  FileCheck,
  TrendingDown,
  Layers,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ReferenceLine,
} from 'recharts';
import { api } from '../services/api';
import { CalibrationReportResponse } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

export const ModelReliabilityPage: React.FC = () => {
  const [report, setReport] = useState<CalibrationReportResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [activeModelView, setActiveModelView] = useState<'image' | 'clinical'>('image');

  useEffect(() => {
    let isMounted = true;
    api.getCalibrationStatus()
      .then((data) => {
        if (isMounted) {
          setReport(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        console.error('Failed to load calibration status:', err);
        if (isMounted) setLoading(false);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  const activeProfile = activeModelView === 'image'
    ? report?.image_model_calibration
    : report?.clinical_model_calibration;

  // Format chart data for reliability diagram
  const chartData = (activeProfile?.reliability_diagram || []).map((bin) => ({
    confidence: Number((bin.bin_center * 100).toFixed(0)),
    perfect: Number((bin.bin_center * 100).toFixed(0)),
    rawAccuracy: Number((bin.raw_accuracy * 100).toFixed(1)),
    calibratedAccuracy: Number((bin.calibrated_accuracy * 100).toFixed(1)),
    sampleCount: bin.sample_count,
  }));

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      {/* Page Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-teal-600 dark:text-teal-400" />
            Model Reliability &amp; Probability Calibration
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Empirical validation of confidence calibration, Expected Calibration Error (ECE), and Brier reliability scores.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="teal" size="sm">
            Validation-Partition Fitted
          </Badge>
          <Badge variant="blue" size="sm">
            Zero Test-Leakage
          </Badge>
        </div>
      </div>

      {/* Primary KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="border-l-4 border-l-teal-500">
          <CardContent className="p-4">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Expected Calibration Error (ECE)
            </span>
            <div className="flex items-baseline gap-2 mt-1">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {activeProfile ? `${(activeProfile.calibrated_metrics.ece * 100).toFixed(2)}%` : '--'}
              </h3>
              <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                -{activeProfile?.ece_reduction_percent}% error
              </span>
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Raw baseline ECE was {activeProfile ? `${(activeProfile.raw_metrics.ece * 100).toFixed(2)}%` : '--'}
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-sky-500">
          <CardContent className="p-4">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Multi-Class Brier Score
            </span>
            <div className="flex items-baseline gap-2 mt-1">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {activeProfile ? activeProfile.calibrated_metrics.brier_score.toFixed(4) : '--'}
              </h3>
              <span className="text-xs text-slate-400">
                (Raw: {activeProfile ? activeProfile.raw_metrics.brier_score.toFixed(4) : '--'})
              </span>
            </div>
            <p className="text-[11px] text-teal-600 dark:text-teal-400 font-medium mt-0.5">
              Strictly lower quadratic penalty
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-indigo-500">
          <CardContent className="p-4">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Maximum Calibration Error (MCE)
            </span>
            <div className="flex items-baseline gap-2 mt-1">
              <h3 className="text-2xl font-bold text-slate-900 dark:text-slate-100">
                {activeProfile ? `${(activeProfile.calibrated_metrics.mce * 100).toFixed(1)}%` : '--'}
              </h3>
              <span className="text-xs text-slate-400 font-mono">
                Worst-bin gap
              </span>
            </div>
            <p className="text-[11px] text-slate-500 mt-0.5">
              Reduced from {activeProfile ? `${(activeProfile.raw_metrics.mce * 100).toFixed(1)}%` : '--'}
            </p>
          </CardContent>
        </Card>

        <Card className="border-l-4 border-l-emerald-500">
          <CardContent className="p-4">
            <span className="text-xs font-medium text-slate-500 dark:text-slate-400">
              Calibration Methodology
            </span>
            <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 mt-1 truncate">
              {activeProfile ? activeProfile.calibration_method : '--'}
            </h3>
            <p className="text-[11px] text-emerald-600 dark:text-emerald-400 font-medium mt-0.5">
              Verified Post-Hoc Alignment
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Model Selection Selector */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">Model Pathway:</span>
        <div className="inline-flex rounded-lg border border-slate-200 dark:border-slate-800 p-0.5 bg-white dark:bg-slate-900">
          <button
            type="button"
            onClick={() => setActiveModelView('image')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
              activeModelView === 'image'
                ? 'bg-teal-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            Deep Learning Radiograph CNNs (DenseNet121)
          </button>
          <button
            type="button"
            onClick={() => setActiveModelView('clinical')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition ${
              activeModelView === 'clinical'
                ? 'bg-teal-600 text-white shadow-xs'
                : 'text-slate-600 dark:text-slate-400 hover:text-slate-900'
            }`}
          >
            Clinical Risk ML (Random Forest Platt Scaled)
          </button>
        </div>
      </div>

      {/* Main Grid: Reliability Diagram & Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (8 cols): Interactive Reliability Diagram */}
        <div className="lg:col-span-8 space-y-5">
          <Card>
            <CardHeader
              title="Empirical Reliability Diagram (Calibration Curve)"
              subtitle="Confidence bins vs Observed accuracy comparing uncalibrated predictions against validation calibration"
              action={
                <Badge variant="teal" size="sm">
                  10 Equal Bins (0% - 100%)
                </Badge>
              }
            />
            <CardContent className="h-96">
              {loading ? (
                <div className="h-full flex items-center justify-center text-xs text-slate-400">
                  Loading calibration curves...
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="confidence"
                      unit="%"
                      label={{ value: 'Forecasted Confidence', position: 'insideBottom', offset: -10, fontSize: 11 }}
                      tick={{ fontSize: 11 }}
                    />
                    <YAxis
                      unit="%"
                      domain={[0, 100]}
                      label={{ value: 'Empirical Accuracy', angle: -90, position: 'insideLeft', fontSize: 11 }}
                      tick={{ fontSize: 11 }}
                    />
                    <Tooltip
                      formatter={(val: any, name: string) => [
                        `${val}%`,
                        name === 'perfect'
                          ? 'Perfect Calibration'
                          : name === 'calibratedAccuracy'
                          ? 'Calibrated Accuracy'
                          : 'Raw Accuracy',
                      ]}
                      contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    {/* Perfect Calibration Diagonal Line */}
                    <Line
                      type="monotone"
                      dataKey="perfect"
                      name="Ideal Calibration (y=x)"
                      stroke="#94a3b8"
                      strokeDasharray="5 5"
                      strokeWidth={1.5}
                      dot={false}
                    />
                    {/* Uncalibrated Raw Curve */}
                    <Line
                      type="monotone"
                      dataKey="rawAccuracy"
                      name="Raw Model Softmax (Overconfident)"
                      stroke="#f43f5e"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    {/* Calibrated Model Curve */}
                    <Line
                      type="monotone"
                      dataKey="calibratedAccuracy"
                      name="Post-Hoc Calibrated Probabilities"
                      stroke="#0d9488"
                      strokeWidth={3}
                      dot={{ r: 4, fill: '#0d9488' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Reliability Observations */}
          <Card>
            <CardHeader
              title="Calibration Methodology &amp; Clinical Impact"
              subtitle="Why post-hoc calibration is essential for medical decision support"
            />
            <CardContent className="text-xs text-slate-600 dark:text-slate-400 space-y-3 leading-relaxed">
              <p>
                {activeProfile?.notes}
              </p>
              <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-[11px] space-y-1">
                <span className="font-semibold text-slate-800 dark:text-slate-200">
                  Regulatory &amp; Research Compliance:
                </span>
                <p>
                  Calibration parameters are strictly determined on validation partition patient records and are never fitted using the final test cohort. Test set integrity is strictly isolated.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column (4 cols): Quality QA Benchmarks & Uncertainty Policies */}
        <div className="lg:col-span-4 space-y-5">
          <Card>
            <CardHeader
              title="Image Quality QA Benchmarks"
              subtitle="Pre-inference computer vision screening thresholds"
            />
            <CardContent className="space-y-3 text-xs">
              <div className="p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900">
                <span className="font-semibold text-slate-800 dark:text-slate-200">Blurriness (Laplacian Var):</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  &gt; 120.0 = Sharp; 40.0 - 120.0 = Borderline; &lt; 40.0 = Severe Blur (Unusable).
                </p>
              </div>

              <div className="p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900">
                <span className="font-semibold text-slate-800 dark:text-slate-200">Radiographic Exposure:</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Mean brightness 45 - 215/255. Extreme underexposure (&lt;25) or burnout (&gt;240) flagged.
                </p>
              </div>

              <div className="p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900">
                <span className="font-semibold text-slate-800 dark:text-slate-200">Dynamic Contrast (Std Dev):</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  &gt; 32.0 = Good; &lt; 18.0 = Flat unusable contrast across trabecular boundaries.
                </p>
              </div>

              <div className="p-2.5 rounded-lg border border-slate-100 dark:border-slate-800 bg-slate-50 dark:bg-slate-900">
                <span className="font-semibold text-slate-800 dark:text-slate-200">Dimension Limits:</span>
                <p className="text-[11px] text-slate-500 mt-0.5">
                  Min 150x150px; Aspect ratio 0.33 to 3.0 to prevent severe anatomical distortion.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader
              title="Uncertainty &amp; Boundary Policies"
              subtitle="Automated clinical flag triggers"
            />
            <CardContent className="space-y-3 text-xs text-slate-600 dark:text-slate-400">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">Equivocal Boundary:</span>
                  <p className="text-[11px] mt-0.5">
                    Triggered when confidence margin between top-2 predicted classes is &lt; 12.0%.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">Low-Confidence Trigger:</span>
                  <p className="text-[11px] mt-0.5">
                    Flagged when maximum class posterior probability is &lt; 50.0%.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-2">
                <Info className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">Mandatory DXA Referral:</span>
                  <p className="text-[11px] mt-0.5">
                    All Osteopenia and Osteoporosis classifications explicitly recommend confirmatory dual-energy X-ray absorptiometry.
                  </p>
                </div>
              </div>

              <p className="text-[10px] text-slate-400 dark:text-slate-500 pt-2 border-t border-slate-100 dark:border-slate-800">
                {report?.disclaimer}
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

