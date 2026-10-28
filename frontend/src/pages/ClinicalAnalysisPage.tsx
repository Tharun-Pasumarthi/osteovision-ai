import React, { useEffect, useState } from 'react';
import {
  Stethoscope,
  BarChart2,
  TrendingDown,
  Info,
  CheckCircle2,
  Sliders,
  Scale,
  Calendar,
  Activity,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { api } from '../services/api';
import { ClinicalFeatureImportance } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

export const ClinicalAnalysisPage: React.FC = () => {
  const [importance, setImportance] = useState<ClinicalFeatureImportance[]>([]);
  const [loading, setLoading] = useState(true);

  // Cohort Reference Medians learned strictly on Mendeley training partition
  const cohortMedians = {
    age: 52.0,
    bmi: 26.3,
    menopause_age: 47.0,
    max_walking_km: 2.0,
    weight_kg: 68.0,
  };

  // Interactive patient simulator state
  const [simAge, setSimAge] = useState(62);
  const [simHeight, setSimHeight] = useState(1.58);
  const [simWeight, setSimWeight] = useState(64);
  const [simMenopauseAge, setSimMenopauseAge] = useState(46);
  const [simFracture, setSimFracture] = useState(true);
  const [simFamilyHistory, setSimFamilyHistory] = useState(true);

  const simBmi = Number((simWeight / (simHeight * simHeight)).toFixed(1));

  useEffect(() => {
    let isMounted = true;
    const fetchImportance = async () => {
      try {
        const res = await api.getClinicalImportance();
        if (isMounted) {
          setImportance(res.features);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to load clinical importance:', err);
        if (isMounted) setLoading(false);
      }
    };
    fetchImportance();
    return () => {
      isMounted = false;
    };
  }, []);

  const formatFeatureName = (rawName: string) => {
    const map: Record<string, string> = {
      age: 'Patient Age',
      bmi: 'Calculated BMI',
      weight_kg: 'Body Weight (kg)',
      height_m: 'Height (m)',
      max_walking_km: 'Max Walking Distance',
      menopause_age: 'Menopause Onset Age',
      is_menopausal: 'Menopausal Status',
      gender_female: 'Female Gender',
      hypothyroidism: 'Hypothyroidism History',
      history_of_fracture: 'Prior Fragility Fracture',
      family_history_osteoporosis: 'Family Osteoporosis History',
      smoker: 'Smoking History',
      diabetic: 'Diabetic Indicator',
      joint_pain: 'Joint Pain Symptoms',
    };
    return map[rawName] || rawName.replace('obesity_', '').replace(/_/g, ' ');
  };

  const chartData = importance
    .slice(0, 10)
    .map((item) => ({
      name: formatFeatureName(item.feature_name),
      importance: Number((item.importance_mean * 100).toFixed(1)),
      std: Number((item.importance_std * 100).toFixed(1)),
    }));

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-teal-600" />
          Clinical Variable Analysis & Predictor Importance
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
          Quantifying the systemic diagnostic contribution of non-imaging clinical risk indicators evaluated across the patient cohort.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Feature Importance Chart */}
        <div className="lg:col-span-7 space-y-6">
          <Card>
            <CardHeader
              title="Clinical Predictor Importance Distribution"
              subtitle="Relative predictive weight (%) derived from clinical cohort analysis"
              action={
                <Badge variant="teal" size="sm">
                  Leakage-Controlled
                </Badge>
              }
            />
            <CardContent className="h-96">
              {loading ? (
                <div className="h-full flex items-center justify-center text-xs text-slate-400">
                  Loading feature importances...
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={chartData}
                    layout="vertical"
                    margin={{ top: 10, right: 30, left: 10, bottom: 10 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#e2e8f0" />
                    <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
                    <YAxis dataKey="name" type="category" tick={{ fontSize: 11 }} width={140} />
                    <Tooltip
                      formatter={(val: any) => [`${val}% relative weight`, 'Impact']}
                      contentStyle={{ borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Bar dataKey="importance" fill="#0d9488" radius={[0, 4, 4, 0]}>
                      {chartData.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={entry.importance >= 10 ? '#0d9488' : entry.importance >= 4 ? '#0ea5e9' : '#64748b'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </CardContent>
          </Card>

          {/* Clinical Insights Card */}
          <Card>
            <CardHeader
              title="Clinical Predictor Significance"
              subtitle="Comparison against published epidemiological osteopenia/osteoporosis literature"
            />
            <CardContent className="text-xs space-y-3 text-slate-600 dark:text-slate-400">
              <p>
                <strong className="text-slate-800 dark:text-slate-200">Height & Frame Assessment:</strong> Height loss is
                a hallmark of progressive vertebral compression and subclinical microfractures, explaining its strong
                permutation importance drop when shuffled.
              </p>
              <p>
                <strong className="text-slate-800 dark:text-slate-200">Menopause Age & Estrogen Deprivation:</strong>{' '}
                Accelerated bone turnover following natural or surgical menopause precipitates rapid cancellous
                demineralization, making early menopause an established clinical risk modifier.
              </p>
              <p>
                <strong className="text-slate-800 dark:text-slate-200">Complementary Image-Clinical Fusion:</strong>{' '}
                While CNNs extract localized cortical bone thickness and trabecular micro-texture from radiographs,
                clinical features capture systemic hormonal and metabolic history that imaging alone cannot register.
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Interactive Patient Simulator */}
        <div className="lg:col-span-5 space-y-6">
          <Card>
            <CardHeader
              title="Patient Feature Profiler"
              subtitle="Inspect patient variables against training cohort medians"
              action={
                <Badge variant="blue" size="sm">
                  Interactive
                </Badge>
              }
            />
            <CardContent className="space-y-5 text-xs">
              {/* Age Slider */}
              <div>
                <div className="flex justify-between font-medium mb-1.5">
                  <span>Patient Age:</span>
                  <span className="font-mono text-teal-600 font-bold">{simAge} years</span>
                </div>
                <input
                  type="range"
                  min={30}
                  max={90}
                  value={simAge}
                  onChange={(e) => setSimAge(Number(e.target.value))}
                  className="w-full accent-teal-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>Younger (30y)</span>
                  <span>Cohort Median: 52y</span>
                  <span>Older (90y)</span>
                </div>
              </div>

              {/* BMI Derived Slider */}
              <div>
                <div className="flex justify-between font-medium mb-1.5">
                  <span>Calculated BMI:</span>
                  <span className="font-mono text-sky-600 font-bold">{simBmi} kg/m²</span>
                </div>
                <div className="grid grid-cols-2 gap-3 mb-1">
                  <div>
                    <label className="text-[10px] text-slate-400">Height (m)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={simHeight}
                      onChange={(e) => setSimHeight(Number(e.target.value))}
                      className="w-full px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900"
                    />
                  </div>
                  <div>
                    <label className="text-[10px] text-slate-400">Weight (kg)</label>
                    <input
                      type="number"
                      value={simWeight}
                      onChange={(e) => setSimWeight(Number(e.target.value))}
                      className="w-full px-2 py-1 rounded border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900"
                    />
                  </div>
                </div>
                <div className="text-[10px] text-slate-400">Cohort Median: 26.3 kg/m²</div>
              </div>

              {/* Menopause Age */}
              <div>
                <div className="flex justify-between font-medium mb-1.5">
                  <span>Menopause Age:</span>
                  <span className="font-mono text-amber-600 font-bold">{simMenopauseAge} years</span>
                </div>
                <input
                  type="range"
                  min={35}
                  max={60}
                  value={simMenopauseAge}
                  onChange={(e) => setSimMenopauseAge(Number(e.target.value))}
                  className="w-full accent-amber-600"
                />
                <div className="flex justify-between text-[10px] text-slate-400 mt-0.5">
                  <span>Early (&lt;40y)</span>
                  <span>Cohort Median: 47y</span>
                  <span>Late (&gt;55y)</span>
                </div>
              </div>

              {/* Historical Indicators */}
              <div className="pt-3 border-t border-slate-100 dark:border-slate-800 space-y-2">
                <label className="flex items-center justify-between p-2 rounded-lg bg-slate-50 dark:bg-slate-800/80 cursor-pointer">
                  <span>Prior Fragility Fracture</span>
                  <input
                    type="checkbox"
                    checked={simFracture}
                    onChange={(e) => setSimFracture(e.target.checked)}
                    className="rounded text-teal-600"
                  />
                </label>
                <label className="flex items-center justify-between p-2 rounded-lg bg-slate-50 dark:bg-slate-800/80 cursor-pointer">
                  <span>Family History of Osteoporosis</span>
                  <input
                    type="checkbox"
                    checked={simFamilyHistory}
                    onChange={(e) => setSimFamilyHistory(e.target.checked)}
                    className="rounded text-teal-600"
                  />
                </label>
              </div>

              {/* Cohort Comparison Summary Card */}
              <div className="p-3.5 rounded-xl bg-teal-50/70 dark:bg-teal-950/40 border border-teal-200/80 dark:border-teal-800/60">
                <div className="flex items-center gap-2 font-semibold text-teal-900 dark:text-teal-200 mb-1.5">
                  <CheckCircle2 className="w-4 h-4 text-teal-600" />
                  <span>Profile Assessment Summary</span>
                </div>
                <p className="text-slate-700 dark:text-slate-300 leading-relaxed text-[11px]">
                  {simAge > cohortMedians.age ? 'Patient age is above' : 'Patient age is below'} the research cohort
                  median ({cohortMedians.age}y).{' '}
                  {simFracture || simFamilyHistory
                    ? 'Positive family or fragility fracture history increases systemic risk weighting in the multimodal classifier.'
                    : 'Absence of secondary fracture indicators corresponds to lower baseline clinical prior.'}
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
