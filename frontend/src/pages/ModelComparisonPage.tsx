import React, { useEffect, useState } from 'react';
import {
  BarChart3,
  Award,
  Layers,
  ArrowUpDown,
  Clock,
  Cpu,
  CheckCircle2,
  Table,
  SlidersHorizontal,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../services/api';
import { BenchmarkMetricsResponse, BenchmarkModelRecord } from '../types/api';
import { Card, CardHeader, CardContent } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

type SortField =
  | 'test_macro_f1'
  | 'val_macro_f1'
  | 'test_accuracy'
  | 'test_balanced_acc'
  | 'osteoporosis_recall'
  | 'osteopenia_recall'
  | 'normal_recall'
  | 'inference_latency_ms';

export const ModelComparisonPage: React.FC = () => {
  const [metrics, setMetrics] = useState<BenchmarkMetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [sortField, setSortField] = useState<SortField>('test_macro_f1');
  const [sortAsc, setSortAsc] = useState(false);
  const [familyFilter, setFamilyFilter] = useState<string>('all');
  const [selectedConfusionModel, setSelectedConfusionModel] = useState<string>('MULTIMODAL_FUSION_DENSENET_CLINICAL');

  useEffect(() => {
    let isMounted = true;
    const fetchMetrics = async () => {
      try {
        const data = await api.getMetrics();
        if (isMounted) {
          setMetrics(data);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to load benchmark metrics:', err);
        if (isMounted) setLoading(false);
      }
    };
    fetchMetrics();
    return () => {
      isMounted = false;
    };
  }, []);

  // Filter models
  const filteredModels = (metrics?.models || []).filter((m) => {
    if (familyFilter === 'all') return true;
    if (familyFilter === 'dl') return m.model_family.includes('Deep Learning');
    if (familyFilter === 'multimodal') return m.model_family.includes('Multimodal');
    if (familyFilter === 'clinical') return m.model_family.includes('Clinical');
    if (familyFilter === 'embeddings') return m.model_family.includes('Embedding');
    return true;
  });

  // Sort models
  const sortedModels = [...filteredModels].sort((a, b) => {
    const valA = (a[sortField] as number) ?? -1;
    const valB = (b[sortField] as number) ?? -1;
    return sortAsc ? valA - valB : valB - valA;
  });

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  // Prepare chart data for per-class sensitivity across models
  const perClassChartData = sortedModels.slice(0, 7).map((m) => ({
    name: m.model_name
      .replace(' (Embeddings + Latent AI Gen)', ' + AI Gen')
      .replace(' (Embeddings)', ' Embed')
      .replace(' (Clinical)', ' Clin')
      .replace('Multimodal Fusion (DenseNet121 + Clinical)', 'Multimodal'),
    Normal: Number(((m.normal_recall || 0) * 100).toFixed(1)),
    Osteopenia: Number(((m.osteopenia_recall || 0) * 100).toFixed(1)),
    Osteoporosis: Number(((m.osteoporosis_recall || 0) * 100).toFixed(1)),
  }));

  // Selected confusion matrix data
  const selectedModelRecord = metrics?.models.find((m) => m.experiment_id === selectedConfusionModel);
  const confusionMatrix = selectedModelRecord?.raw_metrics_test?.confusion_matrix || [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];

  return (
    <div className="space-y-6">
      {/* Title & Filter Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-teal-600" />
            14-Model Scientific Benchmark & Architecture Comparison
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Evaluated on untouched patient-grouped partitions (131 independent test radiographs) with 0% data leakage.
          </p>
        </div>

        {/* Family filter pills */}
        <div className="flex flex-wrap items-center gap-1.5 p-1 rounded-xl bg-slate-100 dark:bg-slate-800 text-xs">
          {[
            { id: 'all', label: 'All Architectures' },
            { id: 'dl', label: 'CNN Image' },
            { id: 'multimodal', label: 'Multimodal' },
            { id: 'embeddings', label: 'Embedding ML' },
            { id: 'clinical', label: 'Clinical ML' },
          ].map((f) => (
            <button
              key={f.id}
              onClick={() => setFamilyFilter(f.id)}
              className={`px-2.5 py-1 rounded-lg font-medium transition-all cursor-pointer ${
                familyFilter === f.id
                  ? 'bg-white dark:bg-slate-900 text-slate-900 dark:text-slate-100 shadow-2xs font-semibold'
                  : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-300'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Top Banner: Best Validation vs Best Test */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-4 rounded-xl bg-teal-50/70 dark:bg-teal-950/40 border border-teal-200 dark:border-teal-800/60 flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-teal-600 text-white flex items-center justify-center shrink-0">
            <Award className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-teal-700 dark:text-teal-300">
              Top Validation Macro-F1 Model
            </span>
            <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 mt-0.5">
              {metrics?.best_validation_model || 'XGBoost (Embeddings + Latent AI Gen)'}
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Val Macro-F1: <strong>87.68%</strong> | Latent class manifold synthesis achieved highest validation performance.
            </p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-sky-50/70 dark:bg-sky-950/40 border border-sky-200 dark:border-sky-800/60 flex items-start gap-3">
          <div className="w-9 h-9 rounded-lg bg-sky-600 text-white flex items-center justify-center shrink-0">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] font-bold uppercase tracking-wider text-sky-700 dark:text-sky-300">
              Top Test Macro-F1 Model
            </span>
            <h4 className="text-sm font-bold text-slate-900 dark:text-slate-100 mt-0.5">
              {metrics?.best_test_macro_f1_model || 'SVM (Embeddings + Latent AI Gen)'}
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-1">
              Test Accuracy: <strong>85.50%</strong> | Test Macro-F1: <strong>0.8661</strong> | 100% Osteopenia sensitivity.
            </p>
          </div>
        </div>
      </div>

      {/* Main Sortable Table */}
      <Card>
        <CardHeader
          title={`Benchmark Table (${sortedModels.length} Models Recorded)`}
          subtitle="Click column header to sort by metric. Validation metrics strictly separated from test set evaluation."
        />
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 dark:bg-slate-850 text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800 font-semibold uppercase tracking-wider text-[11px]">
              <tr>
                <th className="p-3.5">Architecture</th>
                <th className="p-3.5">Family</th>
                <th
                  onClick={() => handleSort('val_macro_f1')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>Val F1</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_macro_f1')}
                  className="p-3.5 cursor-pointer hover:text-teal-600 text-teal-700 dark:text-teal-300"
                >
                  <div className="flex items-center gap-1 font-bold">
                    <span>Test F1</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('test_accuracy')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>Test Acc</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('osteoporosis_recall')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>OS Recall</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('osteopenia_recall')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>OP Recall</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('normal_recall')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>Norm Recall</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
                <th
                  onClick={() => handleSort('inference_latency_ms')}
                  className="p-3.5 cursor-pointer hover:text-teal-600"
                >
                  <div className="flex items-center gap-1">
                    <span>Latency</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {loading ? (
                <tr>
                  <td colSpan={9} className="p-8 text-center text-slate-400">
                    Loading benchmark report...
                  </td>
                </tr>
              ) : (
                sortedModels.map((row) => (
                  <tr
                    key={row.experiment_id}
                    className="hover:bg-slate-50/80 dark:hover:bg-slate-850/50 transition-colors"
                  >
                    <td className="p-3.5 font-medium text-slate-900 dark:text-slate-100">
                      {row.model_name}
                    </td>
                    <td className="p-3.5">
                      <span className="text-[11px] px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400">
                        {row.model_family}
                      </span>
                    </td>
                    <td className="p-3.5 font-mono text-slate-700 dark:text-slate-300">
                      {row.val_macro_f1 ? (row.val_macro_f1 * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono font-bold text-teal-600 dark:text-teal-400">
                      {row.test_macro_f1 ? (row.test_macro_f1 * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono text-slate-800 dark:text-slate-200">
                      {row.test_accuracy ? (row.test_accuracy * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono text-rose-600 dark:text-rose-400">
                      {row.osteoporosis_recall ? (row.osteoporosis_recall * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono text-amber-600 dark:text-amber-400">
                      {row.osteopenia_recall ? (row.osteopenia_recall * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono text-emerald-600 dark:text-emerald-400">
                      {row.normal_recall ? (row.normal_recall * 100).toFixed(1) + '%' : '—'}
                    </td>
                    <td className="p-3.5 font-mono text-slate-500">
                      {row.inference_latency_ms ? `${row.inference_latency_ms.toFixed(2)} ms` : '—'}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Charts & Confusion Matrix Row */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Per-Class Recall Sensitivity Bar Chart */}
        <Card className="lg:col-span-7">
          <CardHeader
            title="Per-Class Sensitivity by Architecture"
            subtitle="Comparing detection rates for Normal, Osteopenia, and Osteoporosis"
          />
          <CardContent className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={perClassChartData} margin={{ top: 15, right: 15, left: -10, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-15} textAnchor="end" />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
                <Tooltip formatter={(val: any) => [`${val}%`, '']} contentStyle={{ borderRadius: '8px', fontSize: '12px' }} />
                <Legend verticalAlign="top" height={36} iconSize={10} wrapperStyle={{ fontSize: '12px' }} />
                <Bar dataKey="Normal" fill="#10b981" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Osteopenia" fill="#f59e0b" radius={[3, 3, 0, 0]} />
                <Bar dataKey="Osteoporosis" fill="#f43f5e" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Interactive Confusion Matrix Viewer */}
        <Card className="lg:col-span-5">
          <CardHeader
            title="Confusion Matrix Inspector"
            subtitle="Actual test predictions on 131 untouched radiographs"
            action={
              <select
                value={selectedConfusionModel}
                onChange={(e) => setSelectedConfusionModel(e.target.value)}
                className="text-xs bg-slate-50 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-md px-2 py-1 max-w-[140px]"
              >
                {metrics?.models.map((m) => (
                  <option key={m.experiment_id} value={m.experiment_id}>
                    {m.model_name}
                  </option>
                ))}
              </select>
            }
          />
          <CardContent className="space-y-4">
            <div className="overflow-hidden border border-slate-200 dark:border-slate-800 rounded-xl">
              <div className="grid grid-cols-4 text-center text-xs font-semibold bg-slate-50 dark:bg-slate-850 p-2.5 border-b border-slate-200 dark:border-slate-800">
                <span className="text-slate-400">True \ Pred</span>
                <span className="text-emerald-600">Normal</span>
                <span className="text-amber-600">Osteopenia</span>
                <span className="text-rose-600">Osteoporosis</span>
              </div>

              {['Normal', 'Osteopenia', 'Osteoporosis'].map((trueCls, rowIdx) => (
                <div key={trueCls} className="grid grid-cols-4 text-center text-xs p-2.5 border-b last:border-b-0 border-slate-100 dark:border-slate-800/80 items-center">
                  <span className="font-semibold text-slate-700 dark:text-slate-300 text-left pl-2">
                    {trueCls}
                  </span>
                  {[0, 1, 2].map((colIdx) => {
                    const count = confusionMatrix[rowIdx]?.[colIdx] ?? 0;
                    const isDiagonal = rowIdx === colIdx;
                    return (
                      <span
                        key={colIdx}
                        className={`font-mono py-1 rounded ${
                          isDiagonal
                            ? 'bg-teal-50 dark:bg-teal-950 text-teal-700 dark:text-teal-300 font-bold'
                            : count > 0
                            ? 'text-slate-600 dark:text-slate-400'
                            : 'text-slate-300 dark:text-slate-600'
                        }`}
                      >
                        {count}
                      </span>
                    );
                  })}
                </div>
              ))}
            </div>

            {selectedModelRecord && (
              <div className="p-3 rounded-lg bg-slate-50 dark:bg-slate-850 text-xs text-slate-600 dark:text-slate-400 space-y-1">
                <div className="flex justify-between">
                  <span>Selected Model:</span>
                  <span className="font-semibold text-slate-800 dark:text-slate-200">
                    {selectedModelRecord.model_name}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Test Accuracy:</span>
                  <span className="font-mono text-teal-600 font-medium">
                    {selectedModelRecord.test_accuracy ? (selectedModelRecord.test_accuracy * 100).toFixed(1) + '%' : '—'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Test Macro-F1:</span>
                  <span className="font-mono text-teal-600 font-medium">
                    {selectedModelRecord.test_macro_f1 ? (selectedModelRecord.test_macro_f1 * 100).toFixed(1) + '%' : '—'}
                  </span>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
