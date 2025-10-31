import type { InferenceResult } from '@/types/results';

export function downloadResultsCsv(results: InferenceResult[]) {
  const headers = ['Image', 'loss_box', 'loss_dfl', 'loss_class', 'Most Impactful Feature', 'Impact %'];
  const rows = results.map(r => [
    escapeCsv(r.image ?? ''),
    numberToFixed(r.loss_box, 4),
    numberToFixed(r.loss_dfl, 4),
    numberToFixed(r.loss_class, 4),
    escapeCsv(r.most_impactful_feature ?? ''),
    numberToFixed(r.impact_percent, 1),
  ]);
  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'results.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function escapeCsv(value?: string) {
  const v = value ?? '';
  if (v.includes(',') || v.includes('"') || v.includes('\n')) {
    return '"' + v.replace(/"/g, '""') + '"';
  }
  return v;
}

function numberToFixed(value: unknown, digits: number) {
  const n = typeof value === 'number' && isFinite(value) ? value : 0;
  return n.toFixed(digits);
}






