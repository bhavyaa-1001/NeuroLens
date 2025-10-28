import type { InferenceResult } from '@/types/results';

export function downloadResultsCsv(results: InferenceResult[]) {
  const headers = ['Image Name', 'Predicted Loss', 'Category', 'Parametric Description'];
  const rows = results.map(r => [
    escapeCsv(r.imageName),
    r.predictedLoss.toString(),
    r.category.toString(),
    escapeCsv(r.parametricDescription),
  ]);
  const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'neurolens_results.csv';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function escapeCsv(value: string) {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return '"' + value.replace(/"/g, '""') + '"';
  }
  return value;
}


