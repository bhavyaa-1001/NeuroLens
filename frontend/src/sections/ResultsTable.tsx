import { useResultsStore } from '@/state/resultsStore';
import { downloadResultsCsv } from '@/utils/csv';

export default function ResultsTable() {
  const { results } = useResultsStore();

  return (
    <div>
      <div className="p-3 flex items-center justify-between">
        <h3 className="font-semibold text-sm">Results</h3>
        <button
          className="text-xs px-2 py-1 rounded-md bg-white/10 hover:bg-white/15"
          onClick={() => downloadResultsCsv(results)}
          disabled={!results.length}
        >
          Download CSV
        </button>
      </div>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-slate-300">
            <tr className="border-y border-white/10">
              <th className="px-3 py-2">Image Name</th>
              <th className="px-3 py-2">Image</th>
              <th className="px-3 py-2">Predicted Loss</th>
              <th className="px-3 py-2">Category</th>
              <th className="px-3 py-2">Parametric Description</th>
            </tr>
          </thead>
          <tbody>
            {!results.length && (
              <tr><td className="px-3 py-3" colSpan={5}>No results yet</td></tr>
            )}
            {results.map((r) => (
              <tr key={r.imageName} className="border-b border-white/5 hover:bg-white/5">
                <td className="px-3 py-2">{r.imageName}</td>
                <td className="px-3 py-2">
                  {r.imageUrl ? (
                    <img src={r.imageUrl} alt={r.imageName} className="h-10 w-16 object-cover rounded" />
                  ) : (
                    <span className="text-xs opacity-60">(no preview)</span>
                  )}
                </td>
                <td className="px-3 py-2">{r.predictedLoss.toFixed(4)}</td>
                <td className="px-3 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs bg-primary-700/40`}>{r.category}</span>
                </td>
                <td className="px-3 py-2 max-w-[320px] truncate" title={r.parametricDescription}>{r.parametricDescription}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


