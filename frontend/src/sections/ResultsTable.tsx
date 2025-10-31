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

      <div className="overflow-x-auto max-h-96 overflow-y-auto">
        <table className="w-full text-sm table-fixed border-collapse">
          <thead className="text-left text-xs uppercase text-slate-300">
            <tr className="border-y border-white/10">
              <th className="px-3 py-2 w-72">Preview</th>
              <th className="px-3 py-2 w-20">loss_box</th>
              <th className="px-3 py-2 w-20">loss_dfl</th>
              <th className="px-3 py-2 w-20">loss_class</th>
              <th className="px-3 py-2 w-48">Most Impactful Feature</th>
              <th className="px-3 py-2 w-20">Impact %</th>
            </tr>
          </thead>

          <tbody>
            {results.length === 0 ? (
              <tr>
                <td className="px-3 py-3 text-center" colSpan={6}>
                  No results yet
                </td>
              </tr>
            ) : (
              results.map((r) => (
                <tr
                  key={r.imageName}
                  className="border-b border-white/5 hover:bg-white/5 align-top"
                >
                  {/* Preview column */}
                  <td className="px-3 py-2 max-w-[18rem]">
                    <div
                      className="truncate overflow-hidden text-ellipsis whitespace-nowrap block"
                      title={r.image}
                    >
                      {r.image?.length > 35
                        ? `${r.image.slice(0, 20)}...${r.image.slice(-10)}`
                        : r.image}
                    </div>
                    {r.imageUrl ? (
                      <img
                        src={r.imageUrl}
                        alt={r.image}
                        className="mt-2 h-12 w-16 object-cover rounded"
                      />
                    ) : (
                      <span className="block text-xs opacity-60 mt-1">
                        (no preview)
                      </span>
                    )}
                  </td>

                  <td className="px-3 py-2 text-center font-mono whitespace-nowrap">
                    {typeof r.loss_box === 'number'
                      ? r.loss_box.toFixed(4)
                      : '0.0000'}
                  </td>
                  <td className="px-3 py-2 text-center font-mono whitespace-nowrap">
                    {typeof r.loss_dfl === 'number'
                      ? r.loss_dfl.toFixed(4)
                      : '0.0000'}
                  </td>
                  <td className="px-3 py-2 text-center font-mono whitespace-nowrap">
                    {typeof r.loss_class === 'number'
                      ? r.loss_class.toFixed(4)
                      : '0.0000'}
                  </td>

                  <td className="px-3 py-2">
                    <div
                      className="truncate whitespace-nowrap"
                      title={r.most_impactful_feature}
                    >
                      {r.most_impactful_feature}
                    </div>
                  </td>

                  <td className="px-3 py-2 text-center font-mono whitespace-nowrap">
                    {typeof r.impact_percent === 'number'
                      ? r.impact_percent.toFixed(1)
                      : '-'}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}