import { useEffect } from 'react';
import { useDatasetStore } from '@/state/datasetStore';

export default function DatasetTable() {
  const { images, refreshImages, loading } = useDatasetStore();

  useEffect(() => {
    refreshImages();
  }, [refreshImages]);

  return (
    <div>
      <div className="p-3 flex items-center justify-between">
        <h3 className="font-semibold text-sm">Dataset</h3>
        <button className="text-xs px-2 py-1 rounded-md bg-white/10 hover:bg-white/15" onClick={refreshImages}>
          Refresh
        </button>
      </div>
      <div className="overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs uppercase text-slate-300">
            <tr className="border-y border-white/10">
              <th className="px-3 py-2">ID</th>
              <th className="px-3 py-2">Image Name</th>
              <th className="px-3 py-2">Preview</th>
              <th className="px-3 py-2">Annotated By</th>
              <th className="px-3 py-2">Completed</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr><td className="px-3 py-3" colSpan={5}>Loading...</td></tr>
            )}
            {!loading && images.length === 0 && (
              <tr><td className="px-3 py-3" colSpan={5}>No images yet</td></tr>
            )}
            {images.map((img) => (
              <tr key={img._id} className="border-b border-white/5 hover:bg-white/5">
                <td className="px-3 py-2 text-xs opacity-70">{img._id.slice(-6)}</td>
                <td className="px-3 py-2">{img.name}</td>
                <td className="px-3 py-2">
                  {img.url ? (
                    <img src={img.url} alt={img.name} className="h-10 w-16 object-cover rounded" />
                  ) : (
                    <span className="text-xs opacity-60">(no preview)</span>
                  )}
                </td>
                <td className="px-3 py-2">{img.annotatedBy || '-'}</td>
                <td className="px-3 py-2">
                  <span className={`px-2 py-0.5 rounded text-xs ${img.completed ? 'bg-green-600/30 text-green-300' : 'bg-slate-700/50 text-slate-300'}`}>
                    {img.completed ? 'Yes' : 'No'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}


