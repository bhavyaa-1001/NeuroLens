import DatasetTable from '@/sections/DatasetTable';
import UploadArea from '@/sections/UploadArea';

export default function EnginePanel() {
  return (
    <div className="space-y-4">
      <div className="glass rounded-xl p-4">
        <h2 className="text-sm font-semibold text-primary-200 mb-2">Model Engine</h2>
        <p className="text-xs text-slate-300">Upload your dataset zip. Files will be listed below.</p>
        <div className="mt-3">
          <UploadArea />
        </div>
      </div>

      <div className="glass rounded-xl p-2">
        <DatasetTable />
      </div>
    </div>
  );
}



