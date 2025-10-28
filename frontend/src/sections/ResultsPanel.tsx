import ResultsTable from '@/sections/ResultsTable';
import RunModel from '@/sections/RunModel';

export default function ResultsPanel() {
  return (
    <div className="space-y-4">
      <div className="glass rounded-xl p-4">
        <RunModel />
      </div>
      <div className="glass rounded-xl p-2">
        <ResultsTable />
      </div>
    </div>
  );
}


