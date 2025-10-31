import { useEffect, useMemo, useState } from 'react';
import { Link, Route, Routes } from 'react-router-dom';
import DashboardPage from '@/pages/DashboardPage';
import WorkspacePage from '@/pages/WorkspacePage';
import { resetDataset } from '@/services/nodeApi';
import { useDatasetStore } from '@/state/datasetStore';
import { useResultsStore } from '@/state/resultsStore';

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const navItems = useMemo(() => ([
    { path: '/', label: 'Workspace' },
    { path: '/dashboard', label: 'Dashboard' },
  ]), []);

  return (
    <div className="min-h-screen text-slate-200 bg-gradient-to-b from-primary-950 to-slate-950">
      {/* Clear dataset on each app mount/refresh */}
      <AppBootstraps />
      <header className="sticky top-0 z-40 border-b border-white/10 bg-slate-950/70 backdrop-blur">
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center gap-4">
          <button
            className="rounded-md px-2 py-1 bg-primary-700/40 hover:bg-primary-700/60"
            onClick={() => setSidebarOpen((v) => !v)}
            aria-label="Toggle sidebar"
          >
            ☰
          </button>
          <div className="font-semibold tracking-wide">
            <span className="text-primary-300">Neuro</span>
            <span className="text-accent-500">Lens</span>
            <span className="ml-2 text-xs opacity-70">The Visual Data Refiner</span>
          </div>
          <nav className="ml-auto flex gap-3 text-sm">
            {navItems.map((n) => (
              <Link key={n.path} to={n.path} className="px-3 py-1 rounded-md hover:bg-white/10">
                {n.label}
              </Link>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-6">
        <Routes>
          <Route path="/" element={<WorkspacePage sidebarOpen={sidebarOpen} />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>
    </div>
  );
}




function AppBootstraps() {
  const { refreshImages } = useDatasetStore();
  const { clear } = useResultsStore();
  useEffect(() => {
    // Clear backend dataset and then refresh UI + clear results
    resetDataset()
      .then(() => {
        clear();
        return refreshImages();
      })
      .catch(() => {});
  }, [refreshImages, clear]);
  return null;
}
