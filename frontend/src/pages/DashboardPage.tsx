import { useEffect, useMemo, useState } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, LineChart, Line, XAxis, YAxis, CartesianGrid } from 'recharts';
import { useResultsStore } from '@/state/resultsStore';

const COLORS = ['#6366f1', '#7c3aed', '#22c55e', '#f59e0b'];

export default function DashboardPage() {
  const { results } = useResultsStore();
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

  const categoryData = useMemo(() => {
    const counts: Record<number, number> = { 1: 0, 2: 0, 3: 0, 4: 0 };
    results.forEach(r => { counts[r.category] = (counts[r.category] || 0) + 1; });
    return Object.entries(counts).map(([k, v]) => ({ name: `Cat ${k}`, value: v }));
  }, [results]);

  const lossSeries = useMemo(() => {
    return results.map((r, idx) => ({ index: idx + 1, loss: r.predictedLoss }));
  }, [results]);

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-5 glass rounded-xl p-4">
        <h2 className="font-semibold mb-4">Category Distribution</h2>
        <div className="h-72">
          <ResponsiveContainer>
            <PieChart>
              <Pie data={categoryData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={100}>
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="col-span-12 lg:col-span-7 glass rounded-xl p-4">
        <h2 className="font-semibold mb-4">Predicted Loss Trend</h2>
        <div className="h-72">
          <ResponsiveContainer>
            <LineChart data={lossSeries} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="index" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip />
              <Line type="monotone" dataKey="loss" stroke="#7c3aed" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}


