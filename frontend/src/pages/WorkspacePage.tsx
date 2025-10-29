import InputsPanel from '@/sections/InputsPanel';
import EnginePanel from '@/sections/EnginePanel';
import ResultsPanel from '@/sections/ResultsPanel';

type Props = {
  sidebarOpen: boolean;
};

export default function WorkspacePage({ sidebarOpen }: Props) {
  return (
    <div className="grid grid-cols-12 gap-4">
      <section className={`col-span-12 lg:col-span-3 transition-all ${sidebarOpen ? '' : 'lg:col-span-1'}`}>
        <InputsPanel compact={!sidebarOpen} />
      </section>

      <section className="col-span-12 lg:col-span-5">
        <EnginePanel />
      </section>

      <section className="col-span-12 lg:col-span-4">
        <ResultsPanel />
      </section>
    </div>
  );
}


