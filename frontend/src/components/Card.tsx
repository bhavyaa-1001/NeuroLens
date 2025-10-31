import { ReactNode } from 'react';

export default function Card({ title, children, actions }: { title?: string; children: ReactNode; actions?: ReactNode }) {
  return (
    <div className="glass rounded-xl">
      {(title || actions) && (
        <div className="p-3 flex items-center justify-between border-b border-white/10">
          <h3 className="font-semibold text-sm">{title}</h3>
          {actions}
        </div>
      )}
      <div className="p-3">
        {children}
      </div>
    </div>
  );
}









