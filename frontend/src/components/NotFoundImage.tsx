export default function NotFoundImage({ label = 'no preview' }: { label?: string }) {
  return (
    <div className="h-10 w-16 rounded bg-slate-800 text-[10px] flex items-center justify-center opacity-60">
      {label}
    </div>
  );
}









